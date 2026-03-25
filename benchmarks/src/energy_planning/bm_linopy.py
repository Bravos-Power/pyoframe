import types

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from benchmark_utils.linopy import Benchmark


class Bench(Benchmark):
    def build(
        self,
        capacity_expansion: bool = True,
        security_constrained: bool = True,
        yearly_limits: bool = True,
        variable_capacity_factors: bool = True,
        **kwargs,
    ):
        assert len(kwargs) == 0, f"Unexpected kwargs: {kwargs}"
        assert self.input_dir is not None, "Input directory must be specified."

        m = lp.Model()

        container = types.SimpleNamespace()
        self.container = container

        ## LOAD DATA
        container.gens = (
            pd.read_parquet("generators.parquet").set_index("gen_id").to_xarray()
        )
        lines = (
            pd.read_parquet("lines_simplified.parquet").set_index("line_id").to_xarray()
        )
        from_buses = lines["from_bus"].rename("bus")
        to_buses = lines["to_bus"].rename("bus")
        loads = (
            pd.read_parquet("loads.parquet")
            .set_index(["bus", "datetime"])["active_load"]
            .to_xarray()
        )
        capex = (
            pd.read_csv("capex_costs.csv")
            .set_index("type")["yearly_capex_cost_per_KW"]
            .to_xarray()
        )
        cost_params = pd.read_csv("cost_parameters.csv").set_index("name")["cost"]

        BASE_MW = 100
        COST_UNSERVED_LOAD = cost_params.loc["load_unserved_MWh"]
        SLACK_BUS = 1

        ## DEFINE SETS AND PARAMS ##
        buses = (
            xr.concat(
                [
                    from_buses.drop_vars("line_id").rename({"line_id": "bus"}),
                    to_buses.drop_vars("line_id").rename({"line_id": "bus"}),
                ],
                dim="bus",
            )
            .reset_coords()
            .drop_duplicates(...)
            .sortby("bus")
            .bus
        )

        # Shrink the time horizon to allow for different model sizes
        hours = loads["datetime"].drop_duplicates(...).sortby("datetime")
        if self.size is not None:
            hours = hours[: self.size]
            loads = loads.sel(datetime=hours)
        container.hours = hours

        container.line_rating = lines["line_rating_MW"]
        susceptance = 1 / lines["reactance"]

        ## DEFINE VARIABLES ##
        if capacity_expansion:
            container.Build_Out = m.add_variables(
                coords=[container.gens.gen_id],
                lower=0,
                upper=container.gens["Pmax"],
                name="Build_Out",
            )
            container.Dispatch = m.add_variables(
                coords=[container.gens.gen_id, container.hours],
                lower=0,
                name="Dispatch",
            )
            container.Dispatch_ub = m.add_constraints(
                container.Dispatch <= container.Build_Out, name="Dispatch_ub"
            )
        else:
            container.Dispatch = m.add_variables(
                coords=[container.gens.gen_id, container.hours],
                lower=0,
                upper=container.gens["Pmax"],
                name="Dispatch",
            )

        container.Voltage_Angle = m.add_variables(
            coords=[buses, container.hours], name="Voltage_Angle"
        )
        container.Power_Flow = m.add_variables(
            coords=[lines.line_id, container.hours], name="Power_Flow"
        )
        container.Power_Flow_lb = m.add_constraints(
            container.Power_Flow >= -container.line_rating, name="Power_Flow_lb"
        )
        container.Power_Flow_ub = m.add_constraints(
            container.Power_Flow <= container.line_rating, name="Power_Flow_ub"
        )
        container.Load_Unserved = m.add_variables(
            lower=0, upper=loads, name="Load_Unserved"
        )

        ## DEFINE CONSTRAINTS ##
        container.Con_Slack_Bus = m.add_constraints(
            container.Voltage_Angle.sel(bus=SLACK_BUS) == 0, name="Con_Slack_Bus"
        )

        container.Con_Power_Flow = m.add_constraints(
            container.Power_Flow
            == BASE_MW
            * susceptance
            * (
                container.Voltage_Angle.sel(bus=to_buses)
                - container.Voltage_Angle.sel(bus=from_buses)
            ),
            name="Con_Power_Flow",
        )

        container.Con_Power_Balance = m.add_constraints(
            0
            == (
                container.Dispatch.groupby(container.gens.bus).sum()
                - loads
                + container.Load_Unserved
                - container.Power_Flow.groupby(from_buses).sum()
                + container.Power_Flow.groupby(to_buses).sum()
            ),
            name="Con_Power_Balance",
        )

        ## SET OBJECTIVE ##
        m.add_objective(COST_UNSERVED_LOAD * container.Load_Unserved.sum())
        m.objective += (
            container.Dispatch * container.gens["cost_per_MWh_linear"]
        ).sum()

        if capacity_expansion:
            m.objective += (
                container.Build_Out.groupby(container.gens.type)
                .sum()
                .reindex(type=capex.type)
                * capex
            ).sum()
            m.objective += (
                container.Build_Out * container.gens["hourly_overhead_per_MW_capacity"]
            ).sum() * len(container.hours)

        if security_constrained:
            self.add_security_constraints(m, container)

        if yearly_limits:
            self.add_yearly_limits(m, container)

        if variable_capacity_factors:
            self.add_vcf(m, container)
        return m

    def add_security_constraints(self, m, container):
        bodf = pd.read_parquet("branch_outage_dist_facts.parquet").rename(
            columns={"affected_line_id": "line_id"}
        )

        # must use loop because the data structure is sparse!
        # otherwise will create way too many constraints
        for outage, bodf_outage in bodf.groupby("outage_line_id"):
            bodf_outage = bodf_outage.set_index("line_id")["factor"].to_xarray()
            extra_flow = container.Power_Flow.sel(line_id=outage) * bodf_outage
            base_flow = container.Power_Flow.sel(line_id=extra_flow.data.line_id)
            total_flow = base_flow + extra_flow

            rating_filtered = container.line_rating.sel(line_id=total_flow.data.line_id)

            container.Con_Security_lb = m.add_constraints(
                total_flow <= rating_filtered, name=f"Con_Security_lb_{outage}"
            )
            container.Con_Security_ub = m.add_constraints(
                total_flow >= -rating_filtered, name=f"Con_Security_ub_{outage}"
            )

    def add_vcf(self, m, container):
        vcf = pd.read_parquet("variable_capacity_factors.parquet")
        vcf = vcf.loc[vcf["datetime"].isin(container.hours.to_numpy())]
        vcf = vcf.set_index(["datetime", "vcf_type"])["capacity_factor"].to_xarray()

        vcf_type_to_type = (
            pd.read_csv("map_type_to_vcf_type.csv")
            .set_index("type")["vcf_type"]
            .to_xarray()
        )

        vcf_by_type = vcf.sel(vcf_type=vcf_type_to_type).drop_vars("vcf_type")

        gens_filtered = container.gens.type
        gens_filtered = gens_filtered.sel(
            gen_id=np.isin(gens_filtered.values, vcf_by_type.type.values)
        )

        vcf_by_gen = vcf_by_type.sel(type=gens_filtered).drop_vars("type")

        max_dispatch = vcf_by_gen * container.gens["Pmax"]

        filtered_dispatch = container.Dispatch.sel(
            gen_id=np.isin(
                container.Dispatch.data.gen_id.values, max_dispatch.gen_id.values
            )
        )

        container.Con_Variable_Dispatch_Limit = m.add_constraints(
            filtered_dispatch <= max_dispatch, name="Con_Variable_Dispatch_Limit"
        )

    def add_yearly_limits(self, m, container):
        yearly_limits = (
            pd.read_parquet("yearly_limits.parquet")
            .set_index("type")["limit"]
            .to_xarray()
        )
        yearly_limits *= len(container.hours) / (24 * 365)

        dispatch_by_type = (
            container.Dispatch.sum("datetime").groupby(container.gens.type).sum()
        )
        dispatch_by_type = dispatch_by_type.sel(
            type=np.isin(
                dispatch_by_type.data["type"].values, yearly_limits["type"].values
            )
        )

        container.Con_Yearly_Limits = m.add_constraints(
            dispatch_by_type <= yearly_limits, name="Con_Yearly_Limits"
        )

    def write_results(
        self,
        model,
        capacity_expansion: bool = True,
        security_constrained: bool = True,
        **kwargs,
    ):
        container = self.container
        pf = container.Power_Flow.solution.to_dataframe()

        pf = pf.join(
            container.Power_Flow_ub.dual.to_dataframe().rename(
                columns={"dual": "ub_dual"}
            ),
            on=["line_id", "datetime"],
        ).join(
            container.Power_Flow_lb.dual.to_dataframe().rename(
                columns={"dual": "lb_dual"}
            ),
            on=["line_id", "datetime"],
        )
        pf.to_parquet("power_flow.parquet")

        if capacity_expansion:
            container.Build_Out.solution.to_dataframe().to_parquet("build_out.parquet")
        container.Dispatch.solution.to_dataframe().to_parquet("dispatch.parquet")
        container.Load_Unserved.solution.to_dataframe().to_parquet(
            "load_unserved.parquet"
        )
        container.Con_Power_Balance.dual.to_dataframe().to_parquet(
            "power_balance_duals.parquet"
        )


if __name__ == "__main__":
    from pathlib import Path

    base_dir = Path(__file__).parent
    benchmark = Bench(
        size=2,
        input_dir=base_dir / "model_data",
        results_dir=base_dir / "results_linopy",
        capacity_expansion=True,
        security_constrained=False,
        yearly_limits=True,
        variable_capacity_factors=True,
    )
    m = benchmark.run()
