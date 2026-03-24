import types

import polars as pl
import pyoptinterface as poi
from benchmark_utils.pyoptinterface import Benchmark


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

        model = super().build(**kwargs)

        container = types.SimpleNamespace()
        self.container = container

        ## LOAD DATA
        gens = pl.read_parquet("generators.parquet")
        lines = pl.read_parquet("lines_simplified.parquet")
        loads_df = pl.read_parquet("loads.parquet")
        capex = pl.read_csv("capex_costs.csv")
        cost_params = pl.read_csv("cost_parameters.csv")

        BASE_MW = 100
        COST_UNSERVED_LOAD = cost_params.filter(name="load_unserved_MWh")["cost"].item()
        SLACK_BUS = 1

        # shrink time horizon
        hours = loads_df.get_column("datetime").unique().sort()
        if self.size is not None:
            hours = hours.head(self.size)
            loads_df = loads_df.filter(pl.col("datetime").is_in(hours.implode()))

        # -------------------------
        # PARAMETERS
        # -------------------------
        container.gen_bus = poi.tupledict(dict(zip(gens["gen_id"], gens["bus"])))
        container.gen_type = poi.tupledict(dict(zip(gens["gen_id"], gens["type"])))
        container.gen_pmax = poi.tupledict(dict(zip(gens["gen_id"], gens["Pmax"])))
        container.gen_cost = poi.tupledict(
            dict(zip(gens["gen_id"], gens["cost_per_MWh_linear"]))
        )
        container.gen_overhead = poi.tupledict(
            zip(gens["gen_id"], gens["hourly_overhead_per_MW_capacity"])
        )
        container.line_from = poi.tupledict(
            dict(zip(lines["line_id"], lines["from_bus"]))
        )
        container.line_to = poi.tupledict(dict(zip(lines["line_id"], lines["to_bus"])))
        container.line_rating = poi.tupledict(
            dict(zip(lines["line_id"], lines["line_rating_MW"]))
        )
        container.susceptance = poi.tupledict(
            {lid: 1 / x for lid, x in zip(lines["line_id"], lines["reactance"])}
        )
        container.load = poi.tupledict(
            zip(zip(loads_df["bus"], loads_df["datetime"]), loads_df["active_load"])
        )
        container.capex = poi.tupledict(
            zip(capex["type"], capex["yearly_capex_cost_per_KW"])
        )

        # -------------------------
        # SETS
        # -------------------------
        container.G = gens["gen_id"]
        container.L = lines["line_id"]
        container.T = hours

        container.B = pl.concat([lines["from_bus"], lines["to_bus"]]).unique()

        container.LOADS = container.load.keys()

        # -------------------------
        # VARIABLES
        # -------------------------
        if capacity_expansion:
            container.Build_Out = poi.make_tupledict(
                container.G,
                rule=lambda g: model.add_variable(lb=0, ub=container.gen_pmax[g]),
            )
            container.Dispatch = model.add_variables(container.G, container.T, lb=0)
            container.Dispatch_ub = poi.make_tupledict(
                container.G,
                container.T,
                rule=lambda g, t: model.add_linear_constraint(
                    container.Dispatch[g, t] <= container.Build_Out[g]
                ),
            )
        else:
            container.Dispatch = poi.make_tupledict(
                container.G,
                container.T,
                rule=lambda g, t: model.add_variable(lb=0, ub=container.gen_pmax[g]),
            )

        container.Voltage_Angle = model.add_variables(container.B, container.T)

        container.Power_Flow = model.add_variables(container.L, container.T)
        container.Power_Flow_ub = poi.make_tupledict(
            container.L,
            container.T,
            rule=lambda l, t: model.add_linear_constraint(
                container.Power_Flow[l, t] <= container.line_rating[l]
            ),
        )
        container.Power_Flow_lb = poi.make_tupledict(
            container.L,
            container.T,
            rule=lambda l, t: model.add_linear_constraint(
                container.Power_Flow[l, t] >= -container.line_rating[l]
            ),
        )

        container.Load_Unserved = poi.make_tupledict(
            container.LOADS,
            rule=lambda b, t: model.add_variable(lb=0, ub=container.load[b, t]),
        )

        # -------------------------
        # CONSTRAINTS
        # -------------------------
        # Slack bus
        container.Con_Slack_Bus = poi.make_tupledict(
            container.T,
            rule=lambda t: model.add_linear_constraint(
                container.Voltage_Angle[SLACK_BUS, t] == 0
            ),
        )

        container.Con_Power_Flow = poi.make_tupledict(
            container.L,
            container.T,
            rule=lambda l, t: model.add_linear_constraint(
                container.Power_Flow[l, t]
                == (
                    BASE_MW
                    * container.susceptance[l]
                    * (
                        container.Voltage_Angle[container.line_to[l], t]
                        - container.Voltage_Angle[container.line_from[l], t]
                    )
                )
            ),
        )

        container.Con_Power_Balance = poi.make_tupledict(
            container.B,
            container.T,
            rule=lambda b, t: model.add_linear_constraint(
                container.load.get((b, t), 0)
                == poi.quicksum(
                    container.Dispatch[g, t]
                    for g in container.G
                    if container.gen_bus[g] == b
                )
                + poi.quicksum(
                    container.Power_Flow[l, t]
                    for l in container.L
                    if container.line_to[l] == b
                )
                - poi.quicksum(
                    container.Power_Flow[l, t]
                    for l in container.L
                    if container.line_from[l] == b
                )
                + (container.Load_Unserved[b, t] if (b, t) in container.LOADS else 0)
            ),
        )

        container.Obj = model.set_objective(
            (
                poi.quicksum(
                    COST_UNSERVED_LOAD * container.Load_Unserved[b, t]
                    for b, t in container.LOADS
                )
                + poi.quicksum(
                    container.gen_cost[g] * container.Dispatch[g, t]
                    for g in container.G
                    for t in container.T
                )
                + (
                    (
                        poi.quicksum(
                            container.capex[container.gen_type[g]]
                            * container.Build_Out[g]
                            for g in container.G
                            if container.gen_type[g] in container.capex
                        )
                        + poi.quicksum(
                            container.gen_overhead[g]
                            * container.Build_Out[g]
                            * len(container.T)
                            for g in container.G
                        )
                    )
                    if capacity_expansion
                    else 0
                )
            ),
            poi.ObjectiveSense.Minimize,
        )

        if security_constrained:
            self.add_security_constraints(model, container)

        if yearly_limits:
            self.add_yearly_limits(model, container)

        if variable_capacity_factors:
            self.add_vcf(model, container)

        return model

    def add_security_constraints(self, model, container):
        bodf = pl.read_parquet("branch_outage_dist_facts.parquet").select(
            ["outage_line_id", "affected_line_id", "factor"]
        )
        bodf_dict = poi.tupledict(
            {(row[0], row[1]): row[2] for row in bodf.iter_rows()}
        )

        container.Con_Contingency_Pos = poi.make_tupledict(
            bodf_dict.keys(),
            container.T,
            rule=lambda out, aff, t: model.add_linear_constraint(
                container.Power_Flow[aff, t]
                + bodf_dict[out, aff] * container.Power_Flow[out, t]
                <= container.line_rating[aff]
            ),
        )

        container.Con_Contingency_Neg = poi.make_tupledict(
            bodf_dict.keys(),
            container.T,
            rule=lambda out, aff, t: model.add_linear_constraint(
                container.Power_Flow[aff, t]
                + bodf_dict[out, aff] * container.Power_Flow[out, t]
                >= -container.line_rating[aff]
            ),
        )

    def add_vcf(self, model, container):
        vcf_df = pl.read_parquet("variable_capacity_factors.parquet")
        vcf_type_map = pl.read_csv("map_type_to_vcf_type.csv")

        type_to_vcf_type = poi.tupledict(
            dict(zip(vcf_type_map["type"], vcf_type_map["vcf_type"]))
        )

        vcf_dict = {
            (row["vcf_type"], row["datetime"]): row["capacity_factor"]
            for row in vcf_df.iter_rows(named=True)
            if row["datetime"] in container.T
        }
        container.Con_Variable_Dispatch_Limit = poi.make_tupledict(
            container.G,
            container.T,
            rule=lambda g, t: model.add_linear_constraint(
                container.Dispatch[g, t]
                <= vcf_dict[type_to_vcf_type[container.gen_type[g]], t]
                * container.gen_pmax[g]
            )
            if container.gen_type[g] in type_to_vcf_type
            else None,
        )

    def add_yearly_limits(self, model, container):
        yearly_limits_df = pl.read_parquet("yearly_limits.parquet")
        yearly_limit = poi.tupledict(
            dict(zip(yearly_limits_df["type"], yearly_limits_df["limit"]))
        )

        container.Con_Yearly_Limits = poi.make_tupledict(
            yearly_limit.keys(),
            rule=lambda cat: model.add_linear_constraint(
                poi.quicksum(
                    container.Dispatch[g, t]
                    for g in container.G
                    for t in container.T
                    if container.gen_type[g] == cat
                )
                <= yearly_limit[cat] * len(container.T) / (24 * 365)
            ),
        )

    def write_results(
        self,
        model,
        capacity_expansion: bool = True,
        security_constrained: bool = True,
        **kwargs,
    ):
        container = self.container

        power_flow_keys = ["line_id", "datetime"]
        _tuple_dict_to_df(model, container.Power_Flow, *power_flow_keys).join(
            _tuple_dict_to_df(
                model, container.Power_Flow_ub, *power_flow_keys, dual=True
            ).rename({"dual": "ub_dual"}),
            on=power_flow_keys,
        ).join(
            _tuple_dict_to_df(
                model, container.Power_Flow_lb, *power_flow_keys, dual=True
            ).rename({"dual": "lb_dual"}),
            on=power_flow_keys,
        ).write_parquet("power_flow.parquet")

        if capacity_expansion:
            _tuple_dict_to_df(model, container.Build_Out, "gen_id").write_parquet(
                "build_out.parquet"
            )
        _tuple_dict_to_df(
            model, container.Dispatch, "gen_id", "datetime"
        ).write_parquet("dispatch.parquet")
        _tuple_dict_to_df(
            model, container.Load_Unserved, "bus", "datetime"
        ).write_parquet("load_unserved.parquet")

        _tuple_dict_to_df(
            model, container.Con_Power_Balance, "bus", "datetime", dual=True
        ).write_parquet("power_balance_duals.parquet")


def _tuple_dict_to_df(model, tdict, *key_names, dual=False):
    if dual:
        tdict = tdict.map(
            lambda x: model.get_constraint_attribute(x, poi.ConstraintAttribute.Dual)
        )
    else:
        tdict = tdict.map(model.get_value)

    if len(key_names) > 1:
        gen_expr = ((*key, value) for key, value in tdict.items())
    else:
        gen_expr = ((key, value) for key, value in tdict.items())
    return pl.DataFrame(
        gen_expr,
        orient="row",
        schema=list(key_names) + ["dual" if dual else "solution"],
    )


if __name__ == "__main__":
    from pathlib import Path

    base_dir = Path(__file__).parent
    benchmark = Bench(
        size=2,
        input_dir=base_dir / "model_data",
        results_dir=base_dir / "results_pyoptinterface",
        security_constrained=False,
        capacity_expansion=False,
        yearly_limits=False,
    )
    benchmark.run()
