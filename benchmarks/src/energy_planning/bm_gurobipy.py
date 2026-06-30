"""GurobiPy implementation of the energy planning benchmark."""

import types
from pathlib import Path

import gurobipy as gp
import polars as pl
from benchmark_utils.gurobipy import Benchmark


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

        m = gp.Model()

        container = types.SimpleNamespace()
        self.container = container

        gens = pl.read_parquet("generators.parquet")
        lines = pl.read_parquet("lines_simplified.parquet")
        loads_df = pl.read_parquet("loads.parquet")
        capex = pl.read_csv("capex_costs.csv")
        cost_params = pl.read_csv("cost_parameters.csv")

        COST_UNSERVED_LOAD = cost_params.filter(name="load_unserved_MWh")["cost"].item()
        SLACK_BUS = 1

        hours = loads_df.get_column("datetime").unique().sort()
        if self.size is not None:
            hours = hours.head(self.size)
            loads_df = loads_df.filter(pl.col("datetime").is_in(hours.implode()))

        container.G = gens["gen_id"].to_list()
        container.L = lines["line_id"].to_list()
        container.T = hours.to_list()
        container.B = sorted(
            set(lines["from_bus"].to_list() + lines["to_bus"].to_list())
        )
        container.LOADS = set(zip(loads_df["bus"], loads_df["datetime"]))

        container.gen_bus = dict(zip(gens["gen_id"], gens["bus"]))
        container.gen_type = dict(zip(gens["gen_id"], gens["type"]))
        container.gen_pmax = dict(zip(gens["gen_id"], gens["Pmax"]))
        container.gen_cost = dict(zip(gens["gen_id"], gens["cost_per_MWh_linear"]))
        container.gen_overhead = dict(
            zip(gens["gen_id"], gens["hourly_overhead_per_MW_capacity"])
        )
        container.line_from = dict(zip(lines["line_id"], lines["from_bus"]))
        container.line_to = dict(zip(lines["line_id"], lines["to_bus"]))
        container.line_rating = dict(zip(lines["line_id"], lines["line_rating_MW"]))
        container.susceptance = {
            lid: 1 / x for lid, x in zip(lines["line_id"], lines["reactance"])
        }
        container.load = dict(
            zip(zip(loads_df["bus"], loads_df["datetime"]), loads_df["active_load"])
        )
        container.capex = dict(zip(capex["type"], capex["yearly_capex_cost_per_KW"]))

        container.gens_at_bus = {
            b: [g for g in container.G if container.gen_bus[g] == b]
            for b in container.B
        }
        container.lines_to_bus = {
            b: [l for l in container.L if container.line_to[l] == b]
            for b in container.B
        }
        container.lines_from_bus = {
            b: [l for l in container.L if container.line_from[l] == b]
            for b in container.B
        }

        if capacity_expansion:
            container.Build_Out = m.addVars(
                container.G,
                ub={g: container.gen_pmax[g] for g in container.G},
                name="Build_Out",
            )
            container.Dispatch = m.addVars(container.G, container.T, name="Dispatch")
            container.Dispatch_ub = m.addConstrs(
                (
                    container.Dispatch[g, t] <= container.Build_Out[g]
                    for g in container.G
                    for t in container.T
                ),
                name="Dispatch_ub",
            )
        else:
            container.Dispatch = m.addVars(
                container.G,
                container.T,
                ub={
                    (g, t): container.gen_pmax[g]
                    for g in container.G
                    for t in container.T
                },
                name="Dispatch",
            )

        container.Voltage_Angle = m.addVars(
            container.B,
            container.T,
            name="Voltage_Angle",
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
        )
        container.Power_Flow = m.addVars(
            container.L,
            container.T,
            name="Power_Flow",
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
        )
        container.Power_Flow_ub = m.addConstrs(
            (
                (container.Power_Flow[l, t] <= container.line_rating[l])
                for l in container.L
                for t in container.T
            ),
            name="Power_Flow_ub",
        )
        container.Power_Flow_lb = m.addConstrs(
            (
                (container.Power_Flow[l, t] >= -container.line_rating[l])
                for l in container.L
                for t in container.T
            ),
            name="Power_Flow_lb",
        )
        container.Load_Unserved = m.addVars(
            container.LOADS,
            ub={(b, t): container.load[(b, t)] for b, t in container.LOADS},
            name="Load_Unserved",
        )

        container.Con_Slack_Bus = m.addConstrs(
            (container.Voltage_Angle[SLACK_BUS, t] == 0 for t in container.T),
            name="Con_Slack_Bus",
        )

        container.Con_Power_Flow = m.addConstrs(
            (
                (
                    container.Power_Flow[l, t]
                    == container.susceptance[l]
                    * (
                        container.Voltage_Angle[container.line_to[l], t]
                        - container.Voltage_Angle[container.line_from[l], t]
                    )
                )
                for l in container.L
                for t in container.T
            ),
            name="Con_Power_Flow",
        )

        container.Con_Power_Balance = m.addConstrs(
            (
                container.load.get((b, t), 0)
                == gp.quicksum(
                    container.Dispatch[g, t] for g in container.gens_at_bus[b]
                )
                + gp.quicksum(
                    container.Power_Flow[l, t] for l in container.lines_to_bus[b]
                )
                - gp.quicksum(
                    container.Power_Flow[l, t] for l in container.lines_from_bus[b]
                )
                + container.Load_Unserved.get((b, t), 0)
                for b in container.B
                for t in container.T
            ),
            name="Con_Power_Balance",
        )

        objective = (
            COST_UNSERVED_LOAD * container.Load_Unserved.sum()
            + container.Dispatch.prod(
                {
                    (g, t): container.gen_cost[g]
                    for g in container.G
                    for t in container.T
                }
            )
        )
        if capacity_expansion:
            capex_coeffs = {
                g: container.capex[container.gen_type[g]]
                for g in container.G
                if container.gen_type[g] in container.capex
            }
            objective += container.Build_Out.prod(capex_coeffs)
            overhead_coeffs = {
                g: container.gen_overhead[g] * len(container.T) for g in container.G
            }
            objective += container.Build_Out.prod(overhead_coeffs)

        m.setObjective(objective, gp.GRB.MINIMIZE)

        if security_constrained:
            self.add_security_constraints(m, container)

        if yearly_limits:
            self.add_yearly_limits(m, container)

        if variable_capacity_factors:
            self.add_vcf(m, container)

        return m

    def add_security_constraints(self, model, container):
        bodf = pl.read_parquet("branch_outage_dist_facts.parquet").select(
            "outage_line_id", "affected_line_id", "factor"
        )
        bodf_dict = {(row[0], row[1]): row[2] for row in bodf.iter_rows()}

        container.Con_Contingency_Pos = model.addConstrs(
            (
                container.Power_Flow[aff, t]
                + bodf_dict[out, aff] * container.Power_Flow[out, t]
                <= container.line_rating[aff]
                for out, aff in bodf_dict.keys()
                for t in container.T
            ),
            name="Con_Contingency_Pos",
        )
        container.Con_Contingency_Neg = model.addConstrs(
            (
                -container.line_rating[aff]
                <= container.Power_Flow[aff, t]
                + bodf_dict[out, aff] * container.Power_Flow[out, t]
                for out, aff in bodf_dict.keys()
                for t in container.T
            ),
            name="Con_Contingency_Neg",
        )

    def add_vcf(self, model, container):
        vcf_df = pl.read_parquet("variable_capacity_factors.parquet")
        vcf_type_map = pl.read_csv("map_type_to_vcf_type.csv")
        type_to_vcf_type = dict(zip(vcf_type_map["type"], vcf_type_map["vcf_type"]))

        vcf_dict = {
            (row["vcf_type"], row["datetime"]): row["capacity_factor"]
            for row in vcf_df.iter_rows(named=True)
            if row["datetime"] in container.T
        }

        container.Con_Variable_Dispatch_Limit = model.addConstrs(
            (
                container.Dispatch[g, t]
                <= vcf_dict[type_to_vcf_type[container.gen_type[g]], t]
                * container.gen_pmax[g]
                for g in container.G
                for t in container.T
                if container.gen_type[g] in type_to_vcf_type
            ),
            name="Con_Variable_Dispatch_Limit",
        )

    def add_yearly_limits(self, model, container):
        yearly_limits_df = pl.read_parquet("yearly_limits.parquet")
        yearly_limit = dict(zip(yearly_limits_df["type"], yearly_limits_df["limit"]))

        container.Con_Yearly_Limits = model.addConstrs(
            (
                gp.quicksum(
                    container.Dispatch[g, t]
                    for g in container.G
                    for t in container.T
                    if container.gen_type[g] == cat
                )
                <= yearly_limit[cat] * len(container.T) / (24 * 365)
                for cat in yearly_limit.keys()
            ),
            name="Con_Yearly_Limits",
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

        power_flow = self._items_to_df(
            model, container.Power_Flow, "line_id", "datetime"
        )
        power_flow = power_flow.join(
            self._items_to_df(
                model, container.Power_Flow_ub, "line_id", "datetime", dual=True
            ).rename({"dual": "ub_dual"}),
            on=power_flow_keys,
        ).join(
            self._items_to_df(
                model, container.Power_Flow_lb, "line_id", "datetime", dual=True
            ).rename({"dual": "lb_dual"}),
            on=power_flow_keys,
        )
        power_flow.write_parquet("power_flow.parquet")

        if capacity_expansion:
            self._items_to_df(model, container.Build_Out, "gen_id").write_parquet(
                "build_out.parquet"
            )

        self._items_to_df(
            model, container.Dispatch, "gen_id", "datetime"
        ).write_parquet("dispatch.parquet")
        self._items_to_df(
            model, container.Load_Unserved, "bus", "datetime"
        ).write_parquet("load_unserved.parquet")
        self._items_to_df(
            model, container.Con_Power_Balance, "bus", "datetime", dual=True
        ).write_parquet("power_balance_duals.parquet")

    def _items_to_df(self, model, tdict, *key_names, dual=False):
        rows = []
        attr = (
            gp.GRB.Attr.BarPi
            if dual and model.Params.Method == 2 and model.Params.Crossover == 0
            else gp.GRB.Attr.Pi
        )
        for key, obj in tdict.items():
            if not isinstance(key, tuple):
                key = (key,)
            value = obj.getAttr(attr) if dual else obj.getAttr(gp.GRB.Attr.X)
            rows.append((*key, value))

        return pl.DataFrame(
            rows,
            schema=list(key_names) + ["dual" if dual else "solution"],
            orient="row",
        )


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    benchmark = Bench(
        size=2,
        input_dir=base_dir / "model_data",
        results_dir=base_dir / "results_gurobipy",
        security_constrained=False,
        capacity_expansion=False,
        yearly_limits=False,
        variable_capacity_factors=False,
    )
    model = benchmark.run()
