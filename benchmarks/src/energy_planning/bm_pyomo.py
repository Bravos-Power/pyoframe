from pathlib import Path

import polars as pl
import pyomo.environ as pyo
from benchmark_utils.pyomo import Benchmark


class Bench(Benchmark):
    def build(
        self,
        capacity_expansion: bool = True,
        security_constrained: bool = True,
        yearly_limits: bool = True,
        variable_capacity_factors: bool = True,
        **kwargs,
    ):
        m = pyo.ConcreteModel()
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        assert self.input_dir is not None, "Input directory must be specified."

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
        # SETS
        # -------------------------
        m.G = pyo.Set(initialize=gens["gen_id"].to_list())
        m.L = pyo.Set(initialize=lines["line_id"].to_list())
        m.T = pyo.Set(initialize=hours.to_list())

        m.B = pyo.Set(
            initialize=sorted(
                set(lines["from_bus"].to_list() + lines["to_bus"].to_list())
            )
        )

        # -------------------------
        # PARAMETERS
        # -------------------------
        m.gen_bus = dict(zip(gens["gen_id"], gens["bus"]))
        m.gen_type = dict(zip(gens["gen_id"], gens["type"]))
        m.gen_pmax = dict(zip(gens["gen_id"], gens["Pmax"]))
        m.gen_cost = dict(zip(gens["gen_id"], gens["cost_per_MWh_linear"]))
        m.gen_overhead = dict(
            zip(gens["gen_id"], gens["hourly_overhead_per_MW_capacity"])
        )
        m.line_from = dict(zip(lines["line_id"], lines["from_bus"]))
        m.line_to = dict(zip(lines["line_id"], lines["to_bus"]))
        m.line_rating = dict(zip(lines["line_id"], lines["line_rating_MW"]))
        m.susceptance = {
            lid: 1 / x for lid, x in zip(lines["line_id"], lines["reactance"])
        }
        m.load = dict(
            zip(zip(loads_df["bus"], loads_df["datetime"]), loads_df["active_load"])
        )
        m.capex = dict(zip(capex["type"], capex["yearly_capex_cost_per_KW"]))

        m.LOADS = pyo.Set(initialize=m.load.keys(), dimen=2)

        # -------------------------
        # VARIABLES
        # -------------------------
        if capacity_expansion:
            m.Build_Out = pyo.Var(m.G, bounds=lambda m, g: (0, m.gen_pmax[g]))
            m.Dispatch = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)
            m.Dispatch_ub = pyo.Constraint(
                m.G,
                m.T,
                rule=lambda m, g, t: m.Dispatch[g, t] <= m.Build_Out[g],
            )
        else:
            m.Dispatch = pyo.Var(m.G, m.T, bounds=lambda m, g, t: (0, m.gen_pmax[g]))

        m.Voltage_Angle = pyo.Var(m.B, m.T)

        m.Power_Flow = pyo.Var(m.L, m.T)
        m.Power_Flow_ub = pyo.Constraint(
            m.L, m.T, rule=lambda m, l, t: m.Power_Flow[l, t] <= m.line_rating[l]
        )
        m.Power_Flow_lb = pyo.Constraint(
            m.L, m.T, rule=lambda m, l, t: m.Power_Flow[l, t] >= -m.line_rating[l]
        )

        m.Load_Unserved = pyo.Var(m.LOADS, bounds=lambda m, b, t: (0, m.load[b, t]))

        # -------------------------
        # CONSTRAINTS
        # -------------------------
        # Slack bus
        m.Con_Slack_Bus = pyo.Constraint(
            m.T, rule=lambda m, t: m.Voltage_Angle[SLACK_BUS, t] == 0
        )

        m.Con_Power_Flow = pyo.Constraint(
            m.L,
            m.T,
            rule=lambda m, l, t: (
                m.Power_Flow[l, t]
                == (
                    BASE_MW
                    * m.susceptance[l]
                    * (
                        m.Voltage_Angle[m.line_to[l], t]
                        - m.Voltage_Angle[m.line_from[l], t]
                    )
                )
            ),
        )

        m.Con_Power_Balance = pyo.Constraint(
            m.B,
            m.T,
            rule=lambda m, b, t: (
                m.load.get((b, t), 0)
                == sum(m.Dispatch[g, t] for g in m.G if m.gen_bus[g] == b)
                + sum(m.Power_Flow[l, t] for l in m.L if m.line_to[l] == b)
                - sum(m.Power_Flow[l, t] for l in m.L if m.line_from[l] == b)
                + (m.Load_Unserved[b, t] if (b, t) in m.LOADS else 0)
            ),
        )

        m.Obj = pyo.Objective(
            sense=pyo.minimize,
            rule=lambda m: (
                sum(COST_UNSERVED_LOAD * m.Load_Unserved[b, t] for b, t in m.LOADS)
                + sum(m.gen_cost[g] * m.Dispatch[g, t] for g in m.G for t in m.T)
                + (
                    (
                        sum(
                            m.capex[m.gen_type[g]] * m.Build_Out[g]
                            for g in m.G
                            if m.gen_type[g] in m.capex
                        )
                        + sum(
                            m.gen_overhead[g] * m.Build_Out[g] * len(m.T) for g in m.G
                        )
                    )
                    if capacity_expansion
                    else 0
                )
            ),
        )

        if security_constrained:
            self.add_security_constraints(m)

        if yearly_limits:
            self.add_yearly_limits(m)

        if variable_capacity_factors:
            self.add_vcf(m)

        return m

    def add_security_constraints(self, m):
        bodf = pl.read_parquet("branch_outage_dist_facts.parquet").select(
            ["outage_line_id", "affected_line_id", "factor"]
        )
        bodf_dict = {(row[0], row[1]): row[2] for row in bodf.iter_rows()}

        m.CONTIG = pyo.Set(initialize=bodf_dict.keys(), dimen=2)

        m.Con_Contingency_Pos = pyo.Constraint(
            m.CONTIG,
            m.T,
            rule=lambda m, out, aff, t: (
                m.Power_Flow[aff, t] + bodf_dict[out, aff] * m.Power_Flow[out, t]
                <= m.line_rating[aff]
            ),
        )

        m.Con_Contingency_Neg = pyo.Constraint(
            m.CONTIG,
            m.T,
            rule=lambda m, out, aff, t: (
                m.Power_Flow[aff, t] + bodf_dict[out, aff] * m.Power_Flow[out, t]
                >= -m.line_rating[aff]
            ),
        )

    def add_vcf(self, m):
        vcf_df = pl.read_parquet("variable_capacity_factors.parquet")
        vcf_type_map = pl.read_csv("map_type_to_vcf_type.csv")

        type_to_vcf_type = dict(zip(vcf_type_map["type"], vcf_type_map["vcf_type"]))

        vcf_dict = {
            (row["vcf_type"], row["datetime"]): row["capacity_factor"]
            for row in vcf_df.iter_rows(named=True)
            if row["datetime"] in m.T
        }
        m.Con_Variable_Dispatch_Limit = pyo.Constraint(
            m.G,
            m.T,
            rule=lambda m, g, t: (
                m.Dispatch[g, t]
                <= vcf_dict[type_to_vcf_type[m.gen_type[g]], t] * m.gen_pmax[g]
            )
            if m.gen_type[g] in type_to_vcf_type
            else pyo.Constraint.Skip,
        )

    def add_yearly_limits(self, m):
        yearly_limits_df = pl.read_parquet("yearly_limits.parquet")
        yearly_limit = dict(zip(yearly_limits_df["type"], yearly_limits_df["limit"]))

        m.TYPES_WITH_LIMIT = pyo.Set(initialize=yearly_limit.keys())
        m.Con_Yearly_Limits = pyo.Constraint(
            m.TYPES_WITH_LIMIT,
            rule=lambda m, cat: (
                sum(m.Dispatch[g, t] for g in m.G for t in m.T if m.gen_type[g] == cat)
                <= yearly_limit[cat] * len(m.T) / (24 * 365)
            ),
        )

    def write_results(
        self,
        model,
        capacity_expansion: bool = True,
        security_constrained: bool = True,
        **kwargs,
    ):
        power_flow_keys = ["line_id", "datetime"]
        create_result_df_for_var(model.Power_Flow, *power_flow_keys).join(
            create_result_df_for_constr(
                model, model.Power_Flow_ub, *power_flow_keys
            ).rename({"dual": "ub_dual"}),
            on=power_flow_keys,
        ).join(
            create_result_df_for_constr(
                model, model.Power_Flow_lb, *power_flow_keys
            ).rename({"dual": "lb_dual"}),
            on=power_flow_keys,
        ).write_parquet("power_flow.parquet")

        if capacity_expansion:
            create_result_df_for_var(model.Build_Out, "gen_id").write_parquet(
                "build_out.parquet"
            )
        create_result_df_for_var(model.Dispatch, "gen_id", "datetime").write_parquet(
            "dispatch.parquet"
        )
        create_result_df_for_var(model.Load_Unserved, "bus", "datetime").write_parquet(
            "load_unserved.parquet"
        )


def create_result_df_for_var(var, *key_names):
    if len(key_names) > 1:
        data = {
            key: [key[i] for key in var.index_set()] for i, key in enumerate(key_names)
        }
    else:
        data = {key_names[0]: [key for key in var.index_set()]}

    data["solution"] = [pyo.value(var[key]) for key in var.index_set()]

    return pl.DataFrame(data)


def create_result_df_for_constr(m, constr, *key_names):
    if len(key_names) > 1:
        data = {
            key: [key[i] for key in constr.index_set()]
            for i, key in enumerate(key_names)
        }
    else:
        data = {key_names[0]: [key for key in constr.index_set()]}

    data["dual"] = [m.dual[constr[key]] for key in constr.index_set()]
    return pl.DataFrame(data)


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    benchmark = Bench(
        size=1,
        input_dir=base_dir / "model_data",
        results_dir=base_dir / "results_pyomo",
        security_constrained=True,
        capacity_expansion=False,
        yearly_limits=False,
        variable_capacity_factors=False,
    )
    m = benchmark.run()
