from pathlib import Path

import pandas as pd
from amplpy import AMPL
from benchmark_utils.ampl import Benchmark


class Bench(Benchmark):
    def build(
        self,
        capacity_expansion: bool = True,
        security_constrained: bool = True,
        yearly_limits: bool = True,
        variable_capacity_factors: bool = True,
        **kwargs,
    ):
        ampl = AMPL()

        ampl.eval("""
        set G;
        set L;
        set B;
        set T;
        set LOADS within B cross T;
        set GEN_TYPES;
        
        param COST_UNSERVED_LOAD;
        param BASE_MW;
        param SLACK_BUS;
        param gen_bus {G};
        param gen_type {G} symbolic;
        param gen_pmax {G};
        param gen_cost {G};
        param gen_overhead {G};
        param line_from {L};
        param line_to {L};
        param line_rating {L};
        param susceptance {L};
        param load {LOADS};
        param capex {GEN_TYPES} default 0;
        

        var Voltage_Angle {b in B, t in T};
        var Power_Flow {l in L, t in T};
        var Load_Unserved {(b,t) in LOADS} >= 0, <= load[b,t];
        """)

        if capacity_expansion:
            ampl.eval("""
            var Build_Out {g in G} >= 0, <= gen_pmax[g];
            var Dispatch {g in G, t in T} >= 0;
            s.t. Dispatch_ub {g in G, t in T}: Dispatch[g,t] <= Build_Out[g];
            """)
        else:
            ampl.eval("var Dispatch {g in G, t in T} >= 0, <= gen_pmax[g];")

        ampl.eval("""
        s.t. Power_Flow_ub {l in L, t in T}: Power_Flow[l,t] <= line_rating[l];
        s.t. Power_Flow_lb {l in L, t in T}: Power_Flow[l,t] >= -line_rating[l];
        s.t. Con_Slack_Bus {t in T}: Voltage_Angle[SLACK_BUS,t] = 0;
        s.t. Con_Power_Flow {l in L, t in T}:
            Power_Flow[l,t] = BASE_MW * susceptance[l] * (Voltage_Angle[line_to[l],t] - Voltage_Angle[line_from[l],t]);
        s.t. Con_Power_Balance {b in B, t in T}:
            if (b,t) in LOADS then load[b,t] else 0 =
                sum {g in G: gen_bus[g] = b} Dispatch[g,t] +
                sum {l in L: line_to[l]=b} Power_Flow[l,t] -
                sum {l in L: line_from[l]=b} Power_Flow[l,t] +
                if (b,t) in LOADS then Load_Unserved[b,t] else 0;
        """)

        if capacity_expansion:
            ampl.eval("""
            minimize obj:
                sum {(b,t) in LOADS} COST_UNSERVED_LOAD * Load_Unserved[b,t] +
                sum {g in G, t in T} gen_cost[g] * Dispatch[g,t] +
                sum {g in G} capex[gen_type[g]] * Build_Out[g] +
                sum {g in G} gen_overhead[g] * Build_Out[g] * card(T);
            """)
        else:
            ampl.eval("""
            minimize obj:
                sum {(b,t) in LOADS} COST_UNSERVED_LOAD * Load_Unserved[b,t] +
                sum {g in G, t in T} gen_cost[g] * Dispatch[g,t];
            """)

        gens = pd.read_parquet("generators.parquet").set_index("gen_id")
        lines = pd.read_parquet("lines_simplified.parquet").set_index("line_id")
        loads_df = pd.read_parquet("loads.parquet")
        loads_df["datetime"] = loads_df["datetime"].astype(str)
        capex_df = pd.read_csv("capex_costs.csv").set_index("type")
        cost_params = pd.read_csv("cost_parameters.csv")

        hours = sorted(loads_df["datetime"].unique())
        if self.size is not None:
            hours = hours[: self.size]
            loads_df = loads_df[loads_df["datetime"].isin(hours)]

        loads_df = loads_df.set_index(["bus", "datetime"])

        ampl.set["G"] = gens.index
        ampl.set["L"] = lines.index
        ampl.set["B"] = sorted(set(lines["from_bus"]).union(lines["to_bus"]))
        ampl.set["T"] = hours
        ampl.set["LOADS"] = loads_df.index
        ampl.set["GEN_TYPES"] = sorted(gens["type"].unique())

        ampl.param["gen_bus"] = gens["bus"]
        ampl.param["gen_type"] = gens["type"]
        ampl.param["gen_pmax"] = gens["Pmax"]
        ampl.param["gen_cost"] = gens["cost_per_MWh_linear"]
        ampl.param["gen_overhead"] = gens["hourly_overhead_per_MW_capacity"]
        ampl.param["capex"] = capex_df["yearly_capex_cost_per_KW"]

        ampl.param["line_from"] = lines["from_bus"]
        ampl.param["line_to"] = lines["to_bus"]
        ampl.param["line_rating"] = lines["line_rating_MW"]
        ampl.param["susceptance"] = 1 / lines["reactance"]

        ampl.param["load"] = loads_df["active_load"]
        ampl.param["BASE_MW"] = 100
        ampl.param["COST_UNSERVED_LOAD"] = cost_params.query(
            'name=="load_unserved_MWh"'
        )["cost"].iloc[0]
        ampl.param["SLACK_BUS"] = 1

        if security_constrained:
            self.add_security_constraints(ampl)

        if yearly_limits:
            self.add_yearly_limits(ampl)

        if variable_capacity_factors:
            self.add_vcf(ampl, hours)

        return ampl

    def add_security_constraints(self, ampl):
        ampl.eval("""
        set CONTIG within L cross L;
        param bodf {CONTIG};
                  
        s.t. Con_Contingency_Pos {(out,aff) in CONTIG, t in T}:
            Power_Flow[aff,t] + bodf[out,aff]*Power_Flow[out,t] <= line_rating[aff];
        s.t. Con_Contingency_Neg {(out,aff) in CONTIG, t in T}:
            Power_Flow[aff,t] + bodf[out,aff]*Power_Flow[out,t] >= -line_rating[aff];
        """)

        bodf = pd.read_parquet("branch_outage_dist_facts.parquet").set_index(
            ["outage_line_id", "affected_line_id"]
        )
        ampl.set["CONTIG"] = bodf.index
        ampl.param["bodf"] = bodf["factor"]

    def add_vcf(self, ampl, hours):
        ampl.eval("""
        set VCF_TYPES;
        set VCF_GEN_TYPES within GEN_TYPES;

        param gen_to_vcf_type {VCF_GEN_TYPES} symbolic;
        param vcf {vt in VCF_TYPES, t in T};
        
        s.t. Con_Variable_Dispatch_Limit {g in G, t in T: gen_type[g] in VCF_GEN_TYPES}:
            Dispatch[g,t] <= vcf[gen_to_vcf_type[gen_type[g]], t] * gen_pmax[g];
        """)

        vcf_map = pd.read_csv("map_type_to_vcf_type.csv").set_index("type")
        ampl.set["VCF_GEN_TYPES"] = vcf_map.index
        ampl.set["VCF_TYPES"] = vcf_map["vcf_type"].unique()
        ampl.param["gen_to_vcf_type"] = vcf_map

        vcf_df = pd.read_parquet("variable_capacity_factors.parquet")
        vcf_df["datetime"] = vcf_df["datetime"].astype(str)
        vcf_df = vcf_df[vcf_df["datetime"].isin(hours)]
        vcf_df = vcf_df.set_index(["vcf_type", "datetime"])
        ampl.param["vcf"] = vcf_df

    def add_yearly_limits(self, ampl):
        ampl.eval("""
        set TYPES_WITH_LIMIT within GEN_TYPES;
        param yearly_limit {TYPES_WITH_LIMIT};
        s.t. Con_Yearly_Limits {cat in TYPES_WITH_LIMIT}:
            sum {g in G, t in T: gen_type[g]=cat} Dispatch[g,t] <= yearly_limit[cat]*card(T)/(24*365);
        """)

        yearly_df = pd.read_parquet("yearly_limits.parquet").set_index("type")
        ampl.set["TYPES_WITH_LIMIT"] = yearly_df.index
        ampl.param["yearly_limit"] = yearly_df["limit"]

    def write_results(
        self,
        model,
        capacity_expansion: bool = True,
        **kwargs,
    ):
        def _entity_df(entity, key_names, value_kind: str, value_name: str):
            df = entity.get_values().to_pandas().reset_index()
            value_col = next(
                (col for col in df.columns if col.endswith(f".{value_kind}")), None
            )
            if value_col is None:
                raise ValueError(f"No .{value_kind} column found in AMPL entity values")

            key_cols = [col for col in df.columns if col != value_col]
            rename_map = dict(zip(key_cols, key_names))
            rename_map[value_col] = value_name

            df = df.rename(columns=rename_map)

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"]).astype("datetime64[us]")

            return df

        power_flow_keys = ["line_id", "datetime"]
        power_flow = _entity_df(
            model.var["Power_Flow"], power_flow_keys, "val", "solution"
        )
        power_flow_ub = _entity_df(
            model.con["Power_Flow_ub"], power_flow_keys, "dual", "ub_dual"
        )
        power_flow_lb = _entity_df(
            model.con["Power_Flow_lb"], power_flow_keys, "dual", "lb_dual"
        )
        power_flow.merge(power_flow_ub, on=power_flow_keys, how="left").merge(
            power_flow_lb, on=power_flow_keys, how="left"
        ).to_parquet("power_flow.parquet", index=False)

        if capacity_expansion:
            _entity_df(
                model.var["Build_Out"], ["gen_id"], "val", "solution"
            ).to_parquet("build_out.parquet", index=False)

        _entity_df(
            model.var["Dispatch"], ["gen_id", "datetime"], "val", "solution"
        ).to_parquet("dispatch.parquet", index=False)

        _entity_df(
            model.var["Load_Unserved"], ["bus", "datetime"], "val", "solution"
        ).to_parquet("load_unserved.parquet", index=False)

        _df = _entity_df(
            model.con["Con_Power_Balance"], ["bus", "datetime"], "dual", "dual"
        )
        _df["dual"] *= -1  # Fix sign
        _df.to_parquet("power_balance_duals.parquet", index=False)


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    benchmark = Bench(
        size=1,
        input_dir=base_dir / "model_data",
        results_dir=base_dir / "results_ampl",
        security_constrained=False,
        capacity_expansion=True,
        yearly_limits=False,
        variable_capacity_factors=False,
    )
    m = benchmark.run()
