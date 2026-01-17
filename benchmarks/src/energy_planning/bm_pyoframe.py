"""Pyoframe implementation of the facility location benchmark."""

from pathlib import Path

import polars as pl
from benchmark_utils.pyoframe import PyoframeBenchmark

from pyoframe import Model, Param, Set, Variable


class Bench(PyoframeBenchmark):
    def build(self):
        assert self.input_dir is not None, "Input directory must be specified."

        m = Model(solver=self.solver, solver_uses_variable_names=True, verbose=True)

        ## LOAD DATA
        BASE_MW = 100

        m.gens = pl.read_parquet(self.input_dir / "generators.parquet")
        lines = pl.read_parquet(self.input_dir / "lines_simplified.parquet")
        from_buses = lines.select("line_id", bus="from_bus")
        to_buses = lines.select("line_id", bus="to_bus")
        loads = pl.read_parquet(self.input_dir / "loads.parquet")
        capex = pl.read_csv(self.input_dir / "capex_costs.csv")
        cost_params = pl.read_csv(self.input_dir / "cost_parameters.csv")
        COST_UNSERVED_LOAD = cost_params.filter(name="load_unserved_MWh")["cost"].item()

        ## DEFINE SETS AND PARAMS ##
        buses = Set(from_buses["bus"].unique()) + Set(to_buses["bus"].unique())

        # Shrink the time horizon to allow for different model sizes
        hours = loads.get_column("datetime").unique().sort()
        if self.size is not None:
            print(f"Shrinking model to first {self.size} hours")
            hours = hours.head(self.size)
        m.hours = Set(hours)

        m.line_rating = Param(lines["line_id", "line_rating_MW"])
        susceptance = 1 / Param(lines["line_id", "reactance"])
        loads = Param(loads["bus", "datetime", "active_load"]).within(m.hours)

        ## DEFINE VARIABLES ##
        m.Build_Out = Variable(m.gens["gen_id"], lb=0, ub=m.gens["gen_id", "Pmax"])
        m.Dispatch = Variable(m.gens["gen_id"], m.hours, lb=0, ub=m.Build_Out)
        m.Voltage_Angle = Variable(buses, m.hours)
        m.Power_Flow = Variable(
            lines["line_id"], m.hours, lb=-m.line_rating, ub=m.line_rating
        )
        m.Load_Unserved = Variable(loads, lb=0, ub=loads)

        ## DEFINE CONSTRAINTS ##
        m.Con_Slack_Bus = m.Voltage_Angle.pick(bus=1) == 0

        m.Con_Power_Flow = m.Power_Flow == BASE_MW * susceptance * (
            m.Voltage_Angle.map(to_buses) - m.Voltage_Angle.map(from_buses)
        )

        m.Con_Power_Balance = 0 == (
            m.Dispatch.map(m.gens[["gen_id", "bus"]])
            | -loads
            | m.Load_Unserved
            | -m.Power_Flow.map(from_buses)
            | m.Power_Flow.map(to_buses)
        )

        m.minimize = COST_UNSERVED_LOAD * m.Load_Unserved.sum()

        ## SET OBJECTIVE ##
        operating_costs = (
            m.Dispatch * m.gens["gen_id", "cost_per_MWh_linear"]
        ).sum() + (m.Build_Out.map(m.gens[["gen_id", "type"]]) * capex).sum()
        capital_costs = (
            m.Build_Out * m.gens["gen_id", "hourly_overhead_per_MW_capacity"]
        ).sum() * len(m.hours)
        m.minimize += operating_costs + capital_costs

        self.add_dispatch_capacity_constraints(m)
        # self.add_security_constraints(m)

        m.params.Crossover = False
        m.params.Method = 2

        print(m.variables_size_info(memory_unit="mb"))
        print(m.constraints_size_info(memory_unit="mb"))

        return m

    def add_security_constraints(self, m: Model):
        bodf = Param(self.input_dir / "branch_outage_dist_facts.parquet")

        # Power flow on outage_line * factor + affected_line_flow <= affected_line_rating
        conting_flow = m.Power_Flow.over(
            "outage_line_id"
        ).drop_extras() + m.Power_Flow.rename(
            {"line_id": "outage_line_id"}
        ) * bodf.rename({"affected_line_id": "line_id"})
        line_rating = m.line_rating.over("outage_line_id", "datetime").drop_extras()
        m.Con_Contingency_Pos = conting_flow <= line_rating
        m.Con_Contingency_Neg = -line_rating <= conting_flow

    def add_dispatch_capacity_constraints(self, m: Model):
        m.Con_Yearly_Limits = m.Dispatch.map(m.gens[["gen_id", "type"]]).sum(
            "datetime"
        ).drop_extras() <= Param(self.input_dir / "yearly_limits.parquet") * len(
            m.hours
        ) / (24 * 365)

        vcf = Param(self.input_dir / "variable_capacity_factors.parquet").within(
            m.hours
        )
        vcf_type_to_type = pl.read_csv(self.input_dir / "map_type_to_vcf_type.csv")

        m.Con_Variable_Dispatch_Limit = m.Dispatch.drop_extras() <= (
            vcf.map(vcf_type_to_type).map(m.gens[["gen_id", "type"]])
            * m.gens["gen_id", "Pmax"]
        )

    def save_results(self, m: Model, path: Path) -> None:
        m.Power_Flow.solution.join(
            m.Power_Flow_ub.dual.rename({"dual": "ub_dual"}), on=["line_id", "datetime"]
        ).join(
            m.Power_Flow_lb.dual.rename({"dual": "lb_dual"}), on=["line_id", "datetime"]
        ).write_parquet(path / "power_flow.parquet")

        m.Build_Out.solution.write_parquet(path / "build_out.parquet")
        m.Dispatch.solution.write_parquet(path / "dispatch.parquet")
        m.Load_Unserved.solution.filter(pl.col("solution") != 0).write_parquet(
            path / "load_unserved.parquet"
        )


if __name__ == "__main__":
    from pyoptinterface import TerminationStatusCode

    input_dir = Path(__file__).parent / "model_data"
    benchmark = Bench("gurobi", size=24 * 31, input_dir=input_dir, block_solver=False)
    m = benchmark.run()
    if m.attr.TerminationStatus in {
        TerminationStatusCode.INFEASIBLE,
        TerminationStatusCode.INFEASIBLE_OR_UNBOUNDED,
    }:
        import warnings

        warnings.warn("Model is infeasible, computing IIS...")
        m.compute_IIS()
        m.write("inf.ilp")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    benchmark.save_results(m, results_dir)
