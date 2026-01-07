"""Pyoframe implementation of the facility location benchmark."""

from pathlib import Path

import polars as pl
from benchmark_utils import PyoframeBenchmark

import pyoframe as pf


class Bench(PyoframeBenchmark):
    def build(self):
        assert self.input_dir is not None, "Input directory must be specified."

        ## LOAD DATA

        BASE_MW = 100

        # bodf = pl.read_parquet(self.input_dir / "branch_outage_dist_facts.parquet")
        gens = pl.read_parquet(self.input_dir / "generators.parquet")
        lines = pl.read_parquet(self.input_dir / "lines_simplified.parquet")
        from_buses = lines[["line_id", "from_bus"]].rename({"from_bus": "bus"})
        to_buses = lines[["line_id", "to_bus"]].rename({"to_bus": "bus"})
        loads = pl.read_parquet(self.input_dir / "loads.parquet")
        vcf = pl.read_parquet(self.input_dir / "variable_capacity_factors.parquet")
        yearly_limits = pl.read_parquet(self.input_dir / "yearly_limits.parquet")

        ## DEFINE SETS AND PARAMS ##

        buses = pf.Set(from_buses["bus"].unique()) + pf.Set(to_buses["bus"].unique())
        hours = loads.get_column("datetime").unique().sort()

        # Shrink the time horizon to allow for different model sizes
        if self.size is not None:
            print(f"Shrinking model to first {self.size} hours")
            hours = hours.head(self.size)
        NUM_HOURS = len(hours)

        gen_max = pf.Param(gens["gen_id", "Pmax"])
        line_rating = pf.Param(lines["line_id", "line_rating_MW"])
        susceptance = 1 / pf.Param(lines["line_id", "reactance"])
        loads = pf.Param(loads["bus", "datetime", "active_load"]).within(hours)
        vcf = pf.Param(vcf).within(hours)

        ## DEFINE VARIABLES ##
        m = pf.Model(solver=self.solver, solver_uses_variable_names=True, debug=True)

        m.Build_Out = pf.Variable(gens["gen_id"], lb=0, ub=gen_max)
        m.Dispatch = pf.Variable(gens["gen_id"], hours, lb=0, ub=m.Build_Out)
        m.Voltage_Angle = pf.Variable(buses, hours)
        m.Power_Flow = pf.Variable(
            lines["line_id"], hours, lb=-line_rating, ub=line_rating
        )

        ## DEFINE CONSTRAINTS ##
        m.Con_Slack_Bus = m.Voltage_Angle.pick(bus=1) == 0

        m.Con_Power_Flow = m.Power_Flow == BASE_MW * susceptance * (
            m.Voltage_Angle.map(to_buses) - m.Voltage_Angle.map(from_buses)
        )

        m.Con_Power_Balance = 0 == (
            m.Dispatch.map(gens[["gen_id", "bus"]]).keep_extras()
            - loads.keep_extras()
            - m.Power_Flow.map(from_buses).keep_extras()
            + m.Power_Flow.map(to_buses).keep_extras()
        )

        m.Con_Yearly_Limits = (
            m.Dispatch.map(gens[["gen_id", "type"]]).sum("datetime").drop_extras()
            <= yearly_limits[["type", "limit"]]
        )

        m.Con_Variable_Dispatch_Limit = m.Dispatch.drop_extras() <= (
            vcf.map(gens[["gen_id", "type"]]) * gen_max
        )

        ## SET OBJECTIVE ##
        operating_costs = (m.Dispatch * gens["gen_id", "cost_per_MWh_linear"]).sum()
        # TODO include construction costs
        capital_costs = (
            m.Build_Out * gens["gen_id", "hourly_overhead_per_MW_capacity"]
        ).sum() * NUM_HOURS
        m.minimize = operating_costs + capital_costs

        return m

    def save_results(self, m: pf.Model, path: Path) -> None:
        m.Power_Flow.solution.join(
            m.Power_Flow_ub.dual.rename({"dual": "ub_dual"}), on=["line_id", "datetime"]
        ).join(
            m.Power_Flow_lb.dual.rename({"dual": "lb_dual"}), on=["line_id", "datetime"]
        ).write_parquet(path / "power_flow.parquet")

        m.Build_Out.solution.write_parquet(path / "build_out.parquet")


if __name__ == "__main__":
    from pyoptinterface import TerminationStatusCode

    input_dir = Path(__file__).parent / "model_data"
    benchmark = Bench("gurobi", size=None, input_dir=input_dir, block_solver=False)
    m = benchmark.run()
    if m.attr.TerminationStatus in {
        TerminationStatusCode.INFEASIBLE,
        TerminationStatusCode.INFEASIBLE_OR_UNBOUNDED,
    }:
        m.compute_IIS()
        m.write("inf.ilp")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    benchmark.save_results(m, results_dir)
