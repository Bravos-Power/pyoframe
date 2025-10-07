"""Pyoframe implementation of the facility location benchmark."""

from pathlib import Path

import polars as pl
from benchmark_utils import PyoframeBenchmark

import pyoframe as pf


class Bench(PyoframeBenchmark):
    def build(self):
        assert self.input_dir is not None, "Input directory must be specified."

        # bodf = pl.read_parquet(self.input_dir / "branch_outage_dist_facts.parquet")
        generators = pl.read_parquet(
            self.input_dir / "generators.parquet"
        ).with_row_index(name="gen")
        lines = pl.read_parquet(self.input_dir / "lines_simplified.parquet")
        loads = pl.read_parquet(self.input_dir / "loads.parquet")
        vcf = pl.read_parquet(self.input_dir / "variable_capacity_factors.parquet")
        yearly_limits = pl.read_parquet(self.input_dir / "yearly_limits.parquet")

        buses = pf.Set(lines["from_bus"].rename("bus").unique()) + pf.Set(
            lines["to_bus"].rename("bus").unique()
        )

        m = pf.Model(solver=self.solver, solver_uses_variable_names=True)

        hours = loads.get_column("datetime").unique().sort().head(self.size)
        active_load = loads[["bus", "datetime", "active_load"]].to_expr().within(hours)

        # TODO allow broadcasting on add_dim() or improve error messages when using ub=
        m.dispatch = pf.Variable(generators["gen"], hours, lb=0)
        m.voltage_angle = pf.Variable(buses, hours)
        m.power_flow = pf.Variable(
            lines["line_id"],
            hours,
            lb=-(lines[["line_id", "line_rating"]].to_expr()).add_dim("datetime"),
            ub=lines[["line_id", "line_rating"]].to_expr().add_dim("datetime"),
        )

        m.slack_bus_const = m.voltage_angle.pick(bus=buses.data["bus"].max()) == 0
        # TODO allow rename with equality similar to pick
        # TODO support division of constant expressions
        m.constraint_power_flow = m.power_flow == lines.select(
            "line_id", 1 / pl.col("reactance")
        ) * (
            m.voltage_angle.rename({"bus": "to_bus"}).map(lines[["line_id", "to_bus"]])
            - m.voltage_angle.rename({"bus": "from_bus"}).map(
                lines[["line_id", "from_bus"]]
            )
        )

        m.power_balance = 0 == (
            m.dispatch.map(generators[["gen", "bus"]]).keep_extras()
            - active_load.keep_extras()
            - m.power_flow.map(lines[["line_id", "from_bus"]])
            .rename({"from_bus": "bus"})
            .keep_extras()
            + m.power_flow.map(lines[["line_id", "to_bus"]])
            .rename({"to_bus": "bus"})
            .keep_extras()
        )

        m.con_yearly_limits = (
            m.dispatch.map(generators[["gen", "type"]]).sum("datetime").drop_extras()
            <= yearly_limits[["type", "limit"]]
        )

        m.variable_dispatch_limit = (
            m.dispatch.drop_extras()
            <= (
                vcf[["type", "datetime", "capacity_factor"]]
                .to_expr()
                .map(generators[["gen", "type"]])
                * generators[["gen", "Pmax"]]
            ).drop_extras()
        )
        m.dispatch_limit = m.dispatch <= generators[["gen", "Pmax"]].to_expr().add_dim(
            "datetime"
        )

        return m


if __name__ == "__main__":
    input_dir = (
        Path(__file__).parent.parent.parent
        / "input_data"
        / "energy_planning"
        / "final_inputs"
    )
    m: pf.Model = Bench("gurobi", 2, input_dir=input_dir, block_solver=False).run()
    m.compute_IIS()
    # todo autocompute iis
    m.write("inf.ilp")
