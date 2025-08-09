"""Pyoframe implementation of the facility location benchmark."""

from benchmark_utils import PyoframeBenchmark


class Bench(PyoframeBenchmark):
    def build(self):
        ...
        # bodf = pl.read_parquet(self.input_dir / "branch_outage_dist_fact.parquet")
        # generators = pl.read_parquet(self.input_dir / "generators.parquet")
        # lines = pl.read_parquet(self.input_dir / "lines_simplified.parquet")


if __name__ == "__main__":
    Bench("gurobi", 5).run()
