import gurobipy as gp
import polars as pl
from benchmark_utils.gurobipy import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        df = pl.read_parquet(f"input_{self.size}.parquet")

        m = gp.Model()

        self.X = m.addVars(df["id"].to_list(), lb=0, ub=1)
        cost = gp.tupledict(df.rows())
        m.setObjective(self.X.prod(cost), gp.GRB.MINIMIZE)
        m.addConstr(self.X.sum() >= len(df) / 2)

        return m

    def write_results(self, model, **kwargs):
        pl.DataFrame(
            [(id, var.getAttr(gp.GRB.Attr.X)) for id, var in self.X.items()],
            schema={"id": pl.Int64, "value": pl.Float64},
            orient="row",
        ).write_parquet(f"output_{self.size}.parquet")


if __name__ == "__main__":
    from pathlib import Path

    cwd = Path(__file__).parent
    results_dir = cwd / "model_results"
    results_dir.mkdir(exist_ok=True)
    bench = Bench(size=10_000, input_dir=cwd / "model_data", results_dir=results_dir)
    bench.run()
