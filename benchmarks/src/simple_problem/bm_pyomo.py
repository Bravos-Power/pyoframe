import polars as pl
import pyomo.environ as pyo
from benchmark_utils.pyomo import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        df = pl.read_parquet(f"input_{self.size}.parquet")

        m = pyo.ConcreteModel()
        m.Xs = pyo.Set(initialize=df["id"].to_list())
        m.cost = pyo.Param(m.Xs, initialize={id: cost for id, cost in df.iter_rows()})
        m.X = pyo.Var(m.Xs, bounds=(0, 1))
        m.OBJ = pyo.Objective(expr=pyo.sum_product(m.cost, m.X), sense=pyo.minimize)
        m.con_X = pyo.Constraint(expr=pyo.summation(m.X) >= len(df) / 2)
        return m

    def write_results(self, model, **kwargs):
        pl.DataFrame(
            [(id, model.X[id].value) for id in model.Xs],
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
