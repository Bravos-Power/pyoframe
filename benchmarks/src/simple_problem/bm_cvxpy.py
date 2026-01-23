import cvxpy as cp
import polars as pl
from benchmark_utils.cvxpy import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        df = pl.read_parquet(f"input_{self.size}.parquet")

        ids = df["id"].to_list()

        cost = dict(df.rows())

        n = len(ids)

        X = cp.Variable(n)

        c = [cost[i] for i in ids]

        objective = cp.Minimize(c @ X)

        constraints = [
            X >= 0,
            X <= 1,
            cp.sum(X) >= n / 2,
        ]

        problem = cp.Problem(objective, constraints)

        self.problem = problem
        self.X = X
        self.ids = ids

        return problem

    def write_results(self, problem, **kwargs):
        data = [(i, float(self.X.value[k])) for k, i in enumerate(self.ids)]

        pl.DataFrame(
            data,
            schema={"id": pl.Int64, "value": pl.Float64},
            orient="row",
        ).write_parquet(f"output_{self.size}.parquet")


if __name__ == "__main__":
    from pathlib import Path

    cwd = Path(__file__).parent
    results_dir = cwd / "model_results"
    results_dir.mkdir(exist_ok=True)

    bench = Bench(
        size=10_000,
        input_dir=cwd / "model_data",
        results_dir=results_dir,
    )
    bench.run()
