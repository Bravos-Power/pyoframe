import cvxpy as cp
import polars as pl
from benchmark_utils.cvxpy import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        df = pl.read_parquet(f"input_{self.size}.parquet")
        self.ids = df.select("id")
        costs = df["cost"].to_numpy()

        n = len(self.ids)

        self.X = cp.Variable(n)

        objective = cp.Minimize(costs @ self.X)

        constraints = [
            0 <= self.X,
            self.X <= 1,
            cp.sum(self.X) >= n / 2,
        ]

        model = cp.Problem(objective, constraints)

        return model

    def write_results(self, model, **kwargs):
        solution = self.ids.with_columns(solution=pl.lit(self.X.value))

        solution.write_parquet(f"output_{self.size}.parquet")


if __name__ == "__main__":
    from pathlib import Path

    cwd = Path(__file__).parent
    results_dir = cwd / "model_results"
    results_dir.mkdir(exist_ok=True)

    bench = Bench(size=10_000, input_dir=cwd / "model_data", results_dir=results_dir)
    bench.run()
