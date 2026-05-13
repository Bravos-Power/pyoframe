import polars as pl
import pulp
from benchmark_utils.pulp import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        costs = pl.read_parquet(f"input_{self.size}.parquet")
        costs = {r["id"]: r["cost"] for r in costs.to_dicts()}

        model = pulp.LpProblem()

        self.X = pulp.LpVariable.dicts("X", costs.keys(), lowBound=0, upBound=1)

        model += pulp.lpSum(cost * self.X[id] for id, cost in costs.items())
        model += pulp.lpSum(self.X[id] for id in costs.keys()) >= len(costs) / 2

        return model

    def write_results(self, model, **kwargs):
        data = [(i, pulp.value(var)) for i, var in self.X.items()]

        pl.DataFrame(
            data, schema={"id": pl.Int64, "value": pl.Float64}, orient="row"
        ).write_parquet(f"output_{self.size}.parquet")


if __name__ == "__main__":
    from pathlib import Path

    cwd = Path(__file__).parent
    results_dir = cwd / "model_results"
    results_dir.mkdir(exist_ok=True)

    bench = Bench(size=10_000, input_dir=cwd / "model_data", results_dir=results_dir)
    bench.run()
