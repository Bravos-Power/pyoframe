import polars as pl
import pyoptinterface as poi
from benchmark_utils.pyoptinterface import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        m = super().build(**kwargs)

        df = pl.read_parquet(f"input_{self.size}.parquet")
        cost = poi.tupledict({id: cost for id, cost in df.iter_rows()})

        self.X = m.add_variables(df["id"], lb=0, ub=1)

        obj = poi.ExprBuilder()
        for id, x in self.X.items():
            obj += x * cost[id]

        m.set_objective(obj, sense=poi.ObjectiveSense.Minimize)

        m.add_linear_constraint(poi.quicksum(self.X) >= len(cost) / 2)

        return m

    def write_results(self, model, **kwargs):
        pl.DataFrame(
            [
                (id, self.model.get_variable_attribute(x, poi.VariableAttribute.Value))
                for id, x in self.X.items()
            ],
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
