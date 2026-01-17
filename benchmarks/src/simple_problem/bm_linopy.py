import linopy as lp
import pandas as pd
from benchmark_utils.linopy import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        cost = (
            pd.read_parquet(f"input_{self.size}.parquet")
            .set_index("id")["cost"]
            .to_xarray()
        )

        m = lp.Model()
        self.X = m.add_variables(lower=0, upper=1, coords=cost.coords, name="X")
        m.add_objective((self.X * cost).sum(), sense="min")
        m.add_constraints(self.X.sum() >= cost.sizes["id"] / 2)
        return m

    def write_results(self, model, **kwargs):
        self.X.solution.to_dataframe().to_parquet(f"output_{self.size}.parquet")


if __name__ == "__main__":
    from pathlib import Path

    cwd = Path(__file__).parent
    results_dir = cwd / "model_results"
    results_dir.mkdir(exist_ok=True)
    bench = Bench(size=10_000, input_dir=cwd / "model_data", results_dir=results_dir)
    bench.run()
