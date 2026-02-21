from benchmark_utils.pyoframe import Benchmark

import pyoframe as pf


class Bench(Benchmark):
    def build(self, **kwargs):
        m = pf.Model()
        cost = pf.Param(f"input_{self.size}.parquet")
        m.X = pf.Variable(cost, lb=0, ub=1)
        m.minimize = (cost * m.X).sum()
        m.con_X = m.X.sum() >= len(cost) / 2
        return m

    def write_results(self, model, **kwargs):
        model.X.solution.write_parquet(f"output_{self.size}.parquet")


if __name__ == "__main__":
    from pathlib import Path

    cwd = Path(__file__).parent
    results_dir = cwd / "model_results"
    results_dir.mkdir(exist_ok=True)
    bench = Bench(size=100_000, input_dir=cwd / "model_data", results_dir=results_dir)
    bench.run()
