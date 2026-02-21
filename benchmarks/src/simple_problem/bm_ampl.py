from pathlib import Path

import pandas as pd
from amplpy import AMPL
from benchmark_utils.ampl import Benchmark


class Bench(Benchmark):
    def build(self, **kwargs):
        model = AMPL()

        model.eval("""
        set IDS;

        param cost {IDS};

        var X {i in IDS} >= 0, <= 1;

        minimize obj:
            sum {i in IDS} cost[i] * X[i];

        s.t. con_X:
            sum {i in IDS} X[i] >= card(IDS) / 2;
        """)

        cost = pd.read_parquet(f"input_{self.size}.parquet").set_index("id")
        model.set["IDS"] = cost.index
        model.param["cost"] = cost

        return model

    def write_results(self, model, **kwargs):
        model.var["X"].get_values().to_pandas().reset_index(names="id").rename(
            columns={"X.val": "solution"}
        ).to_parquet(f"output_{self.size}.parquet")


if __name__ == "__main__":
    cwd = Path(__file__).parent
    results_dir = cwd / "model_results"
    results_dir.mkdir(exist_ok=True)

    bench = Bench(
        size=10_000,
        input_dir=cwd / "model_data",
        results_dir=results_dir,
    )
    bench.run()
