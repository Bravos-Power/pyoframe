from pathlib import Path

import altair as alt
import polars as pl

INPUT_DIR = (
    Path(__file__).parent / "input_data"
)  # Modify if the input_data folder is located elsewhere


def main():
    ### Load data here
    df_generators = pl.read_parquet(INPUT_DIR / "generators.parquet")
    df_loads = pl.read_parquet(INPUT_DIR / "loads.parquet")

    ### YOUR CODE GOES HERE


def plot_results(dispatch_results, save_to="energy_mix.png"):
    """Plot the energy mix of the dispatch results.

    Parameters
    ----------
    dispatch : polars.DataFrame
        The dispatch results DataFrame.
        Should contain columns "gen_id", "datetime", and "solution".
    save_to : str, optional
        The path to save the plot to.
    """
    if dispatch_results is None:
        print("No results to plot.")
        return

    gens = pl.read_parquet("input_data/generators.parquet")

    dispatch_results = dispatch_results.join(
        gens.select(["gen_id", "type"]), on="gen_id"
    )
    dispatch_results = dispatch_results.group_by("type", "datetime").agg(
        pl.col("solution").sum()
    )
    dispatch_results = dispatch_results.join(
        dispatch_results.group_by("type").agg(std=pl.col("solution").std()), on="type"
    ).sort("std", "type", "datetime")

    plot = dispatch_results.plot.area(
        x="datetime:T",
        y=alt.Y("solution:Q"),
        color=alt.Color("type:N", sort=alt.SortField("std", order="descending")),
        order=alt.Order("std:Q"),
    )

    plot.save(save_to)


if __name__ == "__main__":
    dispatch_results = main()
    plot_results(dispatch_results)
