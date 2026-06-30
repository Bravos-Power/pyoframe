import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    alt.data_transformers.enable("vegafusion")
    return Path, alt, mo, pl


@app.cell
def _(Path):
    RESULTS_DIR = Path("../results/")
    INPUT_DIR = Path("../model_data/")
    return INPUT_DIR, RESULTS_DIR


@app.cell
def _(mo):
    mo.md(r"""
    ## Analayze buildout
    """)
    return


@app.cell
def _(INPUT_DIR, RESULTS_DIR, pl):
    buildout = pl.read_parquet(RESULTS_DIR / "build_out.parquet")
    gens = pl.read_parquet(INPUT_DIR / "generators.parquet")
    gen_data = gens.join(buildout.rename({"solution": "build_mw"}), on="gen_id")
    gen_data
    return gen_data, gens


@app.cell
def _(gen_data, pl):
    _plot_data = gen_data.group_by("type").agg(pl.col("build_mw", "Pmax").sum())

    _plt = _plot_data.plot.bar(
        x="type", y="build_mw", color="type"
    ) + _plot_data.plot.scatter(x="type", y="Pmax", color="type")
    # add title to atlas plot
    _plt.properties(
        title="Built Capacity Relative to Total Potential Capacity by Generator Type",
        # xlabel="Generator Type",
        # ylabel="Capacity (MW)",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Analzye dispatch
    """)
    return


@app.cell
def _(INPUT_DIR, RESULTS_DIR, alt, gens, pl):
    dispatch = pl.read_parquet(RESULTS_DIR / "dispatch.parquet").filter(
        pl.col("solution") != 0
    )

    dispatch = dispatch.join(gens.select(["gen_id", "type"]), on="gen_id")
    dispatch = dispatch.group_by("type", "datetime").agg(pl.col("solution").sum())
    dispatch = dispatch.join(
        dispatch.group_by("type").agg(std=pl.col("solution").std()), on="type"
    ).sort("std", "type", "datetime")
    _plot = dispatch.plot.area(
        x="datetime:T",
        y=alt.Y("solution:Q"),
        color=alt.Color("type:N", sort=alt.SortField("std", order="descending")),
        order=alt.Order("std:Q"),
    )

    load = pl.read_parquet(INPUT_DIR / "loads.parquet")
    load = load.group_by("datetime").agg(pl.col("active_load").sum())
    load = load.filter(pl.col("datetime").is_in(dispatch["datetime"].implode()))
    _plot += load.plot.line(x="datetime:T", y="active_load", color=alt.value("black"))
    _plot
    return (load,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Plot congestion and line flow
    """)
    return


@app.cell
def _(RESULTS_DIR, pl):
    df_flow = pl.read_parquet(RESULTS_DIR / "power_flow.parquet")
    df_flow = df_flow.with_columns(
        congested=(pl.col("ub_dual").abs() + pl.col("lb_dual").abs()) > 0.01
    )
    df_flow
    return (df_flow,)


@app.cell
def _(alt, df_flow, pl):
    df_flow_agg = df_flow.group_by("datetime").agg(
        pl.col("solution").abs().sum().alias("total_flow")
    )
    _left = df_flow_agg.plot.line(x="datetime:T", y="total_flow")

    congested_count = df_flow.group_by("datetime").agg(
        pl.col("congested").sum().alias("num_congested")
    )
    _right = congested_count.plot.line(
        x="datetime:T",
        y=alt.Y("num_congested", axis=alt.Axis(orient="right")),
        color=alt.value("red"),
    )
    chart = alt.layer(_left, _right).resolve_scale(y="independent")
    chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Plot unserved load
    """)
    return


@app.cell
def _(RESULTS_DIR, pl):
    df_unserved = pl.read_parquet(RESULTS_DIR / "load_unserved.parquet")
    df_unserved = df_unserved.filter(pl.col("solution") >= 1e-6)
    df_unserved
    return (df_unserved,)


@app.cell
def _(df_unserved, pl):
    df_unserved.group_by("bus").agg(pl.sum("solution")).plot.bar(
        x="bus:O", y="solution"
    )
    return


@app.cell
def _(df_unserved, load, pl):
    unserved_date = df_unserved.group_by("datetime").agg(pl.sum("solution"))
    unserved_date = unserved_date.join(
        load.group_by("datetime").agg(pl.col("active_load").sum()), on="datetime"
    )
    unserved_date = unserved_date.with_columns(
        percent_unserved=pl.col("solution") / pl.col("active_load")
    )
    (
        unserved_date.plot.line(x="datetime", y="solution")
        | unserved_date.plot.line(x="datetime", y="percent_unserved")
    )
    return


if __name__ == "__main__":
    app.run()
