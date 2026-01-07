import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    alt.data_transformers.enable("vegafusion")
    return Path, mo, pl


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
    gen_data = pl.read_parquet(INPUT_DIR / "generators.parquet")
    gen_data = gen_data.join(buildout.rename({"solution": "build_mw"}), on="gen_id")
    gen_data
    return (gen_data,)


@app.cell
def _(gen_data, pl):
    _plot_data = gen_data.group_by("type").agg(pl.col("build_mw", "Pmax").sum())

    _plot_data.plot.bar(x="type", y="build_mw", color="type") + _plot_data.plot.scatter(
        x="type", y="Pmax", color="type"
    )
    return


if __name__ == "__main__":
    app.run()
