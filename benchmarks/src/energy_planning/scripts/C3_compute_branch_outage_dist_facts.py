import marimo

__generated_with = "0.19.2"
app = marimo.App()


@app.cell
def _():
    # See https://pypsa.readthedocs.io/en/latest/user-guide/contingency-analysis.html#branch-outage-distribution-factors-bodf
    import altair as alt
    import polars as pl

    alt.data_transformers.enable("vegafusion")
    return alt, pl


@app.cell
def _():
    DF_PERC_CUTOFF = 0.02
    MIN_KV = 115
    ISLAND_SENSITIVITY = 1e-5
    return DF_PERC_CUTOFF, ISLAND_SENSITIVITY, MIN_KV


@app.cell
def _():
    LINES_DATA = "../raw_data/preprocessed/lines_simplified.parquet"
    PTDF_DATA = "../raw_data/preprocessed/power_transfer_dist_facts.parquet"
    OUTPUT_PATH = "../raw_data/preprocessed/branch_outage_dist_facts.parquet"
    return LINES_DATA, OUTPUT_PATH, PTDF_DATA


@app.cell
def _(PTDF_DATA, pl):
    ptdf = pl.read_parquet(PTDF_DATA)
    ptdf
    return (ptdf,)


@app.cell
def _(LINES_DATA, pl):
    lines = pl.read_parquet(
        LINES_DATA,
        columns=[
            "line_id",
            "from_bus",
            "to_bus",
            "is_leaf",
            "voltage_kv",
            "line_rating_MW",
        ],
    )
    lines
    return (lines,)


@app.cell
def _(MIN_KV, lines, pl):
    # Step 1: Remove leaf lines because a contingency would create an island.
    lines_1 = lines.filter(~pl.col("is_leaf")).drop("is_leaf")
    # Step 1b: Remove low-voltage lines (distribution system) (and keep transformers)
    lines_1 = lines_1.filter(
        (pl.col("voltage_kv") >= MIN_KV) | pl.col("voltage_kv").is_null()
    )
    lines_1
    return (lines_1,)


@app.cell
def _(lines_1, pl, ptdf):
    # Step 2: Compute the difference in PTDF for every from and to bus.
    bptdf = (
        lines_1.rename({"line_id": "outage_line_id"})
        .join(
            ptdf.rename({"injection": "from_bus", "factor": "injection"}), on="from_bus"
        )
        .join(
            ptdf.rename({"injection": "to_bus", "factor": "withdrawal"}),
            on=["to_bus", "line_id"],
        )
        .select(
            "outage_line_id",
            "line_id",
            factor=pl.col("injection") - pl.col("withdrawal"),
        )
    )
    bptdf
    return (bptdf,)


@app.cell
def _(bptdf, pl):
    # Step 3: Separate diagonal and off-diagonal entries
    diagonal_entries = bptdf.filter(pl.col("outage_line_id") == pl.col("line_id")).drop(
        "line_id"
    )
    offdiagonal_entries = bptdf.filter(pl.col("outage_line_id") != pl.col("line_id"))
    diagonal_entries
    return diagonal_entries, offdiagonal_entries


@app.cell
def _(ISLAND_SENSITIVITY, diagonal_entries, pl):
    # Step 4: Confirm that there are no bridge lines (lines that would cause the grid to split into two unconnected networks)
    bridge_lines = diagonal_entries.filter(pl.col("factor") >= (1 - ISLAND_SENSITIVITY))
    if bridge_lines.height > 0:
        raise ValueError(
            f"Grid is not well connected\n{diagonal_entries.filter(pl.col('factor') >= (1 - ISLAND_SENSITIVITY))}"
        )
    return


@app.cell
def _(diagonal_entries, offdiagonal_entries, pl):
    # Step 5: Normalize the factors to account for the flow on the line in outage itself
    bodf = (
        offdiagonal_entries.join(
            diagonal_entries.select("outage_line_id", denominator=1 - pl.col("factor")),
            on="outage_line_id",
        )
        .with_columns(pl.col("factor") / pl.col("denominator"))
        .drop("denominator")
        .rename({"line_id": "affected_line_id"})
    )
    bodf
    return (bodf,)


@app.cell
def _(alt, bodf, lines_1, pl):
    # Compute percent increase in line rating
    ratings = lines_1[["line_id", "line_rating_MW"]]
    bodf_1 = bodf.join(
        ratings.rename(
            {"line_id": "affected_line_id", "line_rating_MW": "affected_line_rating_MW"}
        ),
        on="affected_line_id",
    ).join(
        ratings.rename(
            {"line_id": "outage_line_id", "line_rating_MW": "outage_line_rating_MW"}
        ),
        on="outage_line_id",
    )
    bodf_1 = bodf_1.with_columns(
        percent_increase=pl.col("outage_line_rating_MW")
        * pl.col("factor").abs()
        / pl.col("affected_line_rating_MW")
    )
    bodf_1 = bodf_1.drop("affected_line_rating_MW", "outage_line_rating_MW")
    bodf_1.select(pl.col("percent_increase").log10()).plot.bar(
        x=alt.X("percent_increase", bin=True), y="count()"
    )
    return (bodf_1,)


@app.cell
def _(DF_PERC_CUTOFF, bodf_1, pl):
    # Plot distribution of factors
    bodf_1.sample(10_000).with_columns(
        pl.col("percent_increase", "factor").abs().log10(),
        keep=pl.col("percent_increase") > DF_PERC_CUTOFF,
    ).plot.scatter(x="factor", y="percent_increase", color="keep")
    return


@app.cell
def _(DF_PERC_CUTOFF, bodf_1, pl):
    # Filter out near zeros
    bodf_2 = bodf_1.filter(pl.col("percent_increase") > DF_PERC_CUTOFF)
    bodf_2 = bodf_2.drop("percent_increase")
    bodf_2 = bodf_2.sort("outage_line_id", "affected_line_id")
    bodf_2
    return (bodf_2,)


@app.cell
def _(OUTPUT_PATH, bodf_2):
    bodf_2.write_parquet(OUTPUT_PATH)
    return


if __name__ == "__main__":
    app.run()
