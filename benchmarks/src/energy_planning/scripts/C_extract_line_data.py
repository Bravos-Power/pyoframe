import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import json

    import altair as alt
    import marimo as mo

    alt.data_transformers.enable("vegafusion")

    import pandas as pd
    import polars as pl

    return alt, json, mo, pd, pl


@app.cell
def _():
    CATS_DATA = "../raw_data/downloads/CATS_lines.json"
    MATPOWER_DATA = "../raw_data/preprocessed/matpower_branch.parquet"
    LINE_DATA_OUT = "../raw_data/preprocessed/lines.parquet"
    return CATS_DATA, LINE_DATA_OUT, MATPOWER_DATA


@app.cell
def _(mo):
    mo.md(r"""
    ## Load CATS data
    """)
    return


@app.cell
def _(CATS_DATA, json, pd):
    geojson = json.load(open(CATS_DATA))
    df_cats = pd.json_normalize(geojson, record_path=["features"])
    df_cats
    return (df_cats,)


@app.cell
def _(df_cats, pl):
    # Clean data

    # Don't include geometry since some lines are MultiLineStrings (lines with multiple segments)
    # and these 2D arrays cannot be represented in the PyArrow format.

    # Simplify column names
    df_cats_clean = pl.from_pandas(
        df_cats.drop(columns=["geometry.coordinates", "geometry.type"])
    )
    df_cats_clean.columns = [
        c.replace("properties.", "") for c in df_cats_clean.columns
    ]

    # # Remove columns with no data and the coords since they're in the geometry column
    df_cats_clean = df_cats_clean.drop(
        "Lat1",
        "Lon1",
        "Lat2",
        "Lon2",
        "type",
        "Structure_Type",
        "Type",
        "Circuit",
        "Structure_Material",
    )

    # # Keep only one set of IDS
    assert df_cats_clean["CATS_ID"].is_unique().all()
    df_cats_clean = df_cats_clean.drop("id").cast({"CATS_ID": pl.Int64})
    # # Improve naming
    col_names = {
        "CATS_ID": "line_id",
        "f_bus": "from_bus",
        "t_bus": "to_bus",
        "kV": "voltage_kv",
        "rate_a": "line_rating",
        "br_r": "resistance",
        "br_x": "reactance",
        "br_b": "total_capactive_susceptance",
    }
    df_cats_clean = df_cats_clean.rename(col_names)
    df_cats_clean = df_cats_clean.select(
        list(col_names.values())
        + [c for c in df_cats_clean.columns if c not in col_names.values()]
    )

    # Reorder from_bus and to_bus so from_bus < to_bus
    df_cats_clean = df_cats_clean.with_columns(
        pl.when(pl.col("from_bus") < pl.col("to_bus"))
        .then(pl.col("from_bus"))
        .otherwise(pl.col("to_bus"))
        .alias("from_bus"),
        pl.when(pl.col("from_bus") < pl.col("to_bus"))
        .then(pl.col("to_bus"))
        .otherwise(pl.col("from_bus"))
        .alias("to_bus"),
    )
    df_cats_clean
    return (df_cats_clean,)


@app.cell
def _(df_cats_clean, pl):
    assert df_cats_clean.filter(pl.col("from_bus") == pl.col("to_bus")).height == 0, (
        "Lines must not connect a bus to itself"
    )

    assert (df_cats_clean["reactance"] > 0).all(), (
        "Cannot have zero or negative reactance"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load MATPOWER data
    """)
    return


@app.cell
def _(MATPOWER_DATA, pl):
    df_matpower = pl.read_parquet(MATPOWER_DATA)
    df_matpower
    return (df_matpower,)


@app.cell
def _(df_matpower, pl):
    # Clean
    df_matpower_clean = df_matpower.with_row_index("line_id", 1)
    df_matpower_clean = df_matpower_clean.rename(
        {
            "f_bus": "from_bus",
            "t_bus": "to_bus",
            "x": "reactance",
            "r": "resistance",
            "b": "total_capactive_susceptance",
            "rateA": "line_rating",
            "ratio": "transformer",
        }
    )

    # Reorder from_bus and to_bus so from_bus < to_bus
    df_matpower_clean = df_matpower_clean.with_columns(
        pl.when(pl.col("from_bus") < pl.col("to_bus"))
        .then(pl.col("from_bus"))
        .otherwise(pl.col("to_bus"))
        .alias("from_bus"),
        pl.when(pl.col("from_bus") < pl.col("to_bus"))
        .then(pl.col("to_bus"))
        .otherwise(pl.col("from_bus"))
        .alias("to_bus"),
    )

    df_matpower_clean
    return (df_matpower_clean,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Check data matches
    """)
    return


@app.cell
def _(df_cats_clean, df_matpower_clean):
    df_joined = df_cats_clean.join(
        df_matpower_clean, on=["line_id"], validate="1:1", how="full", coalesce=True
    )

    # Drop columns that are identical
    potential_duplicates = [
        c for c in df_joined.columns if f"{c}_right" in df_joined.columns
    ]
    for c in potential_duplicates:
        if df_joined[f"{c}"].equals(df_joined[f"{c}_right"]):
            df_joined = df_joined.drop(f"{c}_right")

    df_joined
    return (df_joined,)


@app.cell
def _(alt, df_joined):
    x_axis = alt.X("line_rating", title="Line Rating (?)", scale=alt.Scale(type="log"))
    x_axis_right = alt.X(
        "line_rating_right", title="Line Rating (MVA)", scale=alt.Scale(type="log")
    )
    df_joined.plot.bar(
        x=x_axis, y="count()", color="voltage_kv:N"
    ) | df_joined.plot.bar(x=x_axis_right, y="count()", color="voltage_kv:N")
    return


@app.cell
def _(df_joined):
    # Matpower values are accurate. For some reason the GIS values were divided by 100. We keep the Matpower values.
    assert (df_joined["line_rating"] * 100 == df_joined["line_rating_right"]).all()
    df_final = df_joined.drop("line_rating").rename(
        {"line_rating_right": "line_rating_MW"}
    )
    return (df_final,)


@app.cell
def _(LINE_DATA_OUT, df_final):
    df_final.write_parquet(LINE_DATA_OUT)
    return


if __name__ == "__main__":
    app.run()
