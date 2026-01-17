import marimo

__generated_with = "0.19.2"
app = marimo.App()


@app.cell
def _():
    import polars as pl

    DEBUG = False
    return DEBUG, pl


@app.cell
def _():
    CATS_DATA = "../raw_data/downloads/CATS_loads.csv"
    LOAD_DATA_OUT = "../raw_data/preprocessed/loads.parquet"
    return CATS_DATA, LOAD_DATA_OUT


@app.cell
def _(CATS_DATA, pl):
    # read dataframe
    df = pl.scan_csv(CATS_DATA, has_header=False)
    df.head().collect()
    return (df,)


@app.cell
def _(df, pl):
    # unpivot
    df2 = (
        df.with_row_index(name="bus", offset=1)
        .unpivot(index="bus", value_name="load", variable_name="hour")
        .with_columns(pl.col("hour").str.strip_prefix("column_").cast(pl.Int32))
        .sort("bus", "hour")
        # cache result since .unpivot cannot be streamed yet https://github.com/pola-rs/polars/issues/20947
        .collect()
        .lazy()
    )
    df2.collect()
    return (df2,)


@app.cell
def _(DEBUG, df2, pl):
    if DEBUG:
        issues = df2.filter(~pl.col("load").str.ends_with("i")).collect()
        if not issues.is_empty():
            raise ValueError(f"Some loads don't end with 'i' as expected\n: {issues}")
    return


@app.cell
def _(df2, pl):
    # keep only active loads
    df3 = df2.with_columns(
        pl.col("load")
        .str.split_exact("+", 1)
        .struct.field("field_0")
        .cast(pl.Float64)
        .alias("active_load")
    ).drop("load")

    df3.head().collect()
    return (df3,)


@app.cell
def _(df3, pl):
    # parse date
    df4 = df3.with_columns(
        (pl.col("hour") - 1) * pl.duration(hours=1) + pl.datetime(2019, 1, 1)
    ).rename({"hour": "datetime"})
    df4.head(25).collect()
    return (df4,)


@app.cell
def _(DEBUG, df2, df4, pl):
    # Confirm loads are always positive (i.e. dataset doesn't include generation) and drop zero loads
    if DEBUG:
        assert df4.filter(pl.col("active_load") < 0).collect().height == 0, (
            "Negative active loads found!"
        )

    df5 = df4.filter(pl.col("active_load") != 0)

    print(
        df5.explain(optimized=True)
    )  # Check the query is optimal (i.e. filter is done before other operations)

    if DEBUG:
        df5 = df5.collect().lazy()  # cache to avoid double computation
        print(
            f"{df5.collect().height / df2.collect().height:.2%} of loads are non-zero."
        )
    return (df5,)


@app.cell
def _(DEBUG, df5, pl):
    _plt = None
    if DEBUG:
        load_ca = df5.group_by("datetime").sum().drop("bus").collect()
        _plt = (
            load_ca.group_by(pl.col("datetime").dt.hour())
            .mean()
            .plot.line(x="datetime", y="active_load")
            .properties(title="Average Load by Hour in California")
        )
    _plt
    return (load_ca,)


@app.cell
def _(DEBUG, load_ca, pl):
    _plt = None
    if DEBUG:
        _plt = (
            load_ca.group_by(pl.col("datetime").dt.month())
            .mean()
            .plot.line(x="datetime", y="active_load")
            .properties(title="Average Load by Month in California")
        )
    _plt
    return


@app.cell
def _(LOAD_DATA_OUT, df5):
    # Save processed data
    df5.sink_parquet(LOAD_DATA_OUT)
    return


if __name__ == "__main__":
    app.run()
