import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    alt.data_transformers.enable("vegafusion")
    return mo, pl


@app.cell
def _():
    GENERATOR_DATA = "../raw_data/preprocessed/generators.parquet"
    HOURLY_GENERATION_DATA = "../raw_data/downloads/CATS_generation.csv"
    TYPE_MAPPING = "../raw_data/constants/map_type_to_vcf_type.csv"
    YEARLY_LIMIT_OUTPUT = "../raw_data/preprocessed/yearly_limits.parquet"
    VCF_OUTPUT = "../raw_data/preprocessed/variable_capacity_factors.parquet"
    return (
        GENERATOR_DATA,
        HOURLY_GENERATION_DATA,
        TYPE_MAPPING,
        VCF_OUTPUT,
        YEARLY_LIMIT_OUTPUT,
    )


@app.cell
def _(TYPE_MAPPING, pl):
    df_type_mapping = pl.read_csv(TYPE_MAPPING)
    df_type_mapping
    VCF_TYPES = df_type_mapping["vcf_type"].unique().to_list()
    return VCF_TYPES, df_type_mapping


@app.cell
def _(GENERATOR_DATA, pl):
    gen = pl.read_parquet(GENERATOR_DATA, columns=["gen_id", "type", "Pmax"])
    gen
    return (gen,)


@app.cell
def _(HOURLY_GENERATION_DATA, pl):
    df = pl.read_csv(HOURLY_GENERATION_DATA, null_values=["NA", "#VALUE!"])
    df = df.with_columns(pl.col("Date").str.to_datetime("%d-%m-%Y %H:%M"))
    df = df.rename(lambda c: c.lower()).rename({"date": "datetime"})
    df
    return (df,)


@app.cell
def _(YEARLY_LIMIT_OUTPUT, df, pl):
    # get large hydro upper limit
    max_energy_genearation = pl.DataFrame(
        {"type": ["Hydropower"], "limit": [df.get_column("large hydro").sum()]}
    )
    max_energy_genearation.write_parquet(YEARLY_LIMIT_OUTPUT)
    max_energy_genearation
    return


@app.cell
def _(df_type_mapping, gen, pl):
    max_capacity = gen.join(df_type_mapping, on="type", how="inner")
    max_capacity = max_capacity.group_by("vcf_type").agg(pl.col("Pmax").sum())
    max_capacity
    return (max_capacity,)


@app.cell
def _(VCF_TYPES, df):
    df2 = df.select(["datetime"] + VCF_TYPES)
    df2
    return (df2,)


@app.cell
def _(VCF_TYPES, df2, max_capacity, pl):
    df3 = df2
    for t in VCF_TYPES:
        df3 = df3.with_columns(
            pl.col(t) / max_capacity.filter(vcf_type=t).get_column("Pmax")
        )
    df3 = df3.unpivot(
        index=["datetime"], variable_name="vcf_type", value_name="capacity_factor"
    )
    df3
    return (df3,)


@app.cell
def _(mo):
    mo.md(r"""
    # Plot capacity factors
    """)
    return


@app.cell
def _(df3, pl):
    _cf_by_hour = (
        df3.group_by(pl.col("datetime").dt.hour().alias("hour"), "vcf_type")
        .mean()
        .sort("hour")
        .drop("datetime")
    )
    _cf_by_hour.plot.line(x="hour", y="capacity_factor", color="vcf_type").properties(
        title="Capacity Factors by Hour"
    )
    return


@app.cell
def _(df3, pl):
    _cf_by_hour = (
        df3.group_by(pl.col("datetime").dt.month().alias("month"), "vcf_type")
        .mean()
        .sort("month")
        .drop("datetime")
    )
    _cf_by_hour.plot.line(x="month", y="capacity_factor", color="vcf_type").properties(
        title="Capacity Factors by Month"
    )
    return


@app.cell
def _(VCF_TYPES, df3):
    dist_bucket_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plot = None
    for _t in VCF_TYPES:
        subplot = (
            df3.filter(vcf_type=_t)
            .get_column("capacity_factor")
            .hist(bins=dist_bucket_edges)
            .plot.bar(x="breakpoint:O", y="count")
            .properties(title=f"{_t.title()} Capacity Factor Distribution")
        )
        if plot is None:
            plot = subplot
        else:
            plot |= subplot
    plot
    return


@app.cell
def _(VCF_OUTPUT, df3):
    df3.write_parquet(VCF_OUTPUT)
    df3
    return


if __name__ == "__main__":
    app.run()
