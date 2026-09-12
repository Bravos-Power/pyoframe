import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import altair as alt
    import polars as pl

    alt.data_transformers.enable("vegafusion")
    return (pl,)


@app.cell
def _():
    PAYBACK_PERIOD_YEARS = 20
    BASE_MW = 100
    return BASE_MW, PAYBACK_PERIOD_YEARS


@app.cell
def _():
    NREL_DATA = "../raw_data/downloads/NREL_ATB_data.parquet"
    OUTPUT_PATH = "../raw_data/preprocessed/capex_costs.csv"
    return NREL_DATA, OUTPUT_PATH


@app.cell
def _(NREL_DATA, pl):
    df = pl.read_parquet(NREL_DATA).cast({"default": pl.Boolean})
    df
    return (df,)


@app.cell
def _(PAYBACK_PERIOD_YEARS, df, pl):
    df_filter = (
        df.filter(
            core_metric_parameter="CAPEX",
            core_metric_case="R&D",
            maturity="Y",
            scenario="Moderate",
            core_metric_variable=2030,
            scale="Utility",
            crpyears=str(PAYBACK_PERIOD_YEARS),
        )
        .with_columns(has_default=pl.max("default").over("technology").cast(pl.Boolean))
        .filter((pl.col("has_default") & pl.col("default")) | (~pl.col("has_default")))
    )

    df_filter
    return (df_filter,)


@app.cell
def _(BASE_MW, PAYBACK_PERIOD_YEARS, df_filter, pl):
    df_clean = df_filter
    for c in df_clean.columns:
        if df_clean[c].unique().len() == 1:
            df_clean = df_clean.drop(c)

    df_clean = (
        df_clean.group_by("technology_alias")
        .agg(pl.col("value").mean())
        .rename({"technology_alias": "type"})
    )

    df_clean = df_clean.with_columns(
        hourly_capex_cost_per_KW=pl.col("value") / PAYBACK_PERIOD_YEARS / 365 / 24
    ).drop("value")

    df_clean = df_clean.with_columns(
        (pl.col("hourly_capex_cost_per_KW") * BASE_MW).alias(
            "hourly_capex_cost_k_per_pu"
        )
    ).drop("hourly_capex_cost_per_KW")
    df_clean
    return (df_clean,)


@app.cell
def _(df_clean, pl):
    MAPPING = {
        "Land-Based Wind": "Wind",
        "Utility PV": "Solar PV",
    }

    DROP = [
        "Utility-Scale Battery Storage",
        "Pumped Storage Hydropower",
        "Utility-Scale PV-Plus-Battery",
        "Offshore Wind",
        "Midsize DW",
        "Large DW",
    ]

    df_final = df_clean.with_columns(
        pl.col("type").replace(MAPPING).str.strip_chars()
    ).filter(~pl.col("type").is_in(DROP))
    df_final
    return (df_final,)


@app.cell
def _(OUTPUT_PATH, df_final):
    df_final.sort(df_final.columns).write_csv(OUTPUT_PATH)
    return


if __name__ == "__main__":
    app.run()
