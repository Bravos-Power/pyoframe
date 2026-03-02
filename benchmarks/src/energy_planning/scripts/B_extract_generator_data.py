import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    alt.data_transformers.enable("vegafusion")
    return alt, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    ## Merge CATS and MATPOWER data
    """)
    return


@app.cell
def _():
    CATS_DATA = "../raw_data/downloads/CATS_generators.csv"
    MATPOWER_DATA = "../raw_data/preprocessed/matpower_gen.parquet"
    OUTPUT_PATH = "../raw_data/preprocessed/generators.parquet"
    return CATS_DATA, MATPOWER_DATA, OUTPUT_PATH


@app.cell
def _(CATS_DATA, pl):
    # Load cats data
    cats_df = pl.read_csv(CATS_DATA, encoding="iso-8859-1")
    cats_df
    return (cats_df,)


@app.cell
def _(cats_df, pl):
    # Clean CATS data

    # Order matters, lets add this before anything else happens
    cats_df_clean = cats_df.with_row_index(name="gen_id")

    # file has whitespaces that we must strip
    cats_df_clean.columns = [c.strip() for c in cats_df_clean.columns]
    cats_df_clean = cats_df_clean.with_columns(
        pl.col(
            c
            for c, t in zip(cats_df_clean.columns, cats_df_clean.dtypes)
            if (t == pl.String)
        ).str.strip_chars()
    )

    # now that they're removed we can convert to numbers
    cats_df_clean = cats_df_clean.with_columns(
        pl.col("PlantCode").cast(pl.Int64),
        pl.col("Lat").cast(pl.Float64),
        pl.col("Lon").cast(pl.Float64),
        pl.col("bus").cast(pl.Int64),
    )
    cats_df_clean
    return (cats_df_clean,)


@app.cell
def _(MATPOWER_DATA, cats_df_clean, pl):
    # Load matpower data and check that they line up
    matpower_df = pl.read_parquet(MATPOWER_DATA)
    matpower_df

    from polars.testing import assert_frame_equal

    assert_frame_equal(matpower_df[["gen_id", "bus"]], cats_df_clean[["gen_id", "bus"]])
    return assert_frame_equal, matpower_df


@app.cell
def _(assert_frame_equal, cats_df_clean, matpower_df):
    # Now let's merge with the matpower data to get costs
    joined_df = cats_df_clean.join(
        matpower_df, on=["gen_id", "bus"], how="full", coalesce=True, validate="1:1"
    )

    # Check that these columns are identical across both DFs, then remove
    duplicate_cols = ["Qmax", "Qmin", "Pg", "Pmax"]
    for c in duplicate_cols:
        assert_frame_equal(
            joined_df[[c]], joined_df[[f"{c}_right"]].rename({f"{c}_right": c})
        )
        joined_df = joined_df.drop(f"{c}_right")

    joined_df
    return (joined_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Analyze and clean data
    """)
    return


@app.cell
def _(joined_df, pl):
    # Remove reactive elements
    assert (joined_df["Pmin"] == 0).all()
    df_clean1 = joined_df.drop("Pmin")

    reactive_elements = df_clean1.filter(pl.col("Pmax") == 0)
    assert (
        reactive_elements["FuelType"].is_null().all()
        and (reactive_elements["Qmax"] != 0).all()
    ), "Expected reactive elements to have Pmax == 0 and Qmax != 0"
    df_clean1 = df_clean1.filter(pl.col("Pmax") > 0).drop("Qmax", "Qmin", "Qg")
    df_clean1
    return (df_clean1,)


@app.cell
def _(df_clean1, pl):
    # Plot Pg and ultimately drop it since it only represents an hour (as evident by the constant capacity factor)
    _plot = df_clean1.with_columns(
        capacity_factor=pl.col("Pg") / pl.col("Pmax")
    ).plot.scatter(x="Pmax", y="capacity_factor", color="FuelType")
    df_clean2 = df_clean1.drop("Pg")
    _plot
    return (df_clean2,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Cost analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Costs were constructed in
    >  J. M. Snodgrass, "Tractable Algorithms for Constructing Electric Power Network Models," Ph.D. Thesis, The University of Wisconsin - Madison, 2021. https://github.com/WISPO-POP/SyntheticElectricNetworkModels/blob/master/Synthetic%20Generator%20Cost%20Curves/Creating%20synthetic%20generator%20cost%20curves.pdf

    We confirm that often the fixed overhead costs (`cost_per_h`) often scale with capacity so we choose to integrate them into the capacity expansion (CAPEX) costs.

    Meanwhile, the cost curves show little curvature meaning that a linear approximation is good enough (rather than a quadratic cost curve).
    """)
    return


@app.cell
def _(alt, df_clean2):
    # Plot the various costs. Notice how fixed overheads (cost_per_h) often scale with capacity based on assumptions from data cons
    x_axis = alt.X("Pmax", title="Max Capacity (MW)")
    (
        df_clean2.plot.scatter(x=x_axis, y="cost_per_h", color="FuelType")
        | df_clean2.plot.scatter(x=x_axis, y="cost_per_MWh", color="FuelType")
        | df_clean2.plot.scatter(x=x_axis, y="cost_per_(MW)^2_h", color="FuelType")
    )
    return


@app.cell
def _(alt, df_clean2, pl):
    # Now ignoring fixed costs, plot normalized cost curves. Notice how the quadratic component is quite small!
    import numpy as np

    x_vals = list(np.linspace(0, 1, 11))

    assert (df_clean2["cost_per_(MW)^2_h"] >= 0).all()
    cost_analysis = (
        df_clean2.with_columns(
            cost_Pmax=pl.col("cost_per_(MW)^2_h") * pl.col("Pmax") ** 2
            + pl.col("cost_per_MWh") * pl.col("Pmax"),
        )
        .filter(pl.col("cost_Pmax") != 0)
        .with_columns(
            cost_a=pl.col("cost_per_(MW)^2_h")
            * pl.col("Pmax") ** 2
            / pl.col("cost_Pmax"),
            cost_b=pl.col("cost_per_MWh") * pl.col("Pmax") / pl.col("cost_Pmax"),
        )
        .unique(["cost_a", "cost_b", "cost_per_h", "FuelType"])
        .sort("cost_a", descending=True)
        .sample(100)
        .with_columns(x=pl.lit(x_vals))
        .explode("x")
        .with_columns(
            y=pl.col("cost_a") * pl.col("x") ** 2 + pl.col("cost_b") * pl.col("x"),
            Px=pl.col("Pmax") * pl.col("x"),
        )
    )
    cost_analysis.plot.line(
        x=alt.X("x", title="Fraction of Pmax"),
        y=alt.Y("y", title="Fraction of max cost"),
        detail="gen_id",
        color="FuelType",
    ).mark_line(strokeWidth=0.2)
    return


@app.cell
def _(alt, df_clean2, pl):
    # Develop linear approximation and hourly overhead per MW capacity
    df_clean3 = df_clean2.with_columns(
        cost_per_MWh_linear=pl.col("cost_per_MWh")
        + pl.col("cost_per_(MW)^2_h") * pl.col("Pmax"),
        hourly_overhead_per_MW_capacity=pl.col("cost_per_h") / pl.col("Pmax"),
    ).drop("cost_per_h")
    (
        df_clean3.plot.scatter(x="Pmax", y="cost_per_MWh_linear", color="FuelType")
        | df_clean3.plot.scatter(
            x="Pmax",
            y=alt.Y("hourly_overhead_per_MW_capacity", scale=alt.Scale(type="symlog")),
            color="FuelType",
        )
    )
    return (df_clean3,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Fuel merging
    """)
    return


@app.cell
def _(alt, df_clean3):
    # Inspect fuel types
    df_clean3.group_by("FuelType").sum().plot.bar(
        x=alt.X("FuelType", sort=alt.SortField(field="Pmax", order="descending")),
        y=alt.Y("Pmax", title="Total Capacity (MW)"),
        color="FuelType",
    )
    return


@app.cell
def _(alt, df_clean3, pl):
    # Merge fuel types and remove storage as well as 'all other'

    MAPPING = {
        "Wood/Wood Waste Biomass": "Biopower",
        "Municipal Solid Waste": "Biopower",
        "Other Waste Biomass": "Biopower",
        "Landfill Gas": "Biopower",
        "Solar Photovoltaic": "Solar PV",
        "Solar Thermal without Energy Storage": "CSP",
        "Conventional Hydroelectric": "Hydropower",
        "Onshore Wind Turbine": "Wind",
        "Conventional Steam Coal": "Coal",
        "Natural Gas Fired Combined Cycle": "Natural Gas",
        "Natural Gas Fired Combustion Turbine": "Natural Gas",
        "Natural Gas Steam Turbine": "Natural Gas",
        "Natural Gas Internal Combustion Engine": "Natural Gas",
    }

    DROP_TYPES = [
        "Batteries",
        "All Other",
        "Hydroelectric Pumped Storage",
        "Petroleum Coke",
        "Other Gases",
        "Petroleum Liquids",
    ]

    df_clean4 = df_clean3.with_columns(pl.col("FuelType").replace(MAPPING)).rename(
        {"FuelType": "type"}
    )

    # Remove storage to simplify model and 'all other' since it's negligible
    prior_capacity = df_clean4["Pmax"].sum()
    df_clean4 = df_clean4.filter(~pl.col("type").is_in(DROP_TYPES))
    final_capacity = df_clean4["Pmax"].sum()
    all_other_proportion = (prior_capacity - final_capacity) / prior_capacity

    df_clean4.group_by("type").sum().plot.bar(
        x=alt.X("type", sort=alt.SortField(field="Pmax", order="descending")),
        y=alt.Y("Pmax", title="Total Capacity (MW)"),
        color="type",
    )
    return all_other_proportion, df_clean4


@app.cell
def _(all_other_proportion):
    f"Removed {all_other_proportion:.2%} of total capacity"
    return


@app.cell
def _(df_clean4, pl):
    # Remove mBase since it's always 0
    assert (df_clean4["mBase"] == 0).all()
    df_clean5 = df_clean4.drop("mBase")

    # Merge PlantCode and GenID
    _SEP = "/"
    assert df_clean5["GenID"].str.contains(_SEP).sum() == 0, (
        f"Cannot use {_SEP} as separator since GenID contains it"
    )
    df_clean5 = df_clean5.with_columns(
        pl.concat_str(
            [pl.col("PlantCode").cast(pl.Utf8), pl.col("GenID").cast(pl.Utf8)],
            separator=_SEP,
        ).alias("PlantAndGenID")
    ).drop("PlantCode", "GenID")

    df_clean5
    return (df_clean5,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Aggregate like generators
    """)
    return


@app.cell
def _(df_clean5, pl):
    # Fuse plants that are identical and have costs
    _unique_cols = [
        "bus",
        "type",
        "cost_per_MWh_linear",
        "hourly_overhead_per_MW_capacity",
    ]
    df_clean6 = df_clean5.group_by(_unique_cols).agg(
        pl.col("gen_id").min(),
        pl.col("Pmax").sum(),
        pl.col("PlantAndGenID").unique(),
        pl.col("Lat", "Lon").mean(),
    )

    (
        f"Reduced number of dispatchable generators by {len(df_clean5) - len(df_clean6)} by fusing identical plants.",
        df_clean6,
    )
    return (df_clean6,)


@app.cell
def _(df_clean6, pl):
    # Produce summary table
    summary_table = (
        df_clean6.group_by("type")
        .agg(
            n=pl.len(),
            total_capacity_MW=pl.col("Pmax").sum().round(0),
            median_cost_per_MWh_linear=pl.col("cost_per_MWh_linear").median().round(0),
            meidan_capex_overhead_per_MWh=pl.col("hourly_overhead_per_MW_capacity")
            .median()
            .round(0),
        )
        .sort("total_capacity_MW", descending=True)
    )
    summary_table, summary_table.select("n", "total_capacity_MW").sum()
    return


@app.cell
def _(df_clean6):
    assert df_clean6["gen_id"].is_unique().all()
    return


@app.cell
def _(OUTPUT_PATH, df_clean6):
    df_clean6.write_parquet(OUTPUT_PATH)
    return


if __name__ == "__main__":
    app.run()
