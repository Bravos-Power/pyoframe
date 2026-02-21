import marimo

__generated_with = "0.19.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compute Power Transfer Distribution Factors
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from scipy.sparse import coo_array, diags_array
    from sksparse.cholmod import cholesky

    return cholesky, coo_array, diags_array, mo, pl


@app.cell
def _():
    DF_CUTOFF = 1e-10
    return (DF_CUTOFF,)


@app.cell
def _():
    LINES_DATA = "../raw_data/preprocessed/lines_simplified.parquet"
    OUTPUT_PATH = "../raw_data/preprocessed/power_transfer_dist_facts.parquet"
    return LINES_DATA, OUTPUT_PATH


@app.cell
def _(LINES_DATA, pl):
    lines = pl.read_parquet(
        LINES_DATA, columns=["line_id", "from_bus", "to_bus", "reactance"]
    )
    L = lines.height
    return L, lines


@app.cell
def _(lines):
    # Step 1: Renumber lines from 1 to L contiguously.
    lines_1 = lines.sort("line_id")
    lines_1 = lines_1.with_row_index(name="line_index")
    lines_map = lines_1.select("line_id", "line_index")
    lines_1 = lines_1.drop("line_id")
    lines_1
    return lines_1, lines_map


@app.cell
def _(lines_1, pl):
    # Step 2: Renumber buses such that they go from 1 to N contiguously.
    bus_map = (
        pl.concat([lines_1["from_bus"], lines_1["to_bus"]])
        .unique()
        .sort()
        .rename("bus")
        .to_frame()
        .with_row_index(name="bus_index")
    )
    N = bus_map.height
    lines_2 = (
        lines_1.join(bus_map, left_on="from_bus", right_on="bus", coalesce=True)
        .drop("from_bus")
        .rename({"bus_index": "from_bus"})
        .join(bus_map, left_on="to_bus", right_on="bus", coalesce=True)
        .drop("to_bus")
        .rename({"bus_index": "to_bus"})
    )
    lines_2
    return N, bus_map, lines_2


@app.cell
def _(N, lines_2, pl):
    # Step 3: Form an admittance matrix in COO form (row_index (i), col_index (j), value)
    lines_3 = lines_2.with_columns(suscep=1 / lines_2["reactance"])
    off_diagonal = pl.concat(
        [
            lines_3.select(
                pl.col("from_bus").alias("i"),
                pl.col("to_bus").alias("j"),
                -pl.col("suscep").alias("val"),
            ),
            lines_3.select(
                pl.col("to_bus").alias("i"),
                pl.col("from_bus").alias("j"),
                -pl.col("suscep").alias("val"),
            ),
        ]
    )
    diagonal = (
        pl.concat(
            [
                lines_3.select("suscep", i=pl.col("from_bus")),
                lines_3.select("suscep", i=pl.col("to_bus")),
            ]
        )
        .group_by("i")
        .agg(pl.sum("suscep").alias("val"))
        .select("i", pl.col("i").alias("j"), "val")
    )
    assert len(diagonal) == N, "Diagonal entries should match the number of buses."
    admittance_matrix_df = pl.concat([off_diagonal, diagonal])
    admittance_matrix_df
    return admittance_matrix_df, lines_3


@app.cell
def _(N, admittance_matrix_df, coo_array, pl):
    # Step 4: Drop last row and column (slack bus).
    admittance_matrix_df_1 = admittance_matrix_df.filter(
        pl.col("i") < N - 1, pl.col("j") < N - 1
    )
    admittance_matrix = coo_array(
        (
            admittance_matrix_df_1["val"].to_numpy(),
            (
                admittance_matrix_df_1["i"].cast(pl.Float64).to_numpy(),
                admittance_matrix_df_1["j"].cast(pl.Float64).to_numpy(),
            ),
        ),
        shape=(N - 1, N - 1),
    )
    admittance_matrix
    return (admittance_matrix,)


@app.cell
def _(admittance_matrix, cholesky):
    # Step 5: Inverse the admittance matrix to get voltage angles.
    # We use the sparse Cholesky factorization since, based on my tests, this is much faster than using any of the following:
    # scipy.linalg.inv, scipy.sparse.linalg.inv, scipy.linag.pinv, scipy.sparse.linalg.spsolve
    factor = cholesky(admittance_matrix.tocsc())
    voltage_angles = factor.inv()
    voltage_angles
    return (voltage_angles,)


@app.cell
def _(L, N, coo_array, diags_array, lines_3, pl, voltage_angles):
    # Step 6: Calculate the power flow by multiplying the voltage angles by diag(S)*A^T
    # A^T is the L by N adjacency matrix
    adjacency_matrix_T = pl.concat(
        [
            lines_3.select(
                i=pl.col("line_index"), j=pl.col("from_bus"), val=pl.lit(1)
            ).filter(pl.col("j") < N - 1),
            lines_3.select(
                i=pl.col("line_index"), j=pl.col("to_bus"), val=pl.lit(-1)
            ).filter(pl.col("j") < N - 1),
        ]
    )
    adjacency_matrix_T = coo_array(
        (
            adjacency_matrix_T["val"].to_numpy(),
            (adjacency_matrix_T["i"].to_numpy(), adjacency_matrix_T["j"].to_numpy()),
        ),
        shape=(L, N - 1),
    )
    diag_suscep = diags_array(lines_3.sort("line_index")["suscep"].to_numpy())
    power_flow = diag_suscep @ adjacency_matrix_T @ voltage_angles
    power_flow = power_flow.tocoo()
    power_flow_df = pl.DataFrame(
        {
            "injection": power_flow.col,
            "line_index": power_flow.row,
            "factor": power_flow.data,
        }
    )
    power_flow_df  # Exclude slack bus,  # Exclude slack bus
    return (power_flow_df,)


@app.cell
def _(DF_CUTOFF, pl, power_flow_df):
    # Step 7: Filter out small values
    power_flow_df_1 = power_flow_df.filter(pl.col("factor").abs() > DF_CUTOFF)
    power_flow_df_1
    return (power_flow_df_1,)


@app.cell
def _(bus_map, lines_map, power_flow_df_1):
    # Step 8: Unmap buses and lines to original IDs.
    power_flow_df_unmapped = (
        power_flow_df_1.join(
            bus_map,
            left_on="injection",
            right_on="bus_index",
            coalesce=True,
            validate="m:1",
        )
        .drop("injection")
        .rename({"bus": "injection"})
        .join(lines_map, on="line_index")
        .drop("line_index")
        .select("injection", "line_id", "factor")
        .sort("injection", "line_id")
    )
    power_flow_df_unmapped
    return (power_flow_df_unmapped,)


@app.cell
def _(OUTPUT_PATH, power_flow_df_unmapped):
    power_flow_df_unmapped.write_parquet(OUTPUT_PATH)
    return


if __name__ == "__main__":
    app.run()
