import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _():
    MATPOWER_CASE = "../raw_data/downloads/CaliforniaTestSystem.m"
    BUS_OUTPUT_PATH = "../raw_data/preprocessed/matpower_bus.parquet"
    BRANCH_OUTPUT_PATH = "../raw_data/preprocessed/matpower_branch.parquet"
    GEN_OUTPUT_PATH = "../raw_data/preprocessed/matpower_gen.parquet"
    return BRANCH_OUTPUT_PATH, BUS_OUTPUT_PATH, GEN_OUTPUT_PATH, MATPOWER_CASE


@app.cell
def _(MATPOWER_CASE):
    # Load the MATPOWER case file
    with open(MATPOWER_CASE) as f:
        matpower_case = f.read()
    matpower_case[:100]
    return (matpower_case,)


@app.cell
def _(matpower_case, pl):
    # Create tables for each section
    tables = {}
    ignored_sections = ("version", "baseMVA")
    header = None
    for i, section in enumerate(matpower_case.split("\nmpc.")):
        if i == 0:
            continue
        section_name, _, section_content = section.partition("=")
        section_content, _, next_header = section_content.partition(
            ";"
        )  # before first element nothing to do
        section_name = section_name.strip()
        section_content = section_content.strip("[]\n ")
        next_header = next_header.strip()
        if section_name in ignored_sections:
            header = next_header
            print(f"Ignoring: {section_name}")
            continue
        print(f"Processing: {section_name}")
        assert header is not None, f"Last section should not be: {section_name}"
        header = header.split("\n")
        header = [h.strip() for h in header]
        header = [h for h in header if h.startswith("%")]
        header = header[-1].strip("%").strip()
        header = header.split()
        section_content = [
            [val for val in row.strip().split("\t")]
            for row in section_content.split("\n")
        ]
        _table = pl.DataFrame(section_content, schema=header, orient="row")
        tables[section_name] = _table
        header = next_header
    tables
    return (tables,)


@app.cell
def _(pl, tables):
    # Clean tables (we do not expect any strings!)
    tables2 = {}
    for table_name, _table in tables.items():
        for _col in _table.columns:
            _table = _table.with_columns(pl.col(_col).cast(pl.Float64))
            if (_table[_col].round() == _table[_col]).all():
                _table = _table.with_columns(pl.col(_col).cast(pl.Int64))
            if _table[_col].is_in([0, 1]).all():
                _table = _table.with_columns(pl.col(_col).cast(pl.Boolean))
            _table = _table.with_columns(pl.col(_col).shrink_dtype())
        tables2[table_name] = _table
    tables2
    return (tables2,)


@app.cell
def _(tables2):
    assert {"bus", "gen", "branch", "gencost"} == set(tables2.keys())
    assert tables2["gen"].height == tables2["gencost"].height
    bus, gen, branch, gencost = (
        tables2["bus"],
        tables2["gen"],
        tables2["branch"],
        tables2["gencost"],
    )
    assert (gencost["n"] == 3).all()
    assert (gencost["2"] == 2).all()
    gencost = gencost.drop("2", "n").rename(
        {"c(n-1)": "cost_per_(MW)^2_h", "...": "cost_per_MWh", "c0": "cost_per_h"}
    )
    gencost
    return branch, bus, gen, gencost


@app.cell
def _(gen, gencost, pl):
    gen_merged = pl.concat(
        [gen.with_row_index(name="gen_id"), gencost], how="horizontal"
    )
    gen_merged
    return (gen_merged,)


@app.cell
def _(branch, bus, gen_merged):
    # Drop columns with a single unique value
    tables3 = (bus, branch, gen_merged)
    tables4 = []
    for _table in tables3:
        to_drop = []
        for _col in _table.columns:
            if len(_table[_col].unique()) == 1:
                to_drop.append(_col)
        tables4.append(_table.drop(*to_drop))
    bus_1, branch_1, gen_merged_1 = tables4
    bus_1, branch_1, gen_merged_1
    return branch_1, bus_1, gen_merged_1


@app.cell
def _(
    BRANCH_OUTPUT_PATH,
    BUS_OUTPUT_PATH,
    GEN_OUTPUT_PATH,
    branch_1,
    bus_1,
    gen_merged_1,
):
    bus_1.write_parquet(BUS_OUTPUT_PATH)
    branch_1.write_parquet(BRANCH_OUTPUT_PATH)
    gen_merged_1.write_parquet(GEN_OUTPUT_PATH)
    return


if __name__ == "__main__":
    app.run()
