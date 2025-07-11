"""
Simplifies the network topology by
    - combining lines in parallel,
    - removing lines that lead nowhere,
    - and removing intermediary buses with no loads or generators.

Line ratings and reactances are updated accordingly.

As of June 25, 2025, this script reduces the number of lines by 25%.
"""

import polars as pl

from benchmarks.util import mock_snakemake

f_bus = pl.col("from_bus")
t_bus = pl.col("to_bus")


def get_buses_degree(lines: pl.DataFrame, degree: int, exclude: pl.Series) -> pl.Series:
    """Returns a Series of buses with degree `degree` that are not in `exclude`."""
    return (
        pl.concat([lines["from_bus"], lines["to_bus"]])
        .value_counts(name="degree")
        .filter(pl.col("degree") == degree)
        .filter((~f_bus.is_in(exclude.implode())))
        .get_column("from_bus")
    )


def remove_dead_ends(lines, buses_to_keep):
    """Remove lines going from or to a bus with degree 1 (no other connections) since the line serves no purpose."""
    dead_end_buses = get_buses_degree(lines, degree=1, exclude=buses_to_keep)

    return lines.filter(
        ~f_bus.is_in(dead_end_buses.implode()) & ~t_bus.is_in(dead_end_buses.implode())
    ), len(dead_end_buses)


def combine_parallel_lines(lines: pl.DataFrame):
    """Combine lines in parallel (e.g., two lines going from bus 1 to bus 2)."""
    initial = lines.height
    lines = swap_direction(lines, condition=f_bus > t_bus)
    lines = (
        lines.with_columns((1 / pl.col("reactance")).alias("reactance"))
        .group_by([f_bus, t_bus])
        .sum()
        .with_columns((1 / pl.col("reactance")).alias("reactance"))
    )
    return lines, initial - lines.height


def swap_direction(
    lines: pl.DataFrame,
    condition: pl.Expr,
    from_col=f_bus,
    to_col=t_bus,
):
    """Swap the direction of lines based on a condition."""
    return lines.with_columns(
        pl.when(condition)
        .then(pl.struct(from_bus=to_col, to_bus=from_col))
        .otherwise(pl.struct(from_col, to_col))
        .struct.unnest()
    )


def combine_sequential_line(lines: pl.DataFrame, buses_to_keep: pl.Series):
    """
    Combine lines in series (e.g., a line going from bus 1 to bus 2 with a line going from bus 2 to bus 3).

    This is done by doing two types of combinations:
    1. Combining line series of length 2 (e.g. A-B-C becomes A-C).
    2. Combining the first two lines in a series of length 3 or more (in series of length 3, A-B-C-D becomes A-C-D; in series of length 4 or more A-B-C-D-...-X-Y-Z becomes A-C-D-...-X-Z).
    Note that this second type of operation doesn't fully combine the series, but if this function is called repeatedly, eventually all series will be combined."""
    bus_to_remove = get_buses_degree(lines, degree=2, exclude=buses_to_keep).implode()

    l_edge = lines.filter(f_bus.is_in(bus_to_remove) ^ t_bus.is_in(bus_to_remove))
    l_middle = lines.filter(f_bus.is_in(bus_to_remove) & t_bus.is_in(bus_to_remove))
    l_keep = lines.filter(~f_bus.is_in(bus_to_remove) & ~t_bus.is_in(bus_to_remove))

    # order such that the series bus is in "to_bus"
    l_edge = swap_direction(l_edge, condition=f_bus.is_in(bus_to_remove))

    l_edge_short = l_edge.filter(t_bus.is_duplicated())
    l_edge_long = l_edge.filter(~t_bus.is_duplicated())

    l_edge_short = (
        l_edge_short.rename({"to_bus": "mid_bus"})
        .group_by("mid_bus")
        .agg(
            from_bus=f_bus.first(),
            to_bus=f_bus.last(),
            line_rating=pl.col("line_rating").min(),
            reactance=pl.col("reactance").sum(),
        )
        .drop("mid_bus")
    )

    l_middle = swap_direction(
        l_middle, condition=~t_bus.is_in(l_edge_long["to_bus"].implode())
    )

    l_edge_long_uncombined = l_edge_long.join(l_middle, on=t_bus, how="anti")
    l_edge_long = l_edge_long.join(l_middle, on=t_bus, how="inner")
    l_middle_uncombined = l_middle.join(l_edge_long, on=t_bus, how="anti")

    l_edge_long = l_edge_long.select(
        f_bus,
        pl.col("from_bus_right").alias("to_bus"),
        pl.min_horizontal("line_rating", "line_rating_right").alias("line_rating"),
        pl.sum_horizontal("reactance", "reactance_right").alias("reactance"),
    )

    initial = lines.height
    lines = pl.concat(
        [
            l_keep,
            l_edge_short,
            l_edge_long,
            l_edge_long_uncombined,
            l_middle_uncombined,
        ]
    )
    return lines, initial - lines.height


def remove_self_connections(lines):
    """Remove lines that connect a bus to itself (e.g., a line going from bus 1 to bus 1)."""
    initial = lines.height
    lines = lines.filter(f_bus != t_bus)
    return lines, initial - lines.height


def main(lines, buses_to_keep):
    """Simplify the network until there is no more simplification possible."""
    expected_cols = {"from_bus", "to_bus", "reactance", "line_rating"}
    assert set(lines.columns) == expected_cols, (
        f"Unexpected columns in lines DataFrame ({set(lines.columns) - expected_cols})"
    )

    num_lines_initial = lines.height
    lines_removed = 0
    lines_removed_prev = float("-inf")
    while lines_removed > lines_removed_prev:
        lines_removed_prev = lines_removed
        lines, lines_merged = combine_parallel_lines(lines)
        if lines_merged != 0:
            print(f"Removed {lines_merged} lines in parallel...")

        lines, dead_ends = remove_dead_ends(lines, buses_to_keep)
        if dead_ends != 0:
            print(f"Removed {dead_ends} dead-end lines...")

        lines_concatenated = 0
        while True:
            lines, new_lines_concatenated = combine_sequential_line(
                lines, buses_to_keep
            )
            if new_lines_concatenated == 0:
                break
            lines_concatenated += new_lines_concatenated
        if lines_concatenated != 0:
            print(f"Removed {lines_concatenated} lines in series...")

        lines, self_connections = remove_self_connections(lines)
        if self_connections != 0:
            print(f"Removed {self_connections} lines connected to themselves...")

        lines_removed += (
            dead_ends + lines_merged + lines_concatenated + self_connections
        )

    print(
        f"Removed {lines_removed / num_lines_initial:.2%} of lines in total (l={lines.height})."
    )

    return lines


if __name__ == "__main__":
    if "snakemake" not in globals() or hasattr(snakemake, "mock"):  # noqa: F821
        snakemake = mock_snakemake("simplify_network")

    gens = pl.scan_parquet(snakemake.input.generators)
    loads = pl.scan_parquet(snakemake.input.loads)
    buses_to_keep = (
        pl.concat([gens.select("bus"), loads.select("bus")])
        .unique()
        .collect()
        .to_series()
        .sort()
    )
    print(
        f"{len(buses_to_keep)} buses with loads or generators that should not be removed."
    )

    lines = pl.read_parquet(snakemake.input.lines)

    lines = main(lines, buses_to_keep)
    lines.write_parquet(snakemake.output[0])
