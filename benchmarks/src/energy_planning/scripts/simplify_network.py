"""Simplifies the network topology using various tricks and identifies leaf nodes.

Tricks are:
    - combining lines in parallel,
    - removing lines that lead nowhere,
    - and removing intermediary buses with no loads or generators.

Line ratings and reactances are updated accordingly.
Line IDs are roughly preserved; merged lines take on the IDs of the original lines.
Node IDs are preserved, but the number of nodes is reduced.

As of Aug 12, 2025, this script reduces the number of lines by 14.5%, mainly due to lines in series.

Leaf nodes are nodes containing either loads or generators that can be wholly disconnected from the
grid by a single contingency. They are flagged so such contingencies are not considered.
"""

import polars as pl
from benchmark_utils import mock_snakemake

f_bus = pl.col("from_bus")
t_bus = pl.col("to_bus")
voltage = pl.col("voltage_kv")
EXPECTED_COLS = ["line_id", f_bus, t_bus, "reactance", "line_rating_MW", voltage]


def get_buses_degree(
    lines: pl.DataFrame, degree: int, exclude: pl.Series | None = None
) -> pl.Series:
    """Returns a Series of buses with degree `degree` that are not in `exclude`."""
    lines = (
        pl.concat([lines["from_bus"], lines["to_bus"]])
        .value_counts(name="degree")
        .filter(degree=degree)
    )
    if exclude is not None:
        lines = lines.filter(~f_bus.is_in(exclude.implode()))

    return lines.get_column("from_bus")


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
        .group_by([f_bus, t_bus, voltage])
        .agg(
            pl.col("reactance").sum(),
            pl.col("line_id").min(),
            pl.col("line_rating_MW").sum(),
        )
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
    """Combine lines in series (e.g., a line going from bus 1 to bus 2 with a line going from bus 2 to bus 3).

    This is done by doing two types of combinations:
    1. Combining line series of length 2 (e.g. A-B-C becomes A-C).
    2. Combining the first two lines in a series of length 3 or more (in series of length 3, A-B-C-D becomes A-C-D; in series of length 4 or more A-B-C-D-...-X-Y-Z becomes A-C-D-...-X-Z).
    Note that this second type of operation doesn't fully combine the series, but if this function is called repeatedly, eventually all series will be combined.
    """
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
            pl.col("line_id").min(),
            f_bus.first(),
            to_bus=f_bus.last(),
            line_rating_MW=pl.col("line_rating_MW").min(),
            reactance=pl.col("reactance").sum(),
            # voltage should be identical so we could just use .first(), but if voltage has issue .mean() will reveal the problem
            voltage_kv=voltage.mean(),
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
        "line_id",
        f_bus,
        pl.col("from_bus_right").alias("to_bus"),
        pl.min_horizontal("line_rating_MW", "line_rating_MW_right").alias(
            "line_rating_MW"
        ),
        pl.sum_horizontal("reactance", "reactance_right").alias("reactance"),
        voltage,
    )

    initial = lines.height
    lines = pl.concat(
        [
            l_keep.select(EXPECTED_COLS),
            l_edge_short.select(EXPECTED_COLS),
            l_edge_long.select(EXPECTED_COLS),
            l_edge_long_uncombined.select(EXPECTED_COLS),
            l_middle_uncombined.select(EXPECTED_COLS),
        ]
    )
    return lines, initial - lines.height


def remove_self_connections(lines):
    """Remove lines that connect a bus to itself (e.g., a line going from bus 1 to bus 1)."""
    initial = lines.height
    lines = lines.filter(f_bus != t_bus)
    return lines, initial - lines.height


def identify_leafs(lines, buses_to_keep):
    """Flags lines that are at the very edge of the grid.

    These lines will later be excluded from branch outages since they would effectively split the network.
    """
    lines = lines.with_columns(is_leaf=pl.lit(False))
    passes = 0

    while True:
        leaf_buses = get_buses_degree(lines.filter(~pl.col("is_leaf")), 1)
        if len(leaf_buses) == 0:
            break
        if passes == 0:
            assert leaf_buses.is_in(buses_to_keep.implode()).all(), (
                "Dead end buses were not properly removed"
            )

        lines = lines.with_columns(
            is_leaf=(
                pl.col("is_leaf")
                | f_bus.is_in(leaf_buses.implode())
                | t_bus.is_in(leaf_buses.implode())
            )
        )
        passes += 1
    print(
        f"Detected {lines.filter('is_leaf').height} leaf buses after {passes} passes."
    )
    return lines


def main(lines, buses_to_keep):
    """Simplify the network until there is no more simplification possible."""
    lines = lines.select(*EXPECTED_COLS)

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

    lines = identify_leafs(lines, buses_to_keep)

    return lines


if __name__ == "__main__":
    if "snakemake" not in globals() or hasattr(snakemake, "mock"):  # noqa: F821
        snakemake = mock_snakemake("simplify_network")

    lines = pl.read_parquet(snakemake.input.lines)
    num_buses = len(lines["from_bus"].append(lines["to_bus"]).unique())
    print(f"Network has {lines.height} lines and {num_buses} buses.")

    gens = pl.scan_parquet(snakemake.input.generators)
    loads = pl.scan_parquet(snakemake.input.loads)
    transformers = lines.filter(pl.col("transformer")).lazy()
    buses_to_keep = (
        pl.concat(
            [
                gens.select("bus"),
                loads.select("bus"),
                transformers.select(bus=f_bus),
                transformers.select(bus=t_bus),
            ],
            how="vertical_relaxed",
        )
        .unique()
        .collect()
        .to_series()
        .sort()
    )
    print(
        f"{len(buses_to_keep)} buses with loads or generators that should not be removed."
    )

    lines = main(lines, buses_to_keep)

    num_buses_simplified = len(lines["from_bus"].append(lines["to_bus"]).unique())

    print(
        f"Simplified network has {lines.height} lines and {num_buses_simplified} buses."
    )

    lines.write_parquet(snakemake.output[0])
