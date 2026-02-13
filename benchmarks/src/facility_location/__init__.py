def size_to_num_variables(col_name):
    import polars as pl

    n = pl.col(col_name)
    return (
        1  # max_distance
        + 2 * n  # facility_position
        + (n + 1) ** 2 * n  # binary variables
        + 2 * n * (n + 1)  # dist_x, dist_y
        + (n + 1) ** 2 * n  # dist
    )
