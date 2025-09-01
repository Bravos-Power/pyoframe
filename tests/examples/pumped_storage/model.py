"""Pyoframe formulation of a pumped storage model.

Based on example at page 120 in book:
      N. Sudermann-Merx: Einführung in Optimierungsmodelle, Springer Nature, 2023
"""

from pathlib import Path

import polars as pl

import pyoframe as pf


def solve_model(
    use_var_names: bool = True, hourly_timestep: int = 6, **kwargs
) -> pf.Model:
    """Solve a pump storage model.

    Parameters:
        use_var_names: Whether to use variable names in the model.
        resolution_hours: By increasing the timestep, the model will be solved faster. The default is high so that the tests run faster.
        **kwargs: Additional arguments for the pyoframe Model.

    """
    hourly_prices = read_hourly_prices().filter(
        pl.col("tick").dt.hour() % hourly_timestep == 0
    )
    pump_max, turb_max = 70, 90
    storage_min, storage_max = 100, 630
    storage_level_init_and_final = 300
    efficiency = 0.75

    m = pf.Model(solver_uses_variable_names=use_var_names, **kwargs)
    m.Pump = pf.Variable(hourly_prices["tick"], vtype=pf.VType.BINARY)
    m.Turb = pf.Variable(hourly_prices["tick"], lb=0, ub=turb_max)
    m.Storage_level = pf.Variable(hourly_prices["tick"], lb=storage_min, ub=storage_max)
    m.initial_storage_level = (
        m.Storage_level.filter(pl.col("tick") == hourly_prices["tick"].min())
        == storage_level_init_and_final
    )

    m.intermediate_storage_level = (
        m.Storage_level.next(dim="tick", wrap_around=True)
        == m.Storage_level + m.Pump * pump_max * efficiency - m.Turb
    )

    m.pump_and_turbine_xor = m.Turb <= (1 - m.Pump) * turb_max

    m.maximize = ((m.Turb - pump_max * m.Pump) * hourly_prices).sum()

    m.attr.RelativeGap = 1e-5

    m.optimize()

    return m


def read_hourly_prices():
    """Read hourly prices from CSV file and return a DataFrame.

    Special attention is paid to properly parse daylight saving time (DST) changes.
    """
    df = pl.read_csv(
        Path(__file__).parent / "input_data" / "elspot-prices_2021_hourly_eur.csv",
        try_parse_dates=True,
    ).drop_nulls(subset=["DE-LU"])

    df = df.select(
        pl.datetime(
            pl.col("Date").dt.year(),
            pl.col("Date").dt.month(),
            pl.col("Date").dt.day(),
            pl.col("Hours").str.slice(0, 2).cast(pl.Int32),
            time_zone="Europe/Prague",
            ambiguous=pl.when(
                pl.concat_str(pl.col("Date"), pl.col("Hours")).is_first_distinct()
            )
            .then(pl.lit("earliest"))
            .otherwise(pl.lit("latest")),
        ).alias("tick"),
        pl.col("DE-LU").str.replace(",", ".", literal=True).cast(float).alias("price"),
    )

    return df


if __name__ == "__main__":
    print(solve_model().objective.value)
