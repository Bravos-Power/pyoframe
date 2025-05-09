from pathlib import Path

import polars as pl

import pyoframe as pf


def solve_model(use_var_names=True):
    df = pl.read_csv(
        Path(__file__).parent / "input_data" / "elspot-prices_2021_hourly_eur.csv",
        try_parse_dates=True,
    )
    # use time zone to fix a DST problem at the end of October where one hour is "duplicated"
    range_of_time = df.select(
        pl.col("Date").min().alias("start_timestamp") - pl.duration(hours=3),
        pl.col("Date").max().alias("end_timestamp") + pl.duration(days=1),
    )
    time_tick = (
        pl.datetime_range(
            start=range_of_time["start_timestamp"],
            end=range_of_time["end_timestamp"],
            interval="1h",
            eager=True,
            time_zone="Europe/Prague",
            closed="none",
        )
        .alias("tick")
        .to_frame()
        .filter(
            pl.col("tick").is_between(
                pl.datetime(2021, 1, 1, time_zone="Europe/Prague"),
                pl.datetime(2022, 1, 1, time_zone="Europe/Prague"),
                closed="left",
            )
        )
    )

    # fix a DST problem at the end of March where one hour is "missing" so it is filled with null
    # kids, please do not do use dates without time zone at home
    df = df.drop_nulls()

    hourly_prices = pl.concat((time_tick, df), how="horizontal").select(
        pl.col("tick"),
        pl.col("DE-LU")
        .str.replace(",", ".", literal=True)
        .str.to_decimal()
        .alias("price"),
    )

    pump_max = 70
    turb_max = 90
    effic = 0.75
    storage_capacity = 630
    storage_lower_bound = 100
    storage_level_init = 300
    storage_level_final = 300
    tick_with_initial = pl.concat(
        (time_tick.min().with_columns(pl.col("tick").dt.offset_by("-1h")), time_tick)
    )

    m = pf.Model("unit commitment problem", solver="highs", use_var_names=True)

    m.Pump = pf.Variable(time_tick[["tick"]], vtype=pf.VType.BINARY)
    # ub is redundant since it will be set also in logical condition that pump and turbine cannot work at the same time
    m.Turb = pf.Variable(time_tick[["tick"]], lb=0, ub=turb_max)
    m.Storage_level = pf.Variable(
        tick_with_initial, lb=storage_lower_bound, ub=storage_capacity
    )
    m.initial_storage_level = (
        m.Storage_level.filter(
            pl.col("tick") == tick_with_initial[["tick"]].min().item(0, 0)
        )
        == storage_level_init
    )
    m.final_storage_level = (
        m.Storage_level.filter(pl.col("tick") == time_tick[["tick"]].max().item(0, 0))
        == storage_level_final
    )

    previous_hour_storage_level = m.Storage_level.with_columns(
        pl.col("tick").dt.offset_by("1h")
    )

    m.intermediate_storage_level = (
        m.Storage_level.drop_unmatched()
        == (
            previous_hour_storage_level.drop_unmatched()
            + (m.Pump * pump_max * effic - m.Turb).drop_unmatched()
        ).drop_unmatched()
    )
    m.pump_and_turbine_xor = m.Turb <= (1 - m.Pump) * turb_max

    m.maximize = pf.sum((m.Turb - pump_max * m.Pump) * hourly_prices)

    m.optimize()

    return m


# def write_solution(m: pf.Model, output_dir: Path):
#     m.write(output_dir / 'problem-gurobi-pretty.lp', True)
#     m.write(output_dir / 'problem-gurobi-machine.lp', False)
#
# if __name__ == "__main__":
#     write_solution(solve_model(), Path(__file__).parent / "results")
