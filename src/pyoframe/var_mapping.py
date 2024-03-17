from typing import TYPE_CHECKING
import polars as pl
from pyoframe.dataframe import VAR_KEY, concat_dimensions

if TYPE_CHECKING:
    from pyoframe.model import Model, Variable


class VariableMapping:
    def map_vars(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError


class NamedVariables(VariableMapping):
    VAR_NAME_KEY = "_var_name"

    def __init__(self, m: "Model") -> None:
        self.map = pl.DataFrame(
            {VAR_KEY: [], self.VAR_NAME_KEY: []},
            schema={VAR_KEY: pl.UInt32, self.VAR_NAME_KEY: pl.Utf8},
        )

        for var in m.variables:
            self.add_var(var)

    def add_var(self, var: "Variable") -> None:
        self.map = pl.concat(
            [
                self.map,
                concat_dimensions(var.data, keep_dims=False, prefix=var.name).rename(
                    {"concated_dim": self.VAR_NAME_KEY}
                ),
            ]
        )

    def map_vars(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.join(self.map, on=VAR_KEY, how="left", validate="m:1")
            .drop(VAR_KEY)
            .rename({self.VAR_NAME_KEY: VAR_KEY})
        )


class NumberedVariables(VariableMapping):
    def map_vars(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.concat_str(pl.lit("x"), VAR_KEY).alias(VAR_KEY))


DEFAULT_MAP = NumberedVariables()
