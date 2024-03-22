from typing import TYPE_CHECKING
import polars as pl
from pyoframe.constants import CONST_TERM
from pyoframe.constants import VAR_KEY
from pyoframe.util import concat_dimensions

if TYPE_CHECKING:
    from pyoframe.model import Model, Variable


class VariableMapping:
    def map_vars(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError


class NamedVariables(VariableMapping):
    VAR_NAME_KEY = "_var_name"

    def __init__(self, m: "Model") -> None:
        self.map = pl.DataFrame(
            {
                VAR_KEY: [CONST_TERM],
                self.VAR_NAME_KEY: [""],
            },  # Constant value is variable 0
            schema={VAR_KEY: pl.UInt32, self.VAR_NAME_KEY: pl.String},
        )

        for var in m.variables:
            self.add_var(var)

    def add_var(self, var: "Variable") -> None:
        assert var.name is not None, "Variable must have a name to be used in a NamedVariables mapping."
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
        return df.with_columns(
            pl.when(pl.col(VAR_KEY) == pl.lit(CONST_TERM))
            .then(pl.lit(""))
            .otherwise(pl.concat_str(pl.lit("x"), VAR_KEY))
            .alias(VAR_KEY)
        )


DEFAULT_MAP = NumberedVariables()
