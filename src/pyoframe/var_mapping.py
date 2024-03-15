from typing import TYPE_CHECKING
import polars as pl
from pyoframe.model_element import VAR_KEY

if TYPE_CHECKING:
    from pyoframe.model import Model


class VariableMapping:
    def map_vars(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError


class NamedVariables(VariableMapping):
    VAR_NAME_KEY = "_var_name"

    def __init__(self, m: "Model") -> None:
        var_maps = []
        for var in m.variables:
            df = var.data
            dim = var.dimensions
            if dim:
                df = df.select(
                    pl.concat_str(
                        pl.lit(var.name + "["),
                        pl.concat_str(*var.dimensions, separator=","),
                        pl.lit("]"),
                        separator="",
                    ).alias(self.VAR_NAME_KEY),
                    VAR_KEY,
                )
            else:
                df = df.select(pl.lit(var.name).alias(self.VAR_NAME_KEY), VAR_KEY)
            df = df.with_columns(pl.col(self.VAR_NAME_KEY).str.replace_all(" ", "_"))
            var_maps.append(df)
        self.map = pl.concat(var_maps)

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
