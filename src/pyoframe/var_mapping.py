import math
import string

from typing import TYPE_CHECKING
import polars as pl
from pyoframe.constants import CONST_TERM
from pyoframe.constants import VAR_KEY
from pyoframe.util import concat_dimensions


if TYPE_CHECKING:  # pragma: no cover
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
        assert (
            var.name is not None
        ), "Variable must have a name to be used in a NamedVariables mapping."
        self.map = pl.concat(
            [
                self.map,
                concat_dimensions(var.id, keep_dims=False, prefix=var.name).rename(
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


class Base62EncodedVariables(VariableMapping):
    # Mapping between a base 62 character and its integer value
    _CHAR_TABLE = pl.DataFrame(
        {"char": list(string.digits + string.ascii_letters)},
    ).with_columns(pl.int_range(pl.len()).cast(pl.UInt32).alias("code"))

    _BASE = _CHAR_TABLE.height  # _BASE = 62
    _ZERO = _CHAR_TABLE.filter(pl.col("code") == 0).select("char").item()  # _ZERO = "0"

    def map_vars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Examples:
            >>> import polars as pl
            >>> from pyoframe import Model, Variable
            >>> m = Model()
            >>> m.x = Variable(pl.DataFrame({"t": range(1,63)}))
            >>> (m.x.filter(t=11)+1).to_str()
            '[11]: 1  + x[11]'
            >>> (m.x.filter(t=11)+1).to_str(var_map=Base62EncodedVariables())
            '[11]: 1  + xb'

            >>> Base62EncodedVariables().map_vars(pl.DataFrame({VAR_KEY: []}))
            shape: (0, 1)
            ┌───────────────┐
            │ __variable_id │
            │ ---           │
            │ null          │
            ╞═══════════════╡
            └───────────────┘
        """
        if df.height == 0:
            return df

        return df.with_columns(
            pl.when(pl.col(VAR_KEY) == pl.lit(CONST_TERM))
            .then(pl.lit(""))
            .otherwise(
                pl.concat_str(
                    pl.lit("x"),
                    pl.col(VAR_KEY).map_batches(
                        Base62EncodedVariables._to_base62,
                        return_dtype=pl.String,
                        is_elementwise=True,
                    ),
                )
            )
            .alias(VAR_KEY)
        )

    @classmethod
    def _to_base62(cls, int_col: pl.Series) -> pl.Series:
        """Returns a series of dtype str with a base 62 representation of the integers in int_col.
        The letters 0-9a-zA-Z are used as symbols for the representation.

        Examples:

            >>> import polars as pl
            >>> s = pl.Series([0,10,20,60,53,66], dtype=pl.UInt32)
            >>> Base62EncodedVariables._to_base62(s).to_list()
            ['0', 'a', 'k', 'Y', 'R', '14']

            >>> s = pl.Series([0], dtype=pl.UInt32)
            >>> Base62EncodedVariables._to_base62(s).to_list()
            ['0']
        """
        assert isinstance(
            int_col.dtype, pl.UInt32
        ), "_to_base62() only works for UInt32 id columns"

        largest_id = int_col.max()
        if largest_id == 0:
            max_digits = 1
        else:
            max_digits = math.floor(math.log(largest_id, cls._BASE)) + 1  # type: ignore

        digits = []

        for i in range(max_digits):
            remainder = int_col % cls._BASE

            digits.append(
                remainder.to_frame(name="code")
                .join(cls._CHAR_TABLE, on="code", how="left")
                .select("char")
                .rename({"char": f"digit{i}"})
            )
            int_col //= cls._BASE

        return (
            pl.concat(reversed(digits), how="horizontal")
            .select(pl.concat_str(pl.all()))
            .to_series()
            .str.strip_chars_start(cls._ZERO)
            .replace("", cls._ZERO)
        )
