"""
Defines various methods for mapping a variable or constraint to its string representation.
"""

import math
import string
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Optional, Union
import polars as pl
from pyoframe.constants import CONST_TERM, CONSTRAINT_KEY
from pyoframe.constants import VAR_KEY
from pyoframe.util import concat_dimensions


if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Variable
    from pyoframe.constraints import Constraint


class Mapper(ABC):
    @abstractmethod
    def _map(
        self,
        df: pl.DataFrame,
        from_col: str,
        to_col: Optional[str],
        prefix: str,
        empty_string: Optional[int],
    ) -> pl.DataFrame: ...

    def map_vars(
        self, df: Union[pl.DataFrame, "Variable"], to_col="_var_name"
    ) -> pl.DataFrame:
        return self._map(
            df, VAR_KEY, to_col=to_col, prefix="x", empty_string=CONST_TERM
        )

    def map_consts(
        self, df: Union[pl.DataFrame, "Constraint"], to_col="_const_name"
    ) -> pl.DataFrame:
        return self._map(
            df, CONSTRAINT_KEY, to_col=to_col, prefix="c", empty_string=None
        )


class NamedConstraintMapper(Mapper):
    def _map(
        self,
        df: pl.DataFrame,
        from_col: str,
        to_col: str,
        prefix: str,
        empty_string: Optional[int],
    ) -> pl.DataFrame:
        assert empty_string is None
        if to_col is None:
            to_col = from_col
            keep_dims = False
        else:
            keep_dims = True
        return concat_dimensions(df, prefix=prefix, to_col=to_col, keep_dims=keep_dims)


class PersistentNamedVarMapper(Mapper):
    """
    Maps constraints or variables to a string representation using the object's name and dimensions.

    Examples:

        >>> import polars as pl
        >>> import pyoframe as pf
        >>> m = pf.Model()
        >>> m.foo = pf.Variable(pl.DataFrame({"t": range(4)}))
        >>> pf.sum(m.foo)
        <Expression size=1 dimensions={} terms=4>
        foo[0] + foo[1] + foo[2] + foo[3]
    """

    _VAR_NAME_KEY = "_var_name"

    def __init__(self) -> None:
        self.mapping_registry = pl.DataFrame(
            {
                VAR_KEY: [CONST_TERM],
                self._VAR_NAME_KEY: [""],
            },  # Constant value is variable 0
            schema={VAR_KEY: pl.UInt32, self._VAR_NAME_KEY: pl.String},
        )

    def add_var(self, var: "Variable") -> None:
        assert (
            var.name is not None
        ), "Variable must have a name to be used in a NamedVariables mapping."
        self.mapping_registry = pl.concat(
            [
                self.mapping_registry,
                concat_dimensions(var.id, keep_dims=False, prefix=var.name).rename(
                    {"concated_dim": self._VAR_NAME_KEY}
                ),
            ]
        )

    def _map(
        self,
        df: pl.DataFrame,
        from_col: str,
        to_col: str,
        prefix: str,
        empty_string: Optional[int],
    ) -> pl.DataFrame:
        result = df.join(self.mapping_registry, on=from_col, how="left", validate="m:1")
        if to_col is None:
            result = result.drop(from_col)
            to_col = from_col
        return result.rename({self._VAR_NAME_KEY: to_col})


class NumberedMapper(Mapper):
    def _map(
        self,
        df: pl.DataFrame,
        from_col: str,
        to_col: str,
        prefix: str,
        empty_string: Optional[int],
    ) -> pl.DataFrame:
        query = pl.concat_str(pl.lit(prefix), from_col)
        if empty_string is not None:
            query = query.replace(pl.lit(prefix + str(empty_string)), pl.lit(""))
        if to_col is None:
            to_col = from_col
        return df.with_columns(query.alias(to_col))


class Base62Mapper(Mapper):
    # Mapping between a base 62 character and its integer value
    _CHAR_TABLE = pl.DataFrame(
        {"char": list(string.digits + string.ascii_letters)},
    ).with_columns(pl.int_range(pl.len()).cast(pl.UInt32).alias("code"))

    _BASE = _CHAR_TABLE.height  # _BASE = 62
    _ZERO = _CHAR_TABLE.filter(pl.col("code") == 0).select("char").item()  # _ZERO = "0"

    def _map(
        self,
        df: pl.DataFrame,
        from_col: str,
        to_col: str,
        prefix: str,
        empty_string: Optional[int],
    ) -> pl.DataFrame:
        """
        Examples:
            >>> import polars as pl
            >>> from pyoframe import Model, Variable
            >>> m = Model()
            >>> m.x = Variable(pl.DataFrame({"t": range(1,63)}))
            >>> (m.x.filter(t=11)+1).to_str()
            '[11]: 1  + x[11]'
            >>> (m.x.filter(t=11)+1).to_str(var_map=Base62Mapper())
            '[11]: 1  + xb'

            >>> Base62Mapper().map_vars(pl.DataFrame({VAR_KEY: []}))
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

        query = pl.concat_str(
            pl.lit(prefix),
            pl.col(from_col).map_batches(
                Base62Mapper._to_base62,
                return_dtype=pl.String,
                is_elementwise=True,
            ),
        )
        if empty_string is not None:
            query = (
                pl.when(pl.col(from_col) == pl.lit(empty_string))
                .then(pl.lit(""))
                .otherwise(query)
            )

        if to_col is None:
            to_col = from_col
        return df.with_columns(query.alias(to_col))

    @classmethod
    def _to_base62(cls, int_col: pl.Series) -> pl.Series:
        """Returns a series of dtype str with a base 62 representation of the integers in int_col.
        The letters 0-9a-zA-Z are used as symbols for the representation.

        Examples:

            >>> import polars as pl
            >>> s = pl.Series([0,10,20,60,53,66], dtype=pl.UInt32)
            >>> Base62Mapper._to_base62(s).to_list()
            ['0', 'a', 'k', 'Y', 'R', '14']

            >>> s = pl.Series([0], dtype=pl.UInt32)
            >>> Base62Mapper._to_base62(s).to_list()
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
