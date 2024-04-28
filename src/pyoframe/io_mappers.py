"""
Defines various methods for mapping a variable or constraint to its string representation.
"""

from dataclasses import dataclass
import math
import string
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Optional, Type, Union
import polars as pl
from pyoframe.constants import NAME_COL
from pyoframe.util import concat_dimensions


if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Variable
    from pyoframe.constraints import Constraint
    from pyoframe.util import IdCounterMixin


@dataclass
class IOMappers:
    var_map: "Mapper"
    const_map: "Mapper"


class Mapper(ABC):
    def __init__(self, cls: Type["IdCounterMixin"]) -> None:
        self._ID_COL = cls.get_id_column_name()
        self.mapping_registry = pl.DataFrame(
            {self._ID_COL: [], NAME_COL: []},
            schema={self._ID_COL: pl.UInt32, NAME_COL: pl.String},
        )

    def add(self, element: Union["Variable", "Constraint"]) -> None:
        self.mapping_registry = pl.concat(
            [self.mapping_registry, self._element_to_map(element)]
        )

    @abstractmethod
    def _element_to_map(self, element: "IdCounterMixin") -> pl.DataFrame: ...

    def apply(
        self,
        df: pl.DataFrame,
        to_col: Optional[str],
    ) -> pl.DataFrame:
        result = df.join(
            self.mapping_registry, on=self._ID_COL, how="left", validate="m:1"
        )
        if to_col is None:
            result = result.drop(self._ID_COL)
            to_col = self._ID_COL
        return result.rename({NAME_COL: to_col})

    def undo(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.join(
            self.mapping_registry, on=NAME_COL, how="left", validate="m:1"
        ).drop(NAME_COL)


class NamedMapper(Mapper):
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

    def _element_to_map(self, element) -> pl.DataFrame:
        element_name = element.name  # type: ignore
        assert (
            element_name is not None
        ), "Element must have a name to be used in a named mapping."
        return concat_dimensions(
            element.ids, keep_dims=False, prefix=element_name, to_col=NAME_COL
        )


class Base62Mapper(Mapper, ABC):
    # Mapping between a base 62 character and its integer value
    _CHAR_TABLE = pl.DataFrame(
        {"char": list(string.digits + string.ascii_letters)},
    ).with_columns(pl.int_range(pl.len()).cast(pl.UInt32).alias("code"))

    _BASE = _CHAR_TABLE.height  # _BASE = 62
    _ZERO = _CHAR_TABLE.filter(pl.col("code") == 0).select("char").item()  # _ZERO = "0"

    @property
    @abstractmethod
    def _prefix(self) -> "str": ...

    def apply(
        self,
        df: pl.DataFrame,
        to_col: Optional[str] = None,
    ) -> pl.DataFrame:
        if df.height == 0:
            return df

        query = pl.concat_str(
            pl.lit(self._prefix),
            pl.col(self._ID_COL).map_batches(
                Base62Mapper._to_base62,
                return_dtype=pl.String,
                is_elementwise=True,
            ),
        )

        if to_col is None:
            to_col = self._ID_COL

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

    def _element_to_map(self, element) -> pl.DataFrame:
        return self.apply(element.ids.select(self._ID_COL), to_col=NAME_COL)


class Base62VarMapper(Base62Mapper):
    """
    Examples:
        >>> import polars as pl
        >>> from pyoframe import Model, Variable
        >>> from pyoframe.constants import VAR_KEY
        >>> m = Model()
        >>> m.x = Variable(pl.DataFrame({"t": range(1,63)}))
        >>> (m.x.filter(t=11)+1).to_str()
        '[11]: 1  + x[11]'
        >>> (m.x.filter(t=11)+1).to_str(var_map=Base62VarMapper(Variable))
        '[11]: 1  + xb'

        >>> Base62VarMapper(Variable).apply(pl.DataFrame({VAR_KEY: []}))
        shape: (0, 1)
        ┌───────────────┐
        │ __variable_id │
        │ ---           │
        │ null          │
        ╞═══════════════╡
        └───────────────┘
    """

    @property
    def _prefix(self) -> "str":
        return "x"


class Base62ConstMapper(Base62Mapper):

    @property
    def _prefix(self) -> "str":
        return "c"
