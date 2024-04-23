"""
Defines various methods for mapping a variable or constraint to its string representation.
"""

from dataclasses import dataclass
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


@dataclass
class IOMappers:
    var_map: "Mapper"
    const_map: "Mapper"


class VarMixin:
    @property
    def _ID_COL(self) -> str:
        return VAR_KEY


class ConstMixin:
    @property
    def _ID_COL(self) -> str:
        return CONSTRAINT_KEY


class Mapper(ABC):
    _NAME_COL = "_name"

    def __init__(self, prefix=None) -> None:
        self.prefix = prefix
        self.mapping_registry = pl.DataFrame(
            {self._ID_COL: [], self._NAME_COL: []},
            schema={self._ID_COL: pl.UInt32, self._NAME_COL: pl.String},
        )

    @property
    def _ID_COL(self) -> str:
        raise NotImplementedError

    def add(self, element: Union["Variable", "Constraint"]) -> None:
        self.mapping_registry = pl.concat(
            [self.mapping_registry, self._element_to_map(element)]
        )

    @abstractmethod
    def _element_to_map(
        self, element: Union["Variable", "Constraint"]
    ) -> pl.DataFrame: ...

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
        return result.rename({self._NAME_COL: to_col})


class NamedVarMapper(VarMixin, Mapper):
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

    def __init__(self) -> None:
        super().__init__()
        self.mapping_registry = pl.DataFrame(
            {
                self._ID_COL: [CONST_TERM],
                self._NAME_COL: [""],
            },
            schema={self._ID_COL: pl.UInt32, self._NAME_COL: pl.String},
        )

    def _element_to_map(self, var: "Variable") -> pl.DataFrame:
        assert (
            var.name is not None
        ), "Variable must have a name to be used in a NamedVariables mapping."
        return concat_dimensions(
            var.ids, keep_dims=False, prefix=var.name, to_col=self._NAME_COL
        )


class NamedConstMapper(ConstMixin, Mapper):
    def _element_to_map(self, element: "Constraint") -> pl.DataFrame:
        assert (
            element.name is not None
        ), "Constraint must have a name to be used in a NamedConstraintMapper."
        return concat_dimensions(
            element.ids,
            keep_dims=False,
            prefix=element.name,
            to_col=self._NAME_COL,
        )


class Base62Mapper(Mapper):
    # Mapping between a base 62 character and its integer value
    _CHAR_TABLE = pl.DataFrame(
        {"char": list(string.digits + string.ascii_letters)},
    ).with_columns(pl.int_range(pl.len()).cast(pl.UInt32).alias("code"))

    _BASE = _CHAR_TABLE.height  # _BASE = 62
    _ZERO = _CHAR_TABLE.filter(pl.col("code") == 0).select("char").item()  # _ZERO = "0"

    @abstractmethod
    def _get_prefix(self) -> "str": ...

    def apply(
        self,
        df: pl.DataFrame,
        to_col: Optional[str] = None,
    ) -> pl.DataFrame:
        if df.height == 0:
            return df

        query = pl.concat_str(
            pl.lit(self._get_prefix()),
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


class Base62VarMapper(VarMixin, Base62Mapper):
    """
    Examples:
        >>> import polars as pl
        >>> from pyoframe import Model, Variable
        >>> m = Model()
        >>> m.x = Variable(pl.DataFrame({"t": range(1,63)}))
        >>> (m.x.filter(t=11)+1).to_str()
        '[11]: 1  + x[11]'
        >>> (m.x.filter(t=11)+1).to_str(var_map=Base62VarMapper())
        '[11]: 1  + xb'

        >>> Base62VarMapper().apply(pl.DataFrame({VAR_KEY: []}))
        shape: (0, 1)
        ┌───────────────┐
        │ __variable_id │
        │ ---           │
        │ null          │
        ╞═══════════════╡
        └───────────────┘
    """

    def _get_prefix(self) -> "str":
        return "x"

    def _element_to_map(self, element) -> pl.DataFrame:
        return self.apply(element.data.select(VAR_KEY), to_col=self._NAME_COL)


class Base62ConstMapper(ConstMixin, Base62Mapper):
    def _get_prefix(self) -> "str":
        return "c"

    def _element_to_map(self, element) -> pl.DataFrame:
        return self.apply(
            element.data_per_constraint.select(CONSTRAINT_KEY), to_col=self._NAME_COL
        )
