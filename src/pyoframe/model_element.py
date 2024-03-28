from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import polars as pl
from typing import TYPE_CHECKING

from pyoframe.constants import COEF_KEY, RESERVED_COL_KEYS, VAR_KEY
from pyoframe.arithmetic import get_dimensions

if TYPE_CHECKING:
    from pyoframe.model import Model


def _pass_polars_method(method_name: str):
    def method(self, *args, **kwargs):
        return self._new(getattr(self.data, method_name)(*args, **kwargs))

    return method


class ModelElement(ABC):
    def __init__(self, data: pl.DataFrame, **kwargs) -> None:
        # Sanity checks, no duplicate column names
        assert len(data.columns) == len(
            set(data.columns)
        ), "Duplicate column names found."

        cols = get_dimensions(data)
        if cols is None:
            cols = []
        cols += [col for col in RESERVED_COL_KEYS if col in data.columns]

        # Reorder columns to keep things consistent
        data = data.select(cols)

        # Cast to proper dtype
        if COEF_KEY in data.columns:
            data = data.cast({COEF_KEY: pl.Float64})
        if VAR_KEY in data.columns:
            data = data.cast({VAR_KEY: pl.UInt32})

        self._data = data
        self._model = None
        self.name = None
        super().__init__(**kwargs)

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def dimensions(self) -> Optional[List[str]]:
        """
        The names of the data's dimensions.

        Examples:
            >>> from pyoframe.variables import Variable
            >>> # A variable with no dimensions
            >>> Variable().dimensions

            >>> # A variable with dimensions of "hour" and "city"
            >>> Variable([{"hour": ["00:00", "06:00", "12:00", "18:00"]}, {"city": ["Toronto", "Berlin", "Paris"]}]).dimensions
            ['hour', 'city']
        """
        return get_dimensions(self.data)
    
    @property
    def dimensions_unsafe(self) -> List[str]:
        """
        Same as `dimensions` but returns an empty list if there are no dimensions instead of None.
        When unsure, use `dimensions` instead since the type checker forces users to handle the None case (no dimensions).
        """
        dims = self.dimensions
        if dims is None:
            return []
        return dims

    @property
    def shape(self) -> Dict[str, int]:
        """
        The number of indices in each dimension.

        Examples:
            >>> from pyoframe.variables import Variable
            >>> # A variable with no dimensions
            >>> Variable().shape
            {}
            >>> # A variable with dimensions of "hour" and "city"
            >>> Variable([{"hour": ["00:00", "06:00", "12:00", "18:00"]}, {"city": ["Toronto", "Berlin", "Paris"]}]).shape
            {'hour': 4, 'city': 3}
        """
        dims = self.dimensions
        if dims is None:
            return {}
        return {dim: self.data[dim].n_unique() for dim in dims}

    def __len__(self) -> int:
        dims = self.dimensions
        if dims is None:
            return 1
        return self.data.select(dims).n_unique()

    @abstractmethod
    def _new(self, data: pl.DataFrame):
        raise NotImplementedError

    rename = _pass_polars_method("rename")
    with_columns = _pass_polars_method("with_columns")
    filter = _pass_polars_method("filter")
