from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import polars as pl
from typing import TYPE_CHECKING

from pyoframe.constants import COEF_KEY, RESERVED_COL_KEYS, VAR_KEY
from pyoframe.util import get_dimensions

if TYPE_CHECKING:
    from pyoframe.model import Model


def _pass_polars_method(method_name: str):
    def method(self, *args, **kwargs):
        return self._new(getattr(self.data, method_name)(*args, **kwargs))

    return method


class ModelElement(ABC):
    def __init__(
        self,
        data: pl.DataFrame,
        model: Optional["Model"] = None,
        name: Optional[str] = None,
    ) -> None:
        # Sanity checks, no duplicate column names
        assert len(data.columns) == len(
            set(data.columns)
        ), "Duplicate column names found."

        dimensions = get_dimensions(data)
        reserved_cols = [col for col in RESERVED_COL_KEYS if col in data.columns]

        # Reorder columns to keep things consistent
        data = data.select(dimensions + reserved_cols)

        # Cast to proper dtype
        if COEF_KEY in reserved_cols:
            data = data.cast({COEF_KEY: pl.Float64})
        if VAR_KEY in reserved_cols:
            data = data.cast({VAR_KEY: pl.UInt32})

        self._data = data
        self.name = name
        self._model = model

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def dimensions(self) -> List[str]:
        """
        The names of the data's dimensions.

        Examples
        --------
        >>> from pyoframe.variables import Variable
        >>> # A variable with no dimensions
        >>> Variable().dimensions
        []
        >>> # A variable with dimensions of "hour" and "city"
        >>> Variable([{"hour": ["00:00", "06:00", "12:00", "18:00"]}, {"city": ["Toronto", "Berlin", "Paris"]}]).dimensions
        ['hour', 'city']
        """
        return get_dimensions(self.data)

    @property
    def shape(self) -> Dict[str, int]:
        """
        The number of indices in each dimension.

        Examples
        --------
        >>> from pyoframe.variables import Variable
        >>> # A variable with no dimensions
        >>> Variable().shape
        {}
        >>> # A variable with dimensions of "hour" and "city"
        >>> Variable([{"hour": ["00:00", "06:00", "12:00", "18:00"]}, {"city": ["Toronto", "Berlin", "Paris"]}]).shape
        {'hour': 4, 'city': 3}
        """
        return {dim: self.data[dim].n_unique() for dim in self.dimensions}

    def __len__(self) -> int:
        if not self.dimensions:
            return 1
        return self.data.drop(*RESERVED_COL_KEYS).n_unique()

    @abstractmethod
    def _new(self, data: pl.DataFrame):
        raise NotImplementedError("Subclasses must implement this method")

    rename = _pass_polars_method("rename")
    with_columns = _pass_polars_method("with_columns")
    filter = _pass_polars_method("filter")
