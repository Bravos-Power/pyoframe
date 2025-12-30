"""Defines the base classes used in Pyoframe."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import polars as pl

from pyoframe._arithmetic import _get_dimensions
from pyoframe._constants import (
    COEF_KEY,
    QUAD_VAR_KEY,
    RESERVED_COL_KEYS,
    VAR_KEY,
    Config,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe import Model


class BaseBlock(ABC):
    """The base class for elements of a Model such as [][pyoframe.Variable] and [][pyoframe.Constraint]."""

    def __init__(self, data: pl.DataFrame, name="unnamed") -> None:
        # Sanity checks, no duplicate column names
        assert len(data.columns) == len(set(data.columns)), (
            "Duplicate column names found."
        )

        cols = _get_dimensions(data)
        if cols is None:
            cols = []
        cols += [col for col in RESERVED_COL_KEYS if col in data.columns]

        # Reorder columns to keep things consistent
        data = data.select(cols)

        # Cast to proper dtype
        if COEF_KEY in data.columns:
            data = data.cast({COEF_KEY: pl.Float64})
        if VAR_KEY in data.columns:
            data = data.cast({VAR_KEY: Config.id_dtype})
        if QUAD_VAR_KEY in data.columns:
            data = data.cast({QUAD_VAR_KEY: Config.id_dtype})

        self._data = data
        self._model: Model | None = None
        self.name: str = name  # gets overwritten if object is added to model
        """A user-friendly name that is displayed when printing the object or in error messages.
        When an object is added to a model, this name is updated to the name used in the model."""

    def _on_add_to_model(self, model: Model, name: str):
        self.name = name
        self._model = model

    @property
    def data(self) -> pl.DataFrame:
        """Returns the object's underlying Polars DataFrame."""
        return self._data

    @property
    def dimensions(self) -> list[str] | None:
        """The names of the data's dimensions.

        Examples:
            A variable with no dimensions
            >>> pf.Variable().dimensions

            A variable with dimensions of "hour" and "city"
            >>> pf.Variable(
            ...     [
            ...         {"hour": ["00:00", "06:00", "12:00", "18:00"]},
            ...         {"city": ["Toronto", "Berlin", "Paris"]},
            ...     ]
            ... ).dimensions
            ['hour', 'city']
        """
        return _get_dimensions(self.data)

    @property
    def dimensionless(self) -> bool:
        """Whether the object has no dimensions.

        Examples:
            A variable with no dimensions
            >>> pf.Variable().dimensionless
            True

            A variable with dimensions of "hour" and "city"
            >>> pf.Variable(
            ...     [
            ...         {"hour": ["00:00", "06:00", "12:00", "18:00"]},
            ...         {"city": ["Toronto", "Berlin", "Paris"]},
            ...     ]
            ... ).dimensionless
            False
        """
        return self.dimensions is None

    @property
    def _dimensions_unsafe(self) -> list[str]:
        """Same as `dimensions` but returns an empty list if there are no dimensions instead of `None`.

        When unsure, use `dimensions` instead since the type checker forces users to handle the None case (no dimensions).
        """
        dims = self.dimensions
        if dims is None:
            return []
        return dims

    @property
    def shape(self) -> dict[str, int]:
        """The number of distinct labels in each dimension.

        Examples:
            A variable with no dimensions
            >>> pf.Variable().shape
            {}

            A variable with dimensions of "hour" and "city"
            >>> pf.Variable(
            ...     [
            ...         {"hour": ["00:00", "06:00", "12:00", "18:00"]},
            ...         {"city": ["Toronto", "Berlin", "Paris"]},
            ...     ]
            ... ).shape
            {'hour': 4, 'city': 3}
        """
        dims = self.dimensions
        if dims is None:
            return {}
        return {dim: self.data[dim].n_unique() for dim in dims}

    def estimated_size(self, unit: pl.SizeUnit = "b") -> int | float:
        """Returns the estimated size of the object in bytes.

        Only considers the size of the underlying DataFrame(s) since other components (e.g., the object name) are negligible.

        Parameters:
            unit:
                See [`polars.DataFrame.estimated_size`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.estimated_size.html).

        Examples:
            >>> m = pf.Model()

            A dimensionless variable contains just a 32 bit (4 bytes) unsigned integer (the variable ID).

            >>> m.x = pf.Variable()
            >>> m.x.estimated_size()
            4

            A dimensioned variable contains, for every row, a 32 bit ID and, in this case, a 64 bit `dim_x` value (1200 bytes total).

            >>> m.y = pf.Variable(pf.Set(dim_x=range(100)))
            >>> m.y.estimated_size()
            1200
        """
        return self.data.estimated_size(unit)

    def _add_shape_to_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Adds the shape of the data to the columns of the DataFrame.

        This is used for displaying the shape in the string representation of the object.
        """
        shape = self.shape
        return df.rename(lambda col: f"{col}\n({shape[col]})" if col in shape else col)

    def __len__(self) -> int:
        dims = self.dimensions
        if dims is None:
            return 1
        return self.data.select(dims).n_unique()

    @property
    def _has_ids(self) -> bool:
        id_col = self._get_id_column_name()
        assert id_col is not None, "Cannot check for IDs if no ID column is defined."
        return id_col in self.data.columns

    def _assert_has_ids(self):
        if not self._has_ids:
            raise ValueError(
                f"Cannot use '{self.__class__.__name__}' before it has been added to a model."
            )

    @classmethod
    def _get_id_column_name(cls) -> str | None:
        """Subclasses should override to indicate that `data` contains an ID column."""
        return None
