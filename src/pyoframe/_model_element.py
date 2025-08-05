"""Defines the base classes used in Pyoframe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

from pyoframe._arithmetic import _get_dimensions
from pyoframe._constants import (
    COEF_KEY,
    KEY_TYPE,
    QUAD_VAR_KEY,
    RESERVED_COL_KEYS,
    VAR_KEY,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe import Model


class ModelElement(ABC):
    """The base class for elements of a Model such as [][pyoframe.Variable] and [][pyoframe.Constraint]."""

    def __init__(self, data: pl.DataFrame, **kwargs) -> None:
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
            data = data.cast({VAR_KEY: KEY_TYPE})
        if QUAD_VAR_KEY in data.columns:
            data = data.cast({QUAD_VAR_KEY: KEY_TYPE})

        self._data = data
        self._model: Model | None = None
        self.name = None
        super().__init__(**kwargs)

    def _on_add_to_model(self, model: Model, name: str):
        self.name = name
        self._model = model

    @property
    def data(self) -> pl.DataFrame:
        """Returns the object's underlying Polars DataFrame."""
        return self._data

    @property
    def _friendly_name(self) -> str:
        """Returns the name of the element, or `'unnamed'` if it has no name."""
        return self.name if self.name is not None else "unnamed"

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
        """The number of indices in each dimension.

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


# TODO: merge with SupportsMath?
class SupportPolarsMethodMixin(ABC):
    def rename(self, *args, **kwargs):
        """Renames one or several of the object's dimensions.

        Takes the same arguments as [`polars.DataFrame.rename`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rename.html).

        See the [portfolio optimization example](../examples/portfolio_optimization.md) for a usage example.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable(
            ...     {"hour": ["00:00", "06:00", "12:00", "18:00"]},
            ...     {"city": ["Toronto", "Berlin", "Paris"]},
            ... )
            >>> m.v
            <Variable 'v' height=12>
            ┌───────┬─────────┬──────────────────┐
            │ hour  ┆ city    ┆ variable         │
            │ (4)   ┆ (3)     ┆                  │
            ╞═══════╪═════════╪══════════════════╡
            │ 00:00 ┆ Toronto ┆ v[00:00,Toronto] │
            │ 00:00 ┆ Berlin  ┆ v[00:00,Berlin]  │
            │ 00:00 ┆ Paris   ┆ v[00:00,Paris]   │
            │ 06:00 ┆ Toronto ┆ v[06:00,Toronto] │
            │ 06:00 ┆ Berlin  ┆ v[06:00,Berlin]  │
            │ …     ┆ …       ┆ …                │
            │ 12:00 ┆ Berlin  ┆ v[12:00,Berlin]  │
            │ 12:00 ┆ Paris   ┆ v[12:00,Paris]   │
            │ 18:00 ┆ Toronto ┆ v[18:00,Toronto] │
            │ 18:00 ┆ Berlin  ┆ v[18:00,Berlin]  │
            │ 18:00 ┆ Paris   ┆ v[18:00,Paris]   │
            └───────┴─────────┴──────────────────┘

            >>> m.v.rename({"city": "location"})
            <Expression height=12 terms=12 type=linear>
            ┌───────┬──────────┬──────────────────┐
            │ hour  ┆ location ┆ expression       │
            │ (4)   ┆ (3)      ┆                  │
            ╞═══════╪══════════╪══════════════════╡
            │ 00:00 ┆ Toronto  ┆ v[00:00,Toronto] │
            │ 00:00 ┆ Berlin   ┆ v[00:00,Berlin]  │
            │ 00:00 ┆ Paris    ┆ v[00:00,Paris]   │
            │ 06:00 ┆ Toronto  ┆ v[06:00,Toronto] │
            │ 06:00 ┆ Berlin   ┆ v[06:00,Berlin]  │
            │ …     ┆ …        ┆ …                │
            │ 12:00 ┆ Berlin   ┆ v[12:00,Berlin]  │
            │ 12:00 ┆ Paris    ┆ v[12:00,Paris]   │
            │ 18:00 ┆ Toronto  ┆ v[18:00,Toronto] │
            │ 18:00 ┆ Berlin   ┆ v[18:00,Berlin]  │
            │ 18:00 ┆ Paris    ┆ v[18:00,Paris]   │
            └───────┴──────────┴──────────────────┘

        """
        return self._new(self.data.rename(*args, **kwargs))

    def with_columns(self, *args, **kwargs):
        return self._new(self.data.with_columns(*args, **kwargs))

    def filter(self, *args, **kwargs):
        return self._new(self.data.filter(*args, **kwargs))

    @abstractmethod
    def _new(self, data: pl.DataFrame):
        """Creates a new instance of the same class with the given data (for e.g. on .rename(), .with_columns(), etc.)."""

    @property
    @abstractmethod
    def data(self) -> pl.DataFrame: ...

    def pick(self, **kwargs):
        """Filters elements by the given criteria and then drop the filtered dimensions.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable(
            ...     [
            ...         {"hour": ["00:00", "06:00", "12:00", "18:00"]},
            ...         {"city": ["Toronto", "Berlin", "Paris"]},
            ...     ]
            ... )
            >>> m.v.pick(hour="06:00")
            <Expression height=3 terms=3 type=linear>
            ┌─────────┬──────────────────┐
            │ city    ┆ expression       │
            │ (3)     ┆                  │
            ╞═════════╪══════════════════╡
            │ Toronto ┆ v[06:00,Toronto] │
            │ Berlin  ┆ v[06:00,Berlin]  │
            │ Paris   ┆ v[06:00,Paris]   │
            └─────────┴──────────────────┘
            >>> m.v.pick(hour="06:00", city="Toronto")
            <Expression terms=1 type=linear>
            v[06:00,Toronto]
        """
        return self._new(self.data.filter(**kwargs).drop(kwargs.keys()))


class ModelElementWithId(ModelElement):
    """Extends ModelElement with a method that assigns a unique ID to each row in a DataFrame.

    IDs start at 1 and go up consecutively. No zero ID is assigned since it is reserved for the constant variable term.
    IDs are only unique for the subclass since different subclasses have different counters.
    """

    @property
    def _has_ids(self) -> bool:
        return self._get_id_column_name() in self.data.columns

    def _assert_has_ids(self):
        if not self._has_ids:
            raise ValueError(
                f"Cannot use '{self.__class__.__name__}' before it has been added to a model."
            )

    @classmethod
    @abstractmethod
    def _get_id_column_name(cls) -> str:
        """Returns the name of the column containing the IDs."""
