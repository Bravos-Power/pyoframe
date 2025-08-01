"""Defines the base classes used in Pyoframe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl

from pyoframe._arithmetic import _get_dimensions
from pyoframe._constants import (
    COEF_KEY,
    KEY_TYPE,
    QUAD_VAR_KEY,
    RESERVED_COL_KEYS,
    VAR_KEY,
    Config,
)
from pyoframe._utils import concat_dimensions

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
    def dimensions_unsafe(self) -> list[str]:
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

    def __len__(self) -> int:
        dims = self.dimensions
        if dims is None:
            return 1
        return self.data.select(dims).n_unique()

    def _append_ellipsis(self, input: str) -> str:
        result = input
        if Config.print_max_lines and Config.print_max_lines < len(self):
            result += "\n" + " " * (len(self.name) if self.name else 0) + " ⋮"
        return result

    def _to_str_create_prefix(self, data):
        if self.name is None and self.dimensions is None:
            return data

        return (
            concat_dimensions(data, prefix=self.name, ignore_columns=["expr"])
            .with_columns(
                pl.concat_str("concated_dim", pl.lit(": "), "expr").alias("expr")
            )
            .drop("concated_dim")
        )


def _support_polars_method(method_name: str):
    """Wraps the underlying Polars method to make it accessible on ModelElement."""

    def method(self: SupportPolarsMethodMixin, *args, **kwargs) -> Any:
        result_from_polars = getattr(self.data, method_name)(*args, **kwargs)
        if isinstance(result_from_polars, pl.DataFrame):
            return self._new(result_from_polars)
        else:
            return result_from_polars

    return method


class SupportPolarsMethodMixin(ABC):
    rename = _support_polars_method("rename")
    with_columns = _support_polars_method("with_columns")
    filter = _support_polars_method("filter")
    estimated_size = _support_polars_method("estimated_size")

    @abstractmethod
    def _new(self, data: pl.DataFrame):
        """Creates a new instance of the same class with the given data (for e.g. on .rename(), .with_columns(), etc.)."""

    @property
    @abstractmethod
    def data(self): ...

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
            <Expression size=3 dimensions={'city': 3} terms=3>
            [Toronto]: v[06:00,Toronto]
            [Berlin]: v[06:00,Berlin]
            [Paris]: v[06:00,Paris]
            >>> m.v.pick(hour="06:00", city="Toronto")
            <Expression size=1 dimensions={} terms=1>
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
                f"Cannot use '{self.__class__.__name__}' before it has beed added to a model."
            )

    @classmethod
    @abstractmethod
    def _get_id_column_name(cls) -> str:
        """Returns the name of the column containing the IDs."""
