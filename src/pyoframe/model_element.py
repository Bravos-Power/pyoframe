from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional
import polars as pl
from typing import TYPE_CHECKING

from pyoframe.constants import COEF_KEY, RESERVED_COL_KEYS, VAR_KEY
from pyoframe._arithmetic import _get_dimensions
from pyoframe.user_defined import AttrContainerMixin

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Model


class ModelElement(ABC):
    def __init__(self, data: pl.DataFrame, **kwargs) -> None:
        # Sanity checks, no duplicate column names
        assert len(data.columns) == len(
            set(data.columns)
        ), "Duplicate column names found."

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
            data = data.cast({VAR_KEY: pl.UInt32})

        self._data = data
        self._model: Optional[Model] = None
        self.name = None
        super().__init__(**kwargs)

    def on_add_to_model(self, model: "Model", name: str):
        self.name = name
        self._model = model

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def friendly_name(self) -> str:
        return self.name if self.name is not None else "unnamed"

    @property
    def dimensions(self) -> Optional[List[str]]:
        """
        The names of the data's dimensions.

        Examples:
            >>> from pyoframe.core import Variable
            >>> # A variable with no dimensions
            >>> Variable().dimensions

            >>> # A variable with dimensions of "hour" and "city"
            >>> Variable([{"hour": ["00:00", "06:00", "12:00", "18:00"]}, {"city": ["Toronto", "Berlin", "Paris"]}]).dimensions
            ['hour', 'city']
        """
        return _get_dimensions(self.data)

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
            >>> from pyoframe.core import Variable
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


def _support_polars_method(method_name: str):
    """
    Wrapper to add a method to ModelElement that simply calls the underlying Polars method on the data attribute.
    """

    def method(self: "SupportPolarsMethodMixin", *args, **kwargs) -> Any:
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
        """
        Used to create a new instance of the same class with the given data (for e.g. on .rename(), .with_columns(), etc.).
        """

    @property
    @abstractmethod
    def data(self): ...


class ModelElementWithId(ModelElement, AttrContainerMixin):
    """
    Provides a method that assigns a unique ID to each row in a DataFrame.
    IDs start at 1 and go up consecutively. No zero ID is assigned since it is reserved for the constant variable term.
    IDs are only unique for the subclass since different subclasses have different counters.
    """

    # Keys are the subclass names and values are the next unasigned ID.
    _id_counters: Dict[str, int] = defaultdict(lambda: 1)

    @classmethod
    def reset_counters(cls):
        """
        Resets all the ID counters.
        This function is called before every unit test to reset the code state.
        """
        cls._id_counters = defaultdict(lambda: 1)

    def __init__(self, data: pl.DataFrame, **kwargs) -> None:
        super().__init__(data, **kwargs)
        self._data = self._assign_ids(self.data)

    @classmethod
    def _assign_ids(cls, df: pl.DataFrame) -> pl.DataFrame:
        """
        Adds the column `to_column` to the DataFrame `df` with the next batch
        of unique consecutive IDs.
        """
        cls_name = cls.__name__
        cur_count = cls._id_counters[cls_name]
        id_col_name = cls.get_id_column_name()

        if df.height == 0:
            df = df.with_columns(pl.lit(cur_count).alias(id_col_name))
        else:
            df = df.with_columns(
                pl.int_range(cur_count, cur_count + pl.len()).alias(id_col_name)
            )
        df = df.with_columns(pl.col(id_col_name).cast(pl.UInt32))
        cls._id_counters[cls_name] += df.height
        return df

    @classmethod
    @abstractmethod
    def get_id_column_name(cls) -> str:
        """
        Returns the name of the column containing the IDs.
        """

    @property
    def ids(self) -> pl.DataFrame:
        return self.data.select(self.dimensions_unsafe + [self.get_id_column_name()])

    def _extend_dataframe_by_id(self, addition: pl.DataFrame):
        cols = addition.columns
        assert len(cols) == 2
        id_col = self.get_id_column_name()
        assert id_col in cols
        cols.remove(id_col)
        new_col = cols[0]

        original = self.data

        if new_col in original.columns:
            original = original.drop(new_col)
        self._data = original.join(addition, on=id_col, how="left", validate="1:1")

    def _preprocess_attr(self, name: str, value: Any) -> Any:
        dims = self.dimensions
        ids = self.ids
        id_col = self.get_id_column_name()

        if isinstance(value, pl.DataFrame):
            if value.shape == (1, 1):
                value = value.item()
            else:
                assert (
                    dims is not None
                ), "Attribute must be a scalar since there are no dimensions"
                result = value.join(ids, on=dims, validate="1:1", how="inner").drop(
                    dims
                )
                assert len(result.columns) == 2, "Attribute has too many columns"
                value_col = [c for c in result.columns if c != id_col][0]
                return result.rename({value_col: name})

        assert ids.height == 1, "Attribute is a scalar but there are multiple IDs."
        return pl.DataFrame({name: [value], id_col: ids.get_column(id_col)})
