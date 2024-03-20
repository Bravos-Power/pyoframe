from typing import Iterable, List

import polars as pl

from pyoframe.constraints import (
    AcceptableSets,
    Expression,
    Expressionable,
    _set_to_polars,
)
from pyoframe.dataframe import COEF_KEY, CONST_TERM, VAR_KEY
from pyoframe.model_element import ModelElement
from pyoframe.util import _parse_inputs_as_iterable


class Set(ModelElement, Expressionable):
    def __init__(self, *data: AcceptableSets | Iterable[AcceptableSets], **named_data):
        data_list = list(data)
        for name, set in named_data.items():
            data_list.append({name: set})
        df = self._parse_acceptable_sets(*data_list)
        if df.is_duplicated().any():
            raise ValueError("Duplicate rows found in data.")
        super().__init__(df)

    @staticmethod
    def _parse_acceptable_sets(
        *over: AcceptableSets | Iterable[AcceptableSets],
    ) -> pl.DataFrame:
        """
        >>> import pandas as pd
        >>> dim1 = pd.Index([1, 2, 3], name="dim1")
        >>> dim2 = pd.Index(["a", "b"], name="dim1")
        >>> Set._parse_acceptable_sets([dim1, dim2])
        Traceback (most recent call last):
        ...
        AssertionError: All coordinates must have unique column names.
        >>> dim2.name = "dim2"
        >>> Set._parse_acceptable_sets([dim1, dim2])
        shape: (6, 2)
        ┌──────┬──────┐
        │ dim1 ┆ dim2 │
        │ ---  ┆ ---  │
        │ i64  ┆ str  │
        ╞══════╪══════╡
        │ 1    ┆ a    │
        │ 1    ┆ b    │
        │ 2    ┆ a    │
        │ 2    ┆ b    │
        │ 3    ┆ a    │
        │ 3    ┆ b    │
        └──────┴──────┘
        """
        assert len(over) > 0, "At least one set must be provided."
        over_iter: Iterable[AcceptableSets] = _parse_inputs_as_iterable(over)

        over_frames: List[pl.DataFrame] = [_set_to_polars(set) for set in over_iter]

        over_merged = over_frames[0]

        for df in over_frames[1:]:
            assert (
                set(over_merged.columns) & set(df.columns) == set()
            ), "All coordinates must have unique column names."
            over_merged = over_merged.join(df, how="cross")
        return over_merged

    def to_expr(self) -> Expression:
        return Expression(
            data=self.data.with_columns(
                pl.lit(1).alias(COEF_KEY), pl.lit(CONST_TERM).alias(VAR_KEY)
            )
        )

    def __mul__(self, other):
        if isinstance(other, Set):
            assert (
                set(self.data.columns) & set(other.data.columns) == set()
            ), "Cannot multiply two sets with columns in common."
            return Set(self.data, other.data)
        return super().__mul__(other)

    def __add__(self, other):
        if isinstance(other, Set):
            raise ValueError("Cannot add two sets.")
        return super().__add__(other)
