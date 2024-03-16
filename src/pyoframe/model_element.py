from dataclasses import dataclass
from typing import Dict, List, Optional
import polars as pl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyoframe.model import Model

COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"
RESERVED_COL_KEYS = [COEF_KEY, VAR_KEY]


@dataclass
class ModelElement:
    name: str = "unnamed"
    _model: Optional["Model"] = None


class FrameWrapper:
    def __init__(self, data: pl.DataFrame):
        # Reorder columns to keep things consistent
        data = data.select(
            [col for col in data.columns if col not in RESERVED_COL_KEYS]
            + [col for col in RESERVED_COL_KEYS if col in data.columns]
        )
        self._data = data

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def dimensions(self) -> List[str]:
        return [col for col in self.data.columns if col not in RESERVED_COL_KEYS]

    @property
    def shape(self) -> Dict[str, int]:
        """
        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({VAR_KEY: [1]})
        >>> FrameWrapper(pl.from_pandas(df)).shape
        {}
        """
        dims = self.dimensions
        return {dim: self.data[dim].n_unique() for dim in dims}

    def __len__(self) -> int:
        if not self.dimensions:
            return 1
        return self.data.drop(*RESERVED_COL_KEYS).n_unique()
