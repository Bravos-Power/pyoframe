from dataclasses import dataclass
from typing import Dict, List
import polars as pl

COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"
RESERVED_COL_KEYS = [COEF_KEY, VAR_KEY]


@dataclass
class ModelElement:
    name: str = "unnamed"


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
        dims = self.dimensions
        return {dim: self.data[dim].n_unique() for dim in dims}

    def __len__(self) -> int:
        return self.data.drop(*RESERVED_COL_KEYS).n_unique()
