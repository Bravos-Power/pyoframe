from dataclasses import dataclass
from typing import Dict, List, Optional
import polars as pl
from typing import TYPE_CHECKING

from pyoframe.dataframe import RESERVED_COL_KEYS, get_dimensions

if TYPE_CHECKING:
    from pyoframe.model import Model


@dataclass
class ModelElement:
    name: str = "unnamed"
    _model: Optional["Model"] = None


class FrameWrapper:
    def __init__(self, data: pl.DataFrame):
        # Reorder columns to keep things consistent
        data = data.select(
            get_dimensions(data)
            + [col for col in RESERVED_COL_KEYS if col in data.columns]
        )
        self._data = data

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def dimensions(self) -> List[str]:
        """
        The dimensions of the data.

        Examples
        --------
        >>> from pyoframe.variables import Variable
        >>> Variable().dimensions
        []
        >>> import pandas as pd
        >>> hours = pd.DataFrame({"hour": ["00:00", "06:00", "12:00", "18:00"]})
        >>> cities = pd.DataFrame({"city": ["Toronto", "Berlin", "Paris"]})
        >>> Variable([hours, cities]).dimensions
        ['hour', 'city']
        """
        return get_dimensions(self.data)

    @property
    def shape(self) -> Dict[str, int]:
        """
        Examples
        --------
        >>> import pandas as pd
        >>> from pyoframe.dataframe import VAR_KEY
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
