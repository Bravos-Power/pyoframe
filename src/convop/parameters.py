from pathlib import Path
import polars as pl
from convop.expressionable import Expressionable

from convop.expressions import CONST_KEY, Expression
from convop.model_element import ModelElement


class Parameters:
    def __init__(
        self,
        df: pl.DataFrame | Path | str,
        dim: int,
        defaults: dict[str, float] | None = None,
    ):
        if not isinstance(df, pl.DataFrame):
            df = _read_file(df)
        if defaults is not None:
            df = df.with_columns(
                [pl.col(key).fill_null(value) for key, value in defaults.items()]
            )
        self._data = df
        self.dim = dim

    def __getitem__(self, key):
        return Parameter(self._data, self.dim, key, name=key)


class Parameter(Expressionable, ModelElement):
    def __init__(
        self,
        df: pl.DataFrame | Path | str,
        dim: int | None = None,
        param_key: str | None = None,
        default: float | None = None,
        name: str = "unknown",
    ):
        super().__init__(name)
        if not isinstance(df, pl.DataFrame):
            df = _read_file(df)
        if param_key is None:
            param_key = df.columns[-1]
        if dim is None:
            dim = len(df.columns) - 1
        self._data = df
        self._dim_keys = df.columns[:dim]
        self._param_key = param_key
        self._default = default

    @property
    def data(self):
        df = self._data.select(
            [pl.col(index) for index in self._dim_keys]
            + [pl.col(self._param_key).alias(CONST_KEY)]
        )
        if self._default is not None:
            df = df.with_columns(pl.col(CONST_KEY).fill_null(self._default))
        return df

    def to_expression(self):
        return Expression(constants=self.data)

    def __repr__(self):
        return f"""
        Parameters: {self.name}
        {self.data}
        """


def _read_file(path: str | Path) -> pl.DataFrame:
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == ".csv":
        return pl.read_csv(path)
    elif path.suffix == ".parquet":
        return pl.read_parquet(path)
    else:
        raise ValueError(
            f"File type ({path.suffix}) not supported. Please use .csv or .parquet."
        )
