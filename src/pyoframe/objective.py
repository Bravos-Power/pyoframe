from typing import Literal

from pyoframe.constraints import Expressionable
from pyoframe.model_element import ModelElement


class Objective(ModelElement):
    def __init__(self, expr: Expressionable, sense: Literal["minimize", "maximize"]) -> None:
        super().__init__()
        self.expr = expr.to_expr()
        assert (
            len(self.expr.dimensions) == 0
        ), "Objective can only be a single expression"
        assert sense in ("minimize", "maximize")
        self.sense = sense
