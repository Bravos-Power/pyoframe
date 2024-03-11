from typing import Literal

from convop.expressionable import Expressionable
from convop.model_element import ModelElement


class Objective(ModelElement):
    def __init__(self, expr: Expressionable, sense: Literal["min", "max"]) -> None:
        super().__init__()
        self.expr = expr.to_expression()
        assert (
            len(self.expr.dimensions) == 0
        ), "Objective can only be a single expression"
        assert sense in ("min", "max")
        self.sense = sense
