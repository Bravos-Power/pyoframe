from enum import Enum
from typing import Literal

from convop.expressionable import Expressionable

class Sense(Enum):
    MIN = "min"
    MAX = "max"


class Objective:
    def __init__(self, expr: Expressionable, sense: Literal["min", "max"]) -> None:
        self.expr = expr.to_expression()
        assert len(self.expr.dimensions) == 0, "Objective can only be a single expression"
        assert sense in ("min", "max")
        self.sense = sense


def set_objective(model, expr: Expressionable, direction: Literal["min", "max"]) -> Objective:
    model.objective = Objective(expr, direction)
    return model.objective
