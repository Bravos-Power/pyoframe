from enum import Enum
from typing import Literal

from convop.expressions import Expression

SenseType = Literal["min", "max", "minimize", "maximize"]


class Sense(Enum):
    MIN = "min"
    MAX = "max"


class Objective:
    def __init__(self, expr: Expression, sense: SenseType) -> None:
        self.expr = expr
        self.sense = Objective._parse_sense(sense)

    @staticmethod
    def _parse_sense(sense: SenseType):
        if sense in ("min", "minimize"):
            return Sense.MIN
        elif sense in ("max", "maximize"):
            return Sense.MAX
        else:
            raise ValueError(f"Unknown direction: {sense}")


def set_objective(model, expr: Expression, direction: SenseType) -> Objective:
    model.objective = Objective(expr, direction)
    return model.objective
