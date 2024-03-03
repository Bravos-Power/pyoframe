from enum import Enum
from typing import Literal
from convop.expressions import Expression

DirectionType = Literal["min", "max", "minimize", "maximize"]


class Direction(Enum):
    MIN = "min"
    MAX = "max"


class Objective:
    def __init__(self, expr: Expression, direction: DirectionType) -> None:
        self.expr = expr
        self.direction = Objective._parse_direction(direction)

    @staticmethod
    def _parse_direction(direction: DirectionType):
        if direction in ("min", "minimize"):
            return Direction.MIN
        elif direction in ("max", "maximize"):
            return Direction.MAX
        else:
            raise ValueError(f"Unknown direction: {direction}")


def set_objective(model, expr: Expression, direction: DirectionType):
    model.objective = Objective(expr, direction)
