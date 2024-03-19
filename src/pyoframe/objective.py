from typing import Literal
from pyoframe.constraints import Expressionable, Expression


class Objective(Expression):
    def __init__(
        self, expr: Expressionable, sense: Literal["minimize", "maximize"]
    ) -> None:
        super().__init__(expr.to_expr().data)
        assert (
            not self.dimensions
        ), "Objective cannot have any dimensions as it must be a single expression"
        assert sense in ("minimize", "maximize")
        self.sense = sense
