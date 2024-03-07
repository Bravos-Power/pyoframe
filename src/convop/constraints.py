from enum import Enum
from re import L
from typing import Literal, Self

from convop.expressions import Expression, Expressionable


class Sense(Enum):
    LE = "<="
    GE = ">="
    EQ = "="

    @staticmethod
    def from_str(s: str) -> "Sense":
        if s == "<=":
            return Sense.LE
        elif s == ">=":
            return Sense.GE
        elif s == "=" or s == "==":
            return Sense.EQ
        else:
            raise ValueError(f"Unknown sense: {s}")


class Constraints:
    def __init__(
        self, lhs: Expression, sense: Sense, rhs: Expression, name: str | None
    ):
        self.lhs = lhs - rhs
        self.sense = sense
        self.name = name
    
    def __repr__(self):

        return f"""
        Constraint: {self.name} | ({self.sense} 0)
        {self.lhs}
        """


def add_constraints(
    model,
    lhs: Expressionable,
    sense: Sense | Literal["<=", ">=", "=", "=="],
    rhs: Expressionable,
    name: str | None = None,
):
    """Adds a constraint to the model.

    Parameters
    ----------
    lhs: Expression
        The left hand side of the constraint.
    sense: Sense
        The sense of the constraint.
    rhs: Expression
        The right hand side of the constraint.
    name: str, optional
        The name of the constraint. If using ModelBuilder this is automatically set to match your constraint name.
    """
    if isinstance(sense, str):
        sense = Sense.from_str(sense)
    constraints = Constraints(lhs=lhs.to_expression(), sense=sense, rhs=rhs.to_expression(), name=name)
    model.constraints.append(constraints)
    return constraints