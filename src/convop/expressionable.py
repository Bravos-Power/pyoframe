from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Expressionable:
    """Any object that can be converted into an expression."""

    def to_expression(self):
        """Converts the object into an Expression."""
        raise NotImplementedError(
            "to_expression must be implemented in subclass " + self.__class__.__name__
        )

    def __add__(self, other):
        return self.to_expression() + other

    def __neg__(self):
        return self.to_expression() * -1

    def __sub__(self, other):
        return self.to_expression() + (other * -1)

    def __mul__(self, other):
        return self.to_expression() * other

    def __rmul__(self, other):
        return self.to_expression() * other

    def __radd__(self, other):
        return self.to_expression() + other

    def __le__(self, other):
        return ConstraintExpression(self - other, ConstraintSense.LE)

    def __ge__(self, other):
        return ConstraintExpression(self - other, ConstraintSense.GE)

    def __eq__(self, __value: object):
        return ConstraintExpression(self - __value, ConstraintSense.EQ)


def sum(over: str | Sequence[str], expr: Expressionable):
    return expr.to_expression().sum(over)


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "="


@dataclass
class ConstraintExpression:
    lhs: Expressionable
    sense: ConstraintSense
