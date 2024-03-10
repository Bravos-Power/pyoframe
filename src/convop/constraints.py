from enum import Enum
from convop.expressionable import ConstraintExpression


class Constraint:
    def __init__(
        self,
        constraint: ConstraintExpression,
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
        self.lhs = constraint.lhs.to_expression()
        self.sense = constraint.sense

        self.name = name

    def __repr__(self):

        return f"""
        Constraint: {self.name} | ({self.sense} 0)
        {self.lhs}
        """
