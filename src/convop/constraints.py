from convop.expressionable import ConstraintExpression
from convop.model_element import ModelElement


class Constraint(ModelElement):
    def __init__(
        self,
        constraint: ConstraintExpression,
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
        """
        super().__init__()
        self.lhs = constraint.lhs.to_expression()
        self.sense = constraint.sense

    def __repr__(self):

        return f"""
        Constraint: {self.name} | ({self.sense} 0)
        {self.lhs}
        """
