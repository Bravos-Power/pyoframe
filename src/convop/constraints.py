from enum import Enum
from convop.expressions import Expression


class Sense(Enum):
    LE = "<="
    GE = ">="
    EQ = "="


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
    lhs: Expression,
    sense: Sense,
    rhs: Expression,
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
    constraints = Constraints(lhs=lhs, sense=sense, rhs=rhs, name=name)
    model.constraints.append(constraints)
    return constraints
