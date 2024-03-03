from convop.constraints import add_constraints
from convop.objective import set_objective
from convop.variables import add_variables


class Model:
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.objective = None

    add_variables = add_variables
    add_constraints = add_constraints
    set_objective = set_objective

    def solve(self):
        raise NotImplementedError("TODO")
