from convop.constraints import add_constraints
from convop.objective import Objective, set_objective
from convop.variables import add_variables


class Model:
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.objective: Objective | None = None

    add_variables = add_variables
    add_constraints = add_constraints
    set_objective = set_objective
    to_file = to_file
