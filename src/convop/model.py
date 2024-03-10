from typing import Any, List
from convop.constraints import Constraint
from convop.objective import Objective
from convop.parameters import Parameter
from convop.variables import Variable
from convop.io import to_file


class Model:
    def __init__(self):
        self.variables: List[Variable] = []
        self.constraints: List[Constraint] = []
        self.objective: Objective | None = None

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, (Variable, Constraint, Parameter, Objective)) and __name != "objective":
            assert not hasattr(self, __name), f"Cannot create {__name} since it was already created."

            __value.name = __name
            if isinstance(__value, Variable):
                self.variables.append(__value)
            elif isinstance(__value, Constraint):
                self.constraints.append(__value)
            elif isinstance(__value, Objective):
                self.objective = __value
        return super().__setattr__(__name, __value)

    to_file = to_file
