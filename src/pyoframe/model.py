from typing import Any, Iterable, List, Optional
from pyoframe.constants import ObjSense, VType, Config
from pyoframe.constraints import SupportsMath
from pyoframe.io_mappers import NamedMapper, IOMappers
from pyoframe.model_element import ModelElement
from pyoframe.constraints import Constraint
from pyoframe.objective import Objective
from pyoframe.variables import Variable
from pyoframe.io import to_file
from pyoframe.solvers import solve


class Model:
    """
    Represents a mathematical optimization model. Add variables, constraints, and an objective to the model by setting attributes.
    """

    def __init__(self, name=None):
        self._variables: List[Variable] = []
        self._constraints: List[Constraint] = []
        self._objective: Optional[Objective] = None
        self.var_map = (
            None if Config.shorten_names_everywhere else NamedMapper(Variable)
        )
        self.io_mappers: Optional[IOMappers] = None
        self.name = name

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @property
    def binary_variables(self) -> Iterable[Variable]:
        return (v for v in self.variables if v.vtype == VType.BINARY)

    @property
    def integer_variables(self) -> Iterable[Variable]:
        return (v for v in self.variables if v.vtype == VType.INTEGER)

    @property
    def constraints(self):
        return self._constraints

    @property
    def objective(self):
        return self._objective

    @property
    def maximize(self):
        assert self.objective is not None and self.objective.sense == ObjSense.MAX
        return self.objective

    @property
    def minimize(self):
        assert self.objective is not None and self.objective.sense == ObjSense.MIN
        return self.objective

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ("maximize", "minimize"):
            assert isinstance(
                __value, SupportsMath
            ), f"Setting {__name} on the model requires an objective expression."
            self._objective = Objective(__value, sense=__name)
            self._objective.name = __name
            self._objective._model = self
            return

        if isinstance(__value, ModelElement) and not __name.startswith("_"):
            assert not hasattr(
                self, __name
            ), f"Cannot create {__name} since it was already created."

            __value.name = __name
            __value._model = self

            if isinstance(__value, Objective):
                assert self.objective is None, "Cannot create more than one objective."
                self._objective = __value
            if isinstance(__value, Variable):
                self._variables.append(__value)
                if self.var_map is not None:
                    self.var_map.add(__value)
            elif isinstance(__value, Constraint):
                self._constraints.append(__value)

        return super().__setattr__(__name, __value)

    def __repr__(self) -> str:
        return f"""Model '{self.name}' ({len(self.variables)} vars, {len(self.constraints)} constrs, {1 if self.objective else "no"} obj)"""

    to_file = to_file
    solve = solve
