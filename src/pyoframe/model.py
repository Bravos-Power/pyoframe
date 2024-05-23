from typing import Any, Iterable, List, Optional, Union
from pyoframe.constants import (
    ObjSense,
    VType,
    Config,
    Result,
    PyoframeError,
    ObjSenseValue,
)
from pyoframe.io_mappers import NamedVariableMapper, IOMappers
from pyoframe.model_element import ModelElement, ModelElementWithId
from pyoframe.core import Constraint
from pyoframe.objective import Objective
from pyoframe.user_defined import Container, AttrContainerMixin
from pyoframe.core import Variable
from pyoframe.io import to_file
from pyoframe.solvers import solve, Solver
import polars as pl
import pandas as pd


class Model(AttrContainerMixin):
    """
    Represents a mathematical optimization model. Add variables, constraints, and an objective to the model by setting attributes.
    """

    _reserved_attributes = [
        "_variables",
        "_constraints",
        "_objective",
        "var_map",
        "io_mappers",
        "name",
        "solver",
        "solver_model",
        "params",
        "result",
        "attr",
        "sense",
        "objective",
    ]

    def __init__(self, min_or_max: Union[ObjSense, ObjSenseValue], name=None, **kwargs):
        super().__init__(**kwargs)
        self._variables: List[Variable] = []
        self._constraints: List[Constraint] = []
        self.sense = ObjSense(min_or_max)
        self._objective: Optional[Objective] = None
        self.var_map = (
            NamedVariableMapper(Variable) if Config.print_uses_variable_names else None
        )
        self.io_mappers: Optional[IOMappers] = None
        self.name = name
        self.solver: Optional[Solver] = None
        self.solver_model: Optional[Any] = None
        self.params = Container()
        self.result: Optional[Result] = None

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

    @objective.setter
    def objective(self, value):
        value = Objective(value)
        self._objective = value
        value.on_add_to_model(self, "objective")

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name not in Model._reserved_attributes and not isinstance(
            __value, (ModelElement, pl.DataFrame, pd.DataFrame)
        ):
            raise PyoframeError(
                f"Cannot set attribute '{__name}' on the model because it isn't of type ModelElement (e.g. Variable, Constraint, ...)"
            )

        if (
            isinstance(__value, ModelElement)
            and __name not in Model._reserved_attributes
        ):
            if isinstance(__value, ModelElementWithId):
                assert not hasattr(
                    self, __name
                ), f"Cannot create {__name} since it was already created."

            __value.on_add_to_model(self, __name)

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
