"""
File containing shared constants used across the package.

Code is heavily based on the `linopy` package by Fabian Hofmann.

MIT License
"""

import importlib.metadata
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

import polars as pl
import pyoptinterface as poi
from packaging import version

# We want to try and support multiple major versions of polars
POLARS_VERSION = version.parse(importlib.metadata.version("polars"))

COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"
CONSTRAINT_KEY = "__constraint_id"
SOLUTION_KEY = "solution"
DUAL_KEY = "dual"
RC_COL = "RC"
SLACK_COL = "slack"

COL_DTYPES = {
    COEF_KEY: pl.Float64,
    VAR_KEY: pl.UInt32,
    CONSTRAINT_KEY: pl.UInt32,
    SOLUTION_KEY: pl.Float64,
    DUAL_KEY: pl.Float64,
    RC_COL: pl.Float64,
    SLACK_COL: pl.Float64,
}
VAR_TYPE = COL_DTYPES[VAR_KEY]

CONST_TERM = 0  # 0 is a reserved value which makes it easy to detect. It must be zero for e.g. ensuring consistency during quadratic multiplication.

RESERVED_COL_KEYS = (
    COEF_KEY,
    VAR_KEY,
    CONSTRAINT_KEY,
    SOLUTION_KEY,
    DUAL_KEY,
    RC_COL,
    SLACK_COL,
)


class _ConfigMeta(type):
    """Metaclass for Config that stores the default values of all configuration options."""

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls._defaults = {
            k: v
            for k, v in dct.items()
            if not k.startswith("_") and type(v) != classmethod
        }


class Config(metaclass=_ConfigMeta):
    disable_unmatched_checks: bool = False
    str_float_precision: Optional[int] = 5
    print_uses_variable_names: bool = True
    # Number of elements to show when printing a set to the console (additional elements are replaced with ...)
    print_max_set_elements: int = 50
    enable_is_duplicated_expression_safety_check: bool = False

    @classmethod
    def reset_defaults(cls):
        """
        Resets all configuration options to their default values.
        """
        for key, value in cls._defaults.items():
            setattr(cls, key, value)


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "="

    def to_poi(self):
        if self == ConstraintSense.LE:
            return poi.ConstraintSense.LessEqual
        elif self == ConstraintSense.EQ:
            return poi.ConstraintSense.Equal
        elif self == ConstraintSense.GE:
            return poi.ConstraintSense.GreaterEqual
        else:
            raise ValueError(f"Invalid constraint type: {self}")


class ObjSense(Enum):
    MIN = "min"
    MAX = "max"

    def to_poi(self):
        if self == ObjSense.MIN:
            return poi.ObjectiveSense.Minimize
        elif self == ObjSense.MAX:
            return poi.ObjectiveSense.Maximize
        else:
            raise ValueError(f"Invalid objective sense: {self}")


class VType(Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"

    def to_poi(self):
        if self == VType.CONTINUOUS:
            return poi.VariableDomain.Continuous
        elif self == VType.BINARY:
            return poi.VariableDomain.Binary
        elif self == VType.INTEGER:
            return poi.VariableDomain.Integer
        else:
            raise ValueError(f"Invalid variable type: {self}")


class UnmatchedStrategy(Enum):
    UNSET = "not_set"
    DROP = "drop"
    KEEP = "keep"


# This is a hack to get the Literal type for VType
# See: https://stackoverflow.com/questions/67292470/type-hinting-enum-member-value-in-python
ObjSenseValue = Literal["min", "max"]
VTypeValue = Literal["continuous", "binary", "integer"]
for enum, type in [(ObjSense, ObjSenseValue), (VType, VTypeValue)]:
    assert set(typing.get_args(type)) == {vtype.value for vtype in enum}


class ModelStatus(Enum):
    """
    Model status.

    The set of possible model status is a superset of the solver status
    set.
    """

    ok = "ok"
    warning = "warning"
    error = "error"
    aborted = "aborted"
    unknown = "unknown"
    initialized = "initialized"


class SolverStatus(Enum):
    """
    Solver status.
    """

    ok = "ok"
    warning = "warning"
    error = "error"
    aborted = "aborted"
    unknown = "unknown"

    @classmethod
    def process(cls, status: str) -> "SolverStatus":
        try:
            return cls(status)
        except ValueError:
            return cls("unknown")

    @classmethod
    def from_termination_condition(
        cls, termination_condition: "TerminationCondition"
    ) -> "SolverStatus":
        for (
            status,
            termination_conditions,
        ) in STATUS_TO_TERMINATION_CONDITION_MAP.items():
            if termination_condition in termination_conditions:
                return status
        return cls("unknown")


class TerminationCondition(Enum):
    """
    Termination condition of the solver.
    """

    # UNKNOWN
    unknown = "unknown"

    # OK
    optimal = "optimal"
    time_limit = "time_limit"
    iteration_limit = "iteration_limit"
    terminated_by_limit = "terminated_by_limit"
    suboptimal = "suboptimal"

    # WARNING
    unbounded = "unbounded"
    infeasible = "infeasible"
    infeasible_or_unbounded = "infeasible_or_unbounded"
    other = "other"

    # ERROR
    internal_solver_error = "internal_solver_error"
    error = "error"

    # ABORTED
    user_interrupt = "user_interrupt"
    resource_interrupt = "resource_interrupt"
    licensing_problems = "licensing_problems"

    @classmethod
    def process(
        cls, termination_condition: Union[str, "TerminationCondition"]
    ) -> "TerminationCondition":
        try:
            return cls(termination_condition)
        except ValueError:
            return cls("unknown")


STATUS_TO_TERMINATION_CONDITION_MAP = {
    SolverStatus.ok: [
        TerminationCondition.optimal,
        TerminationCondition.iteration_limit,
        TerminationCondition.time_limit,
        TerminationCondition.terminated_by_limit,
        TerminationCondition.suboptimal,
    ],
    SolverStatus.warning: [
        TerminationCondition.unbounded,
        TerminationCondition.infeasible,
        TerminationCondition.infeasible_or_unbounded,
        TerminationCondition.other,
    ],
    SolverStatus.error: [
        TerminationCondition.internal_solver_error,
        TerminationCondition.error,
    ],
    SolverStatus.aborted: [
        TerminationCondition.user_interrupt,
        TerminationCondition.resource_interrupt,
        TerminationCondition.licensing_problems,
    ],
    SolverStatus.unknown: [TerminationCondition.unknown],
}


@dataclass
class Status:
    """
    Status and termination condition of the solver.
    """

    status: SolverStatus
    termination_condition: TerminationCondition

    @classmethod
    def process(cls, status: str, termination_condition: str) -> "Status":
        return cls(
            status=SolverStatus.process(status),
            termination_condition=TerminationCondition.process(termination_condition),
        )

    @classmethod
    def from_termination_condition(
        cls, termination_condition: Union["TerminationCondition", str]
    ) -> "Status":
        termination_condition = TerminationCondition.process(termination_condition)
        solver_status = SolverStatus.from_termination_condition(termination_condition)
        return cls(solver_status, termination_condition)

    @property
    def is_ok(self) -> bool:
        return self.status == SolverStatus.ok


@dataclass
class Solution:
    """
    Solution returned by the solver.
    """

    primal: pl.DataFrame
    dual: Optional[pl.DataFrame]
    objective: float


@dataclass
class Result:
    """
    Result of the optimization.
    """

    status: Status
    solution: Optional[Solution] = None

    def __repr__(self) -> str:
        res = (
            f"Status: {self.status.status.value}\n"
            f"Termination condition: {self.status.termination_condition.value}\n"
        )
        if self.solution is not None:
            res += (
                f"Solution: {len(self.solution.primal)} primals, {len(self.solution.dual) if self.solution.dual is not None else 0} duals\n"
                f"Objective: {self.solution.objective:.2e}\n"
            )

        return res

    def info(self):
        status = self.status

        if status.is_ok:
            if status.termination_condition == TerminationCondition.suboptimal:
                print(f"Optimization solution is sub-optimal: \n{self}\n")
            else:
                print(f" Optimization successful: \n{self}\n")
        else:
            print(f"Optimization failed: \n{self}\n")


class PyoframeError(Exception):
    pass
