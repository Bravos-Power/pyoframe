from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
import polars as pl
import pyoptinterface as poi

from pyoframe.constants import (
    CONST_TERM,
    SUPPORTED_SOLVER_TYPES,
    SUPPORTED_SOLVERS,
    Config,
    ObjSense,
    ObjSenseValue,
    PyoframeError,
    Solver,
    VType,
)
from pyoframe.core import Constraint, Variable
from pyoframe.model_element import ModelElement, ModelElementWithId
from pyoframe.objective import Objective
from pyoframe.util import Container, NamedVariableMapper, for_solvers, get_obj_repr


class Model:
    """
    The object that holds all the variables, constraints, and the objective.

    Parameters:
        name:
            The name of the model. Currently it is not used for much.
        solver:
            The solver to use. If `None`, `Config.default_solver` will be used.
            If `Config.default_solver` has not been set (`None`), Pyoframe will try to detect whichever solver is already installed.
        solver_env:
            Gurobi only: a dictionary of parameters to set when creating the Gurobi environment.
        use_var_names:
            Whether to pass variable names to the solver. Set to `True` if you'd like outputs from e.g. `Model.write()` to be legible.
            Does not work with HiGHS (see [here](https://github.com/Bravos-Power/pyoframe/issues/102#issuecomment-2727521430)).
        sense:
            Either "min" or "max". Indicates whether it's a minmization or maximization problem.
            Typically, this parameter can be omitted (`None`) as it will automatically be
            set when the objective is set using `.minimize` or `.maximize`.

    Examples:
        >>> m = pf.Model()
        >>> m.X = pf.Variable()
        >>> m.my_constraint = m.X <= 10
        >>> m
        <Model vars=1 constrs=1 objective=False>

        Try setting the Gurobi license:
        >>> m = pf.Model(
        ...     solver="gurobi",
        ...     solver_env=dict(ComputeServer="myserver", ServerPassword="mypassword"),
        ... )
        Traceback (most recent call last):
        ...
        RuntimeError: Could not resolve host: myserver (code 6, command POST http://myserver/api/v1/cluster/jobs)
    """

    _reserved_attributes = [
        "_variables",
        "_constraints",
        "_objective",
        "var_map",
        "io_mappers",
        "name",
        "solver",
        "poi",
        "_params",
        "params",
        "result",
        "_attr",
        "attr",
        "sense",
        "objective",
        "_use_var_names",
        "ONE",
        "solver_name",
        "minimize",
        "maximize",
    ]

    def __init__(
        self,
        name: Optional[str] = None,
        solver: SUPPORTED_SOLVER_TYPES | Solver | None = None,
        solver_env: Optional[Dict[str, str]] = None,
        use_var_names: bool = False,
        sense: Union[ObjSense, ObjSenseValue, None] = None,
    ):
        self.poi, self.solver = Model.create_poi_model(solver, solver_env)
        self.solver_name = self.solver.name
        self._variables: List[Variable] = []
        self._constraints: List[Constraint] = []
        self.sense = ObjSense(sense) if sense is not None else None
        self._objective: Optional[Objective] = None
        self.var_map = (
            NamedVariableMapper(Variable) if Config.print_uses_variable_names else None
        )
        self.name = name

        self._params = Container(self._set_param, self._get_param)
        self._attr = Container(self._set_attr, self._get_attr)
        self._use_var_names = use_var_names

    @property
    def use_var_names(self):
        return self._use_var_names

    @property
    def attr(self):
        """
        An object that allows reading and writing model attributes.

        Several model attributes are common across all solvers making it easy to switch between solvers (see supported attributes for
        [Gurobi](https://metab0t.github.io/PyOptInterface/gurobi.html#supported-model-attribute),
        [HiGHS](https://metab0t.github.io/PyOptInterface/highs.html), and
        [Ipopt](https://metab0t.github.io/PyOptInterface/ipopt.html)).

        We additionally support all of [Gurobi's attributes](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes.html#sec:Attributes) when using Gurobi.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable(lb=1, ub=1, vtype="integer")
            >>> m.attr.Silent = True  # Prevent solver output from being printed
            >>> m.optimize()
            >>> m.attr.TerminationStatus
            <TerminationStatusCode.OPTIMAL: 2>

            Some attributes, like `NumVars`, are solver-specific.
            >>> m = pf.Model(solver="gurobi")
            >>> m.attr.NumConstrs
            0
            >>> m = pf.Model(solver="highs")
            >>> m.attr.NumConstrs
            Traceback (most recent call last):
            ...
            KeyError: 'NumConstrs'

        See also:
            [Variable.attr][pyoframe.Variable.attr] for setting variable attributes and
            [Constraint.attr][pyoframe.Constraint.attr] for setting constraint attributes.
        """
        return self._attr

    @property
    def params(self) -> Container:
        """
        An object that allows reading and writing solver-specific parameters.

        See the list of available parameters for
        [Gurobi](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#sec:Parameters),
        [HiGHS](https://ergo-code.github.io/HiGHS/stable/options/definitions/),
        and [Ipopt](https://coin-or.github.io/Ipopt/OPTIONS.html).

        Examples:
            For example, if you'd like to use Gurobi's barrier method, you can set the `Method` parameter:
            >>> m = pf.Model(solver="gurobi")
            >>> m.params.Method = 2
        """
        return self._params

    @classmethod
    def create_poi_model(
        cls, solver: Optional[str | Solver], solver_env: Optional[Dict[str, str]]
    ):
        if solver is None:
            if Config.default_solver is None:
                for solver_option in SUPPORTED_SOLVERS:
                    try:
                        return cls.create_poi_model(solver_option, solver_env)
                    except RuntimeError:
                        pass
                raise ValueError(
                    'Could not automatically find a solver. Is one installed? If so, specify which one: e.g. Model(solver="gurobi")'
                )
            else:
                solver = Config.default_solver

        if isinstance(solver, str):
            solver = solver.lower()
            for s in SUPPORTED_SOLVERS:
                if s.name == solver:
                    solver = s
                    break
            else:
                raise ValueError(
                    f"Unsupported solver: '{solver}'. Supported solvers are: {', '.join(s.name for s in SUPPORTED_SOLVERS)}."
                )

        if solver.name == "gurobi":
            from pyoptinterface import gurobi

            if solver_env is None:
                env = gurobi.Env()
            else:
                env = gurobi.Env(empty=True)
                for key, value in solver_env.items():
                    env.set_raw_parameter(key, value)
                env.start()
            model = gurobi.Model(env)
        elif solver.name == "highs":
            from pyoptinterface import highs

            model = highs.Model()
        elif solver.name == "ipopt":
            try:
                from pyoptinterface import ipopt
            except ModuleNotFoundError as e:  # pragma: no cover
                raise ModuleNotFoundError(
                    "Failed to import the Ipopt solver. Did you run `pip install pyoptinterface[ipopt]`?"
                ) from e

            try:
                model = ipopt.Model()
            except RuntimeError as e:  # pragma: no cover
                if "IPOPT library is not loaded" in str(e):
                    raise RuntimeError(
                        "Could not find the Ipopt solver. Are you sure you've properly installed it and added it to your PATH?"
                    ) from e
                raise e
        else:
            raise ValueError(
                f"Solver {solver} not recognized or supported."
            )  # pragma: no cover

        constant_var = model.add_variable(lb=1, ub=1, name="ONE")
        if constant_var.index != CONST_TERM:
            raise ValueError(
                "The first variable should have index 0."
            )  # pragma: no cover
        return model, solver

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @property
    def binary_variables(self) -> Iterable[Variable]:
        """
        Examples:
            >>> m = pf.Model()
            >>> m.X = pf.Variable(vtype=pf.VType.BINARY)
            >>> m.Y = pf.Variable()
            >>> len(list(m.binary_variables))
            1
        """
        return (v for v in self.variables if v.vtype == VType.BINARY)

    @property
    def integer_variables(self) -> Iterable[Variable]:
        """
        Examples:
            >>> m = pf.Model()
            >>> m.X = pf.Variable(vtype=pf.VType.INTEGER)
            >>> m.Y = pf.Variable()
            >>> len(list(m.integer_variables))
            1
        """
        return (v for v in self.variables if v.vtype == VType.INTEGER)

    @property
    def constraints(self):
        return self._constraints

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        if self._objective is not None and (
            not isinstance(value, Objective) or not value._constructive
        ):
            raise ValueError("An objective already exists. Use += or -= to modify it.")
        if not isinstance(value, Objective):
            value = Objective(value)
        self._objective = value
        value.on_add_to_model(self, "objective")

    @property
    def minimize(self):
        if self.sense != ObjSense.MIN:
            raise ValueError("Can't get .minimize in a maximization problem.")
        return self._objective

    @minimize.setter
    def minimize(self, value):
        if self.sense is None:
            self.sense = ObjSense.MIN
        if self.sense != ObjSense.MIN:
            raise ValueError("Can't set .minimize in a maximization problem.")
        self.objective = value

    @property
    def maximize(self):
        if self.sense != ObjSense.MAX:
            raise ValueError("Can't get .maximize in a minimization problem.")
        return self._objective

    @maximize.setter
    def maximize(self, value):
        if self.sense is None:
            self.sense = ObjSense.MAX
        if self.sense != ObjSense.MAX:
            raise ValueError("Can't set .maximize in a minimization problem.")
        self.objective = value

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
                assert not hasattr(self, __name), (
                    f"Cannot create {__name} since it was already created."
                )

            __value.on_add_to_model(self, __name)

            if isinstance(__value, Variable):
                self._variables.append(__value)
                if self.var_map is not None:
                    self.var_map.add(__value)
            elif isinstance(__value, Constraint):
                self._constraints.append(__value)
        return super().__setattr__(__name, __value)

    def __repr__(self) -> str:
        return get_obj_repr(
            self,
            name=self.name,
            vars=len(self.variables),
            constrs=len(self.constraints),
            objective=bool(self.objective),
        )

    def write(self, file_path: Union[Path, str], pretty: bool = False):
        """
        Output the model to a file.

        Typical usage includes writing the solution to a `.sol` file as well as writing the problem to a `.lp` or `.mps` file.
        Set `use_var_names` in your model constructor to `True` if you'd like the output to contain human-readable names (useful for debugging).

        Parameters:
            file_path:
                The path to the file to write to.
            pretty:
                Only used when writing .sol files in HiGHS. If `True`, will use HiGH's pretty print columnar style which contains more information.
        """
        self.solver.check_supports_write()

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        kwargs = {}
        if self.solver.name == "highs":
            if self.use_var_names:
                self.params.write_solution_style = 1
            kwargs["pretty"] = pretty
        self.poi.write(str(file_path), **kwargs)

    def optimize(self):
        """
        Optimize the model using your selected solver (e.g. Gurobi, HiGHS).
        """
        self.poi.optimize()

    @for_solvers("gurobi")
    def convert_to_fixed(self) -> None:
        """
        Turns a mixed integer program into a continuous one by fixing
        all the integer and binary variables to their solution values.

        !!! warning "Gurobi only"
            This method only works with the Gurobi solver. Open an issue if you'd like to see support for other solvers.

        Examples:
            >>> m = pf.Model(solver="gurobi")
            >>> m.X = pf.Variable(vtype=pf.VType.BINARY, lb=0)
            >>> m.Y = pf.Variable(vtype=pf.VType.INTEGER, lb=0)
            >>> m.Z = pf.Variable(lb=0)
            >>> m.my_constraint = m.X + m.Y + m.Z <= 10
            >>> m.maximize = 3 * m.X + 2 * m.Y + m.Z
            >>> m.optimize()
            >>> m.X.solution, m.Y.solution, m.Z.solution
            (1, 9, 0.0)
            >>> m.my_constraint.dual
            Traceback (most recent call last):
            ...
            RuntimeError: Unable to retrieve attribute 'Pi'
            >>> m.convert_to_fixed()
            >>> m.optimize()
            >>> m.my_constraint.dual
            1.0

            Only works for Gurobi:

            >>> m = pf.Model("max", solver="highs")
            >>> m.convert_to_fixed()
            Traceback (most recent call last):
            ...
            NotImplementedError: Method 'convert_to_fixed' is not implemented for solver 'highs'.
        """
        self.poi._converttofixed()

    @for_solvers("gurobi", "copt")
    def compute_IIS(self):
        """
        Computes the Irreducible Infeasible Set (IIS) of the model.

        !!! warning "Gurobi only"
            This method only works with the Gurobi solver. Open an issue if you'd like to see support for other solvers.

        Examples:
            >>> m = pf.Model(solver="gurobi")
            >>> m.X = pf.Variable(lb=0, ub=2)
            >>> m.Y = pf.Variable(lb=0, ub=2)
            >>> m.bad_constraint = m.X >= 3
            >>> m.minimize = m.X + m.Y
            >>> m.optimize()
            >>> m.attr.TerminationStatus
            <TerminationStatusCode.INFEASIBLE: 3>
            >>> m.bad_constraint.attr.IIS
            Traceback (most recent call last):
            ...
            RuntimeError: Unable to retrieve attribute 'IISConstr'
            >>> m.compute_IIS()
            >>> m.bad_constraint.attr.IIS
            True
        """
        self.poi.computeIIS()

    def dispose(self):
        """
        Disposes of the model and cleans up the solver environment.

        When using Gurobi compute server, this cleanup will
        ensure your run is not marked as 'ABORTED'.

        Note that once the model is disposed, it cannot be used anymore.

        Examples:
            >>> m = pf.Model()
            >>> m.X = pf.Variable(ub=1)
            >>> m.maximize = m.X
            >>> m.optimize()
            >>> m.X.solution
            1.0
            >>> m.dispose()
        """
        env = None
        if hasattr(self.poi, "_env"):
            env = self.poi._env
        self.poi.close()
        if env is not None:
            env.close()

    def __del__(self):
        # This ensures that the model is closed *before* the environment is. This avoids the Gurobi warning:
        #   Warning: environment still referenced so free is deferred (Continue to use WLS)
        # I include the hasattr check to avoid errors in case __init__ failed and poi was never set.
        if hasattr(self, "poi"):
            self.poi.close()

    def _set_param(self, name, value):
        self.poi.set_raw_parameter(name, value)

    def _get_param(self, name):
        return self.poi.get_raw_parameter(name)

    def _set_attr(self, name, value):
        try:
            self.poi.set_model_attribute(poi.ModelAttribute[name], value)
        except KeyError as e:
            if self.solver.name == "gurobi":
                self.poi.set_model_raw_attribute(name, value)
            else:
                raise e

    def _get_attr(self, name):
        try:
            return self.poi.get_model_attribute(poi.ModelAttribute[name])
        except KeyError as e:
            if self.solver.name == "gurobi":
                return self.poi.get_model_raw_attribute(name)
            else:
                raise e
