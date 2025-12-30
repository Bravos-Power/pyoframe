"""Defines the `Model` class for Pyoframe."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pyoptinterface as poi

from pyoframe._constants import (
    CONST_TERM,
    SUPPORTED_SOLVER_TYPES,
    SUPPORTED_SOLVERS,
    Config,
    ObjSense,
    ObjSenseValue,
    PyoframeError,
    VType,
    _Solver,
)
from pyoframe._core import Constraint, Operable, Variable
from pyoframe._model_element import BaseBlock
from pyoframe._objective import Objective
from pyoframe._utils import Container, NamedVariableMapper, for_solvers, get_obj_repr

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator


class Model:
    """The founding block of any Pyoframe optimization model onto which variables, constraints, and an objective can be added.

    Parameters:
        solver:
            The solver to use. If `None`, Pyoframe will try to use whichever solver is installed
            (unless [Config.default_solver][pyoframe._Config.default_solver] was changed from its default value of `auto`).
        solver_env:
            Gurobi only: a dictionary of parameters to set when creating the Gurobi environment.
        name:
            The name of the model. Currently it is not used for much.
        solver_uses_variable_names:
            If `True`, the solver will use your custom variable names in its outputs (e.g. during [`Model.write()`][pyoframe.Model.write]).
            This can be useful for debugging `.lp`, `.sol`, and `.ilp` files, but may worsen performance.
        print_uses_variable_names:
            If `True`, pyoframe will use your custom variables names when printing elements of the model to the console.
            This is useful for debugging, but may slightly worsen performance.
        sense:
            Either "min" or "max". Indicates whether it's a minimization or maximization problem.
            Typically, this parameter can be omitted (`None`) as it will automatically be
            set when the objective is set using `.minimize` or `.maximize`.

    Examples:
        >>> m = pf.Model()
        >>> m.X = pf.Variable()
        >>> m.my_constraint = m.X <= 10
        >>> m
        <Model vars=1 constrs=1 has_objective=False solver=gurobi>

        Use `solver_env` to, for example, connect to a Gurobi Compute Server:
        >>> m = pf.Model(
        ...     "gurobi",
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
        "objective",
        "_var_map",
        "name",
        "solver",
        "_poi",
        "_params",
        "params",
        "_attr",
        "attr",
        "sense",
        "_solver_uses_variable_names",
        "ONE",
        "solver_name",
        "minimize",
        "maximize",
    ]

    def __init__(
        self,
        solver: SUPPORTED_SOLVER_TYPES | _Solver | None = None,
        solver_env: dict[str, str] | None = None,
        *,
        name: str | None = None,
        solver_uses_variable_names: bool = False,
        print_uses_variable_names: bool = True,
        sense: ObjSense | ObjSenseValue | None = None,
    ):
        self._poi, self.solver = Model._create_poi_model(solver, solver_env)
        self.solver_name: str = self.solver.name
        self._variables: list[Variable] = []
        self._constraints: list[Constraint] = []
        self.sense: ObjSense | None = ObjSense(sense) if sense is not None else None
        self._objective: Objective | None = None
        self._var_map = NamedVariableMapper() if print_uses_variable_names else None
        self.name: str | None = name

        self._params = Container(self._set_param, self._get_param)
        self._attr = Container(self._set_attr, self._get_attr)
        self._solver_uses_variable_names = solver_uses_variable_names

    @property
    def poi(self):
        """The underlying PyOptInterface model used to interact with the solver.

        Modifying the underlying model directly is not recommended and may lead to unexpected behaviors.
        """
        return self._poi

    @property
    def solver_uses_variable_names(self):
        """Whether to pass human-readable variable names to the solver."""
        return self._solver_uses_variable_names

    @property
    def attr(self) -> Container:
        """An object that allows reading and writing model attributes.

        Several model attributes are common across all solvers making it easy to switch between solvers (see supported attributes for
        [Gurobi](https://metab0t.github.io/PyOptInterface/gurobi.html#supported-model-attribute),
        [HiGHS](https://metab0t.github.io/PyOptInterface/highs.html),
        [Ipopt](https://metab0t.github.io/PyOptInterface/ipopt.html)), and
        [COPT](https://metab0t.github.io/PyOptInterface/copt.html).

        We additionally support all of [Gurobi's attributes](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes.html#sec:Attributes) when using Gurobi.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable(lb=1, ub=1, vtype="integer")
            >>> m.attr.Silent = True  # Prevent solver output from being printed
            >>> m.optimize()
            >>> m.attr.TerminationStatus
            <TerminationStatusCode.OPTIMAL: 2>

            Some attributes, like `NumVars`, are solver-specific.
            >>> m = pf.Model("gurobi")
            >>> m.attr.NumConstrs
            0
            >>> m = pf.Model("highs")
            >>> m.attr.NumConstrs
            Traceback (most recent call last):
            ...
            KeyError: 'NumConstrs'

        See Also:
            [Variable.attr][pyoframe.Variable.attr] for setting variable attributes and
            [Constraint.attr][pyoframe.Constraint.attr] for setting constraint attributes.
        """
        return self._attr

    @property
    def params(self) -> Container:
        """An object that allows reading and writing solver-specific parameters.

        See the list of available parameters for
        [Gurobi](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#sec:Parameters),
        [HiGHS](https://ergo-code.github.io/HiGHS/stable/options/definitions/),
        [Ipopt](https://coin-or.github.io/Ipopt/OPTIONS.html),
        and [COPT](https://guide.coap.online/copt/en-doc/parameter.html).

        Examples:
            For example, if you'd like to use Gurobi's barrier method, you can set the `Method` parameter:
            >>> m = pf.Model("gurobi")
            >>> m.params.Method = 2
        """
        return self._params

    @classmethod
    def _create_poi_model(
        cls, solver: str | _Solver | None, solver_env: dict[str, str] | None
    ):
        if solver is None:
            if Config.default_solver == "raise":
                raise ValueError(
                    "No solver specified during model construction and automatic solver detection is disabled."
                )
            elif Config.default_solver == "auto":
                for solver_option in SUPPORTED_SOLVERS:
                    try:
                        return cls._create_poi_model(solver_option, solver_env)
                    except RuntimeError:
                        pass
                raise RuntimeError(
                    'Could not automatically find a solver. Is one installed? If so, specify which one: e.g. Model("gurobi")'
                )
            elif isinstance(Config.default_solver, (_Solver, str)):
                solver = Config.default_solver
            else:
                raise ValueError(
                    f"Config.default_solver has an invalid value: {Config.default_solver}."
                )

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
                    "Failed to import the Ipopt solver. Did you run `pip install pyoptinterface[nlp]`?"
                ) from e

            try:
                model = ipopt.Model()
            except RuntimeError as e:  # pragma: no cover
                if "IPOPT library is not loaded" in str(e):
                    raise RuntimeError(
                        "Could not find the Ipopt solver. Are you sure you've properly installed it and added it to your PATH?"
                    ) from e
                raise e
        elif solver.name == "copt":
            from pyoptinterface import copt

            if solver_env is None:
                env = copt.Env()
            else:
                # COPT uses EnvConfig for configuration
                env_config = copt.EnvConfig()
                for key, value in solver_env.items():
                    env_config.set(key, value)
                env = copt.Env(env_config)
            model = copt.Model(env)
        else:
            raise ValueError(
                f"Solver {solver} not recognized or supported."
            )  # pragma: no cover

        constant_var = model.add_variable(lb=1, ub=1, name="ONE")
        assert constant_var.index == CONST_TERM, (
            "The first variable should have index 0."
        )
        return model, solver

    @property
    def variables(self) -> list[Variable]:
        """Returns a list of the model's variables."""
        return self._variables

    @property
    def binary_variables(self) -> Generator[Variable]:
        """Returns the model's binary variables.

        Examples:
            >>> m = pf.Model()
            >>> m.X = pf.Variable(vtype=pf.VType.BINARY)
            >>> m.Y = pf.Variable()
            >>> len(list(m.binary_variables))
            1
        """
        return (v for v in self.variables if v.vtype == VType.BINARY)

    @property
    def integer_variables(self) -> Generator[Variable]:
        """Returns the model's integer variables.

        Examples:
            >>> m = pf.Model()
            >>> m.X = pf.Variable(vtype=pf.VType.INTEGER)
            >>> m.Y = pf.Variable()
            >>> len(list(m.integer_variables))
            1
        """
        return (v for v in self.variables if v.vtype == VType.INTEGER)

    @property
    def constraints(self) -> list[Constraint]:
        """Returns the model's constraints."""
        return self._constraints

    @property
    def has_objective(self) -> bool:
        """Returns whether the model's objective has been defined.

        Examples:
            >>> m = pf.Model()
            >>> m.has_objective
            False
            >>> m.X = pf.Variable()
            >>> m.maximize = m.X
            >>> m.has_objective
            True
        """
        return self._objective is not None

    @property
    def objective(self) -> Objective:
        """Returns the model's objective.

        Raises:
            ValueError: If the objective has not been defined.

        Examples:
            >>> m = pf.Model()
            >>> m.X = pf.Variable()
            >>> m.objective
            Traceback (most recent call last):
            ...
            ValueError: Objective is not defined.
            >>> m.maximize = m.X
            >>> m.objective
            <Objective (linear) terms=1>
            X

        See Also:
            [`Model.has_objective`][pyoframe.Model.has_objective]
        """
        if self._objective is None:
            raise ValueError("Objective is not defined.")
        return self._objective

    @objective.setter
    def objective(self, value: Operable):
        if self.has_objective and (
            not isinstance(value, Objective) or not value._constructive
        ):
            raise ValueError("An objective already exists. Use += or -= to modify it.")
        if not isinstance(value, Objective):
            value = Objective(value)
        self._objective = value
        value._on_add_to_model(self, "objective")

    @property
    def minimize(self) -> Objective | None:
        """Sets or gets the model's objective for minimization problems."""
        if self.sense != ObjSense.MIN:
            raise ValueError("Can't get .minimize in a maximization problem.")
        return self._objective

    @minimize.setter
    def minimize(self, value: Operable):
        if self.sense is None:
            self.sense = ObjSense.MIN
        if self.sense != ObjSense.MIN:
            raise ValueError("Can't set .minimize in a maximization problem.")
        self.objective = value

    @property
    def maximize(self) -> Objective | None:
        """Sets or gets the model's objective for maximization problems."""
        if self.sense != ObjSense.MAX:
            raise ValueError("Can't get .maximize in a minimization problem.")
        return self._objective

    @maximize.setter
    def maximize(self, value: Operable):
        if self.sense is None:
            self.sense = ObjSense.MAX
        if self.sense != ObjSense.MAX:
            raise ValueError("Can't set .maximize in a minimization problem.")
        self.objective = value

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name not in Model._reserved_attributes and not isinstance(
            __value, (BaseBlock, pl.DataFrame, pd.DataFrame)
        ):
            raise PyoframeError(
                f"Cannot set attribute '{__name}' on the model because it isn't a subtype of BaseBlock (e.g. Variable, Constraint, ...)"
            )

        if isinstance(__value, BaseBlock) and __name not in Model._reserved_attributes:
            if __value._get_id_column_name() is not None:
                assert not hasattr(self, __name), (
                    f"Cannot create {__name} since it was already created."
                )

            __value._on_add_to_model(self, __name)

            if isinstance(__value, Variable):
                self._variables.append(__value)
                if self._var_map is not None:
                    self._var_map.add(__value)
            elif isinstance(__value, Constraint):
                self._constraints.append(__value)
        return super().__setattr__(__name, __value)

    # Defining a custom __getattribute__ prevents type checkers from complaining about attribute access
    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        return get_obj_repr(
            self,
            f"'{self.name}'" if self.name is not None else None,
            vars=len(self.variables),
            constrs=len(self.constraints),
            has_objective=self.has_objective,
            solver=self.solver_name,
        )

    def write(self, file_path: Path | str, pretty: bool = False):
        """Outputs the model or the solution to a file (e.g. a `.lp`, `.sol`, `.mps`, or `.ilp` file).

        These files can be useful for manually debugging a model.
        Consult your solver documentation to learn more.

        When creating your model, set [`solver_uses_variable_names`][pyoframe.Model]
        to make the outputed file human-readable.

        ```python
        m = pf.Model(solver_uses_variable_names=True)
        ```

        For Gurobi, `solver_uses_variable_names=True` is mandatory when using
        .write(). This may become mandatory for other solvers too without notice.

        Parameters:
            file_path:
                The path to the file to write to.
            pretty:
                Only used when writing .sol files in HiGHS. If `True`, will use HiGH's pretty print columnar style which contains more information.
        """
        if not self.solver.supports_write:
            raise NotImplementedError(f"{self.solver.name} does not support .write()")
        if (
            not self.solver_uses_variable_names
            and self.solver.accelerate_with_repeat_names
        ):
            raise ValueError(
                f"{self.solver.name} requires solver_uses_variable_names=True to use .write()"
            )

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        kwargs = {}
        if self.solver.name == "highs":
            if self.solver_uses_variable_names:
                self.params.write_solution_style = 1
            kwargs["pretty"] = pretty
        self.poi.write(str(file_path), **kwargs)

    def optimize(self):
        """Optimizes the model using your selected solver (e.g. Gurobi, HiGHS)."""
        self.poi.optimize()

    @for_solvers("gurobi")
    def convert_to_fixed(self) -> None:
        """Gurobi only: Converts a mixed integer program into a continuous one by fixing all the non-continuous variables to their solution values.

        !!! warning "Gurobi only"
            This method only works with the Gurobi solver. Open an issue if you'd like to see support for other solvers.

        Examples:
            >>> m = pf.Model("gurobi")
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

            >>> m = pf.Model("highs")
            >>> m.convert_to_fixed()
            Traceback (most recent call last):
            ...
            NotImplementedError: Method 'convert_to_fixed' is not implemented for solver 'highs'.
        """
        self.poi._converttofixed()

    @for_solvers("gurobi", "copt")
    def compute_IIS(self):
        """Gurobi and COPT only: Computes the Irreducible Infeasible Set (IIS) of the model.

        !!! warning "Gurobi and COPT only"
            This method only works with the Gurobi and COPT solver. Open an issue if you'd like to see support for other solvers.

        Examples:
            >>> m = pf.Model("gurobi")
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
        """Disposes of the model and cleans up the solver environment.

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
        try:
            self.poi.set_raw_parameter(name, value)
        except KeyError as e:
            raise KeyError(
                f"Unknown parameter: '{name}'. See https://bravos-power.github.io/pyoframe/latest/learn/getting-started/solver-access/ for a list of valid parameters."
            ) from e

    def _get_param(self, name):
        try:
            return self.poi.get_raw_parameter(name)
        except KeyError as e:
            raise KeyError(
                f"Unknown parameter: '{name}'. See https://bravos-power.github.io/pyoframe/latest/learn/getting-started/solver-access/ for a list of valid parameters."
            ) from e

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
