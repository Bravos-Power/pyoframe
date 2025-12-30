"""Defines the `Objective` class for optimization models."""

from __future__ import annotations

import pyoptinterface as poi

from pyoframe._constants import ObjSense
from pyoframe._core import Expression, Operable


# TODO don't subclass Expression to avoid a bunch of unnecessary functions being available.
class Objective(Expression):
    """The objective for an optimization model.

    Examples:
        An `Objective` is automatically created when an `Expression` is assigned to `.minimize` or `.maximize`

        >>> m = pf.Model()
        >>> m.A, m.B = pf.Variable(lb=0), pf.Variable(lb=0)
        >>> m.con = m.A + m.B <= 10
        >>> m.maximize = 2 * m.B + 4
        >>> m.maximize
        <Objective (linear) terms=2>
        2 B +4

        The objective value can be retrieved with from the solver once the model is solved using `.value`.

        >>> m.optimize()
        >>> m.maximize.value
        24.0

        Objectives support `+=` and `-=` operators.

        >>> m.maximize += 3 * m.A
        >>> m.optimize()
        >>> m.A.solution, m.B.solution
        (10.0, 0.0)
        >>> m.maximize -= 2 * m.A
        >>> m.optimize()
        >>> m.A.solution, m.B.solution
        (0.0, 10.0)

        Objectives cannot be created from dimensioned expressions since an objective must be a single expression.

        >>> m = pf.Model()
        >>> m.dimensioned_variable = pf.Variable(
        ...     {"city": ["Toronto", "Berlin", "Paris"]}
        ... )
        >>> m.maximize = m.dimensioned_variable
        Traceback (most recent call last):
        ...
        ValueError: Objective cannot be created from a dimensioned expression. Did you forget to use .sum()?

        Objectives cannot be overwritten.

        >>> m = pf.Model()
        >>> m.A = pf.Variable(lb=0)
        >>> m.maximize = 2 * m.A
        >>> m.maximize = 3 * m.A
        Traceback (most recent call last):
        ...
        ValueError: An objective already exists. Use += or -= to modify it.
    """

    def __init__(self, expr: Operable, _constructive: bool = False) -> None:
        self._constructive = _constructive
        if isinstance(expr, (int, float)):
            expr = Expression.constant(expr)
        else:
            expr = expr.to_expr()  # TODO don't rely on monkey patch
        super().__init__(expr.data, name="objective")
        self._model = expr._model
        if self.dimensions is not None:
            raise ValueError(
                "Objective cannot be created from a dimensioned expression. Did you forget to use .sum()?"
            )

    @property
    def value(self) -> float:
        """The value of the objective function (only available after solving the model).

        This value is obtained by directly querying the solver.
        """
        assert self._model is not None, (
            "Objective must be part of a model before it is queried."
        )

        if (
            self._model.attr.TerminationStatus
            == poi.TerminationStatusCode.OPTIMIZE_NOT_CALLED
        ):
            raise ValueError(
                "Cannot retrieve the objective value before calling model.optimize()."
            )

        obj_value: float = self._model.attr.ObjectiveValue
        if (
            not self._model.solver.supports_objective_sense
            and self._model.sense == ObjSense.MAX
        ):
            obj_value *= -1
        return obj_value

    def _on_add_to_model(self, model, name):
        super()._on_add_to_model(model, name)
        assert self._model is not None
        if self._model.sense is None:
            raise ValueError(
                "Can't set an objective without specifying the sense. Did you use .objective instead of .minimize or .maximize ?"
            )

        kwargs = {}
        if (
            not self._model.solver.supports_objective_sense
            and self._model.sense == ObjSense.MAX
        ):
            poi_expr = (-self)._to_poi()
            kwargs["sense"] = poi.ObjectiveSense.Minimize
        else:
            poi_expr = self._to_poi()
            kwargs["sense"] = self._model.sense._to_poi()
        self._model.poi.set_objective(poi_expr, **kwargs)

    def __iadd__(self, other):
        return Objective(self + other, _constructive=True)

    def __isub__(self, other):
        return Objective(self - other, _constructive=True)
