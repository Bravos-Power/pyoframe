from __future__ import annotations

import pyoptinterface as poi

from pyoframe.constants import ObjSense
from pyoframe.core import Expression, SupportsToExpr


class Objective(Expression):
    """
    Examples:
        An `Objective` is automatically created when an `Expression` is assigned to `.minimize` or `.maximize`

        >>> m = pf.Model()
        >>> m.A, m.B = pf.Variable(lb=0), pf.Variable(lb=0)
        >>> m.con = m.A + m.B <= 10
        >>> m.maximize = 2 * m.B + 4
        >>> m.maximize
        <Objective size=1 dimensions={} terms=2>
        objective: 2 B +4

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
        >>> m.dimensioned_variable = pf.Variable({"city": ["Toronto", "Berlin", "Paris"]})
        >>> m.maximize = m.dimensioned_variable
        Traceback (most recent call last):
        ...
        ValueError: Objective cannot be created from a dimensioned expression. Did you forget to use pf.sum()?

        Objectives cannot be overwritten.

        >>> m = pf.Model()
        >>> m.A = pf.Variable(lb=0)
        >>> m.maximize = 2 * m.A
        >>> m.maximize = 3 * m.A
        Traceback (most recent call last):
        ...
        ValueError: An objective already exists. Use += or -= to modify it.
    """

    def __init__(
        self, expr: SupportsToExpr | int | float, _constructive: bool = False
    ) -> None:
        self._constructive = _constructive
        self._negated_for_ipopt = (
            False  # Add this flag to track if we negated the objective
        )

        if isinstance(expr, (int, float)):
            expr = Expression.constant(expr)
        else:
            expr = expr.to_expr()
        super().__init__(expr.data)
        self._model = expr._model
        if self.dimensions is not None:
            raise ValueError(
                "Objective cannot be created from a dimensioned expression. Did you forget to use pf.sum()?"
            )

    @property
    def value(self) -> float:
        """
        The value of the objective function (only available after solving the model).
        This value is obtained by directly querying the solver.
        """
        obj_value = self._model.poi.get_model_attribute(
            poi.ModelAttribute.ObjectiveValue
        )
        # If we're using IPOPT with maximization, negate the result back
        if self._negated_for_ipopt:
            return -obj_value
        return obj_value

    def on_add_to_model(self, model, name):
        super().on_add_to_model(model, name)
        assert self._model is not None
        if self._model.sense is None:
            raise ValueError(
                "Can't set an objective without specifying the sense. Did you use .objective instead of .minimize or .maximize ?"
            )

        # Check if we're using IPOPT and maximizing
        solver_name = self._model.solver_name
        is_ipopt = "ipopt" in solver_name
        is_maximizing = self._model.sense == ObjSense.MAX

        # Get the original objective function
        original_obj = self.to_poi()

        # Check if it's quadratic
        is_quadratic = isinstance(original_obj, poi.ScalarQuadraticFunction)

        # Handle IPOPT maximization case
        if is_ipopt and is_maximizing:
            # Set flag to remind us to negate the solution later
            self._negated_for_ipopt = True

            if is_quadratic:
                # For quadratic objectives, negate all coefficients
                negated_obj = poi.ScalarQuadraticFunction(
                    coefficients=[-c for c in original_obj.coefficients],
                    var1s=original_obj.var1s,
                    var2s=original_obj.var2s,
                )
            else:
                # For linear objectives, negate coefficients
                negated_obj = poi.ScalarAffineFunction(
                    coefficients=[-c for c in original_obj.coefficients],
                    variables=original_obj.variables,
                )

            # Always use minimize for IPOPT
            self._model.poi.set_objective(
                negated_obj, sense=poi.ObjectiveSense.Minimize
            )
        else:
            # Normal case for other solvers
            self._model.poi.set_objective(
                original_obj, sense=self._model.sense.to_poi()
            )

    def __iadd__(self, other):
        return Objective(self + other, _constructive=True)

    def __isub__(self, other):
        return Objective(self - other, _constructive=True)
