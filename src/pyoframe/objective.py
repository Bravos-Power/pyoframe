import pyoptinterface as poi

from pyoframe.core import Expression, SupportsMath


class Objective(Expression):
    """
    Examples:
        Objectives are created just like expressions.

        >>> m = pf.Model("max")
        >>> m.A, m.B = pf.Variable(lb=0), pf.Variable(lb=0)
        >>> m.con = m.A + m.B <= 10
        >>> m.objective = 2 * m.B + 4
        >>> m.objective
        <Objective size=1 dimensions={} terms=2>
        objective: 2 B +4

        Objective values can be retrieved directly from the solver with `.value`

        >>> m.optimize()
        >>> m.objective.value
        24.0

        Objectives support `+=` and `-=` operators.

        >>> m.objective += 3 * m.A
        >>> m.optimize()
        >>> m.A.solution, m.B.solution
        (10.0, 0.0)
    """

    def __init__(self, expr: SupportsMath) -> None:
        expr = expr.to_expr()
        super().__init__(expr.data)
        self._model = expr._model
        assert (
            self.dimensions is None
        ), "Objective cannot have dimensions as it must be a single expression"

    @property
    def value(self) -> float:
        """
        The value of the objective function (only available after solving the model).

        This value is obtained by directly querying the solver.
        """
        return self._model.solver_model.get_model_attribute(
            poi.ModelAttribute.ObjectiveValue
        )

    def on_add_to_model(self, model, name):
        super().on_add_to_model(model, name)
        self._model.solver_model.set_objective(
            self.to_poi(), sense=self._model.sense.to_poi()
        )
