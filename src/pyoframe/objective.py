import pyoptinterface as poi

from pyoframe.core import Expression, SupportsMath


class Objective(Expression):
    r"""
    Examples:
        >>> from pyoframe import Variable, Model, sum
        >>> m = Model("max")
        >>> m.a = Variable()
        >>> m.b = Variable({"dim1": [1, 2, 3]})
        >>> m.objective = m.a + sum("dim1", m.b)
        >>> m.objective
        <Objective size=1 dimensions={} terms=4>
        objective: a + b[1] + b[2] + b[3]
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
        return self._model.solver_model.get_model_attribute(
            poi.ModelAttribute.ObjectiveValue
        )

    def on_add_to_model(self, model, name):
        super().on_add_to_model(model, name)
        self._model.solver_model.set_objective(
            self.to_poi(), sense=self._model.sense.to_poi()
        )
