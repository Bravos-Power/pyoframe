from typing import Optional
from pyoframe.constants import COEF_KEY
from pyoframe.core import SupportsMath, Expression


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
        self._value: Optional[float] = None

    @property
    def value(self):
        if self._value is None:
            raise ValueError(
                "Objective value is not available before solving the model"
            )
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def has_constant(self):
        constant_terms = self.constant_terms
        if len(constant_terms) == 0:
            return False
        return constant_terms.get_column(COEF_KEY).item() != 0
