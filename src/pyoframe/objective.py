from typing import Union
from pyoframe.constants import ObjSense, ObjSenseValue
from pyoframe.constraints import SupportsMath, Expression


class Objective(Expression):
    r"""
    Examples:
        >>> from pyoframe import Variable, Model, sum
        >>> m = Model()
        >>> m.a = Variable()
        >>> m.b = Variable({"dim1": [1, 2, 3]})
        >>> m.maximize = m.a + sum("dim1", m.b)
        >>> m.maximize
        <Objective size=1 dimensions={} terms=4>
        maximize: a + b[1] + b[2] + b[3]
    """

    def __init__(
        self, expr: SupportsMath, sense: Union[ObjSense, ObjSenseValue]
    ) -> None:
        self.sense = ObjSense(sense)

        expr = expr.to_expr()
        super().__init__(expr.to_expr().data)
        self._model = expr._model
        assert (
            self.dimensions is None
        ), "Objective cannot have dimensions as it must be a single expression"
