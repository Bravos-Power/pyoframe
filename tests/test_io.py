import pytest
import polars as pl

from pyoframe import Expression, Variable, Model


def test_variables_to_string():
    m = Model()
    m.x1 = Variable()
    m.x2 = Variable()
    m.x3 = Variable()
    m.x4 = Variable()
    expression = 5 * m.x1 + 3.4 * m.x2 - 2.1 * m.x3 + 1.1231237019273 * m.x4
    assert str(expression) == "5 x1 +3.4 x2 -2.1 x3 +1.12312 x4"


def test_variables_to_string_with_dimensions():
    df = pl.DataFrame(
        {
            "x": [1, 2, 1, 2],
            "y": [1, 1, 2, 2],
        }
    )
    m = Model()
    m.v1 = Variable(df)
    m.v2 = Variable(df)
    m.v3 = Variable(df)
    m.v4 = Variable(df)

    expression_with_dimensions = (
        5 * m.v1 + 3.4 * m.v2 - 2.1 * m.v3 + 1.1231237019273 * m.v4
    )
    result = expression_with_dimensions.to_str(include_header=False)
    assert (
        result
        == """[1,1]: 5 v1[1,1] +3.4 v2[1,1] -2.1 v3[1,1] +1.12312 v4[1,1]
[2,1]: 5 v1[2,1] +3.4 v2[2,1] -2.1 v3[2,1] +1.12312 v4[2,1]
[1,2]: 5 v1[1,2] +3.4 v2[1,2] -2.1 v3[1,2] +1.12312 v4[1,2]
[2,2]: 5 v1[2,2] +3.4 v2[2,2] -2.1 v3[2,2] +1.12312 v4[2,2]"""
    )


def test_expression_with_const_to_str():
    m = Model()
    m.x1 = Variable()
    expr = 5 + 2 * m.x1
    assert str(expr) == "2 x1 +5"


if __name__ == "__main__":
    pytest.main([__file__])
