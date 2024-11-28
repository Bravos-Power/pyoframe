import pytest
import polars as pl

from pyoframe import Expression, Variable, Model


@pytest.fixture
def expression_with_dimensions():
    df = pl.DataFrame(
        {
            "x": [1, 2, 1, 2],
            "y": [1, 1, 2, 2],
        }
    )

    return (
        5 * Variable(df)
        + 3.4 * Variable(df)
        - 2.1 * Variable(df)
        + 1.1231237019273 * Variable(df)
    )


def test_variables_to_string():
    m = Model()
    m.x1 = Variable()
    m.x2 = Variable()
    m.x3 = Variable()
    m.x4 = Variable()
    expression = 5 * m.x1 + 3.4 * m.x2 - 2.1 * m.x3 + 1.1231237019273 * m.x4
    assert str(expression) == "5 x1 +3.4 x2 -2.1 x3 +1.1231237019273 x4"


def test_variables_to_string_with_dimensions(expression_with_dimensions: Expression):
    result = expression_with_dimensions.to_str(include_header=False)
    expected_result = "\n".join(
        [
            "[1,1]: 5 x1 +3.4 x5 -2.1 x9 +1.1231237019273 x13",
            "[2,1]: 5 x2 +3.4 x6 -2.1 x10 +1.1231237019273 x14",
            "[1,2]: 5 x3 +3.4 x7 -2.1 x11 +1.1231237019273 x15",
            "[2,2]: 5 x4 +3.4 x8 -2.1 x12 +1.1231237019273 x16",
        ]
    )
    assert result == expected_result


def test_expression_with_const_to_str():
    m = Model()
    m.x1 = Variable()
    expr = 5 + 2 * m.x1
    assert str(expr) == "2 x1 +5"

if __name__ == "__main__":
    pytest.main([__file__])
