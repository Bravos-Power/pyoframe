import pytest
import polars as pl
from polars.testing import assert_frame_equal

from pyoframe.variables import Variable
from pyoframe.constraints import Expression


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
    expression = (
        5 * Variable()
        + 3.4 * Variable()
        - 2.1 * Variable()
        + 1.1231237019273 * Variable()
    )
    result = expression.to_str(include_header=False)
    expected_result = "5.0 x1 +3.4 x2 -2.1 x3 +1.1231237019273 x4"
    assert result == expected_result


def test_variables_to_string_with_dimensions(expression_with_dimensions: Expression):
    result = expression_with_dimensions.to_str_table()
    expected_result = pl.DataFrame(
        {
            "x": [1, 2, 1, 2],
            "y": [1, 1, 2, 2],
            "expr": [
                "[1,1]: 5.0 x1 +3.4 x5 -2.1 x9 +1.1231237019273 x13",
                "[2,1]: 5.0 x2 +3.4 x6 -2.1 x10 +1.1231237019273 x14",
                "[1,2]: 5.0 x3 +3.4 x7 -2.1 x11 +1.1231237019273 x15",
                "[2,2]: 5.0 x4 +3.4 x8 -2.1 x12 +1.1231237019273 x16",
            ],
        }
    )
    assert_frame_equal(
        result, expected_result, check_row_order=False, check_column_order=False
    )


def test_expression_with_const_to_str():
    expr = 5 + 2 * Variable()
    result = expr.to_str(include_header=False)
    expected_result = "2.0 x1 +5.0"
    assert result == expected_result


if __name__ == "__main__":
    pytest.main(args=[__file__])
