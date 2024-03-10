import pytest
import polars as pl
from polars.testing import assert_frame_equal
from convop.io import _expression_vars_to_string

from convop.variables import Variable


@pytest.fixture
def expression():
    return (
        5 * Variable()
        + 3.4 * Variable()
        - 2.1 * Variable()
        + 1.1231237019273 * Variable()
    )


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


def test_variables_to_string(expression):
    result = _expression_vars_to_string(expression, sort=True)
    assert len(result) == 1
    result = result["result"][0]
    expected_result = "+5.0 x0\n+3.4 x1\n-2.1 x2\n+1.1231237019273 x3\n"
    print(result)
    assert result == expected_result


def test_variables_to_string_with_dimensions(expression_with_dimensions):
    result = _expression_vars_to_string(expression_with_dimensions, sort=True)
    expected_result = pl.DataFrame(
        {
            "x": [1, 2, 1, 2],
            "y": [1, 1, 2, 2],
            "result": [
                "+5.0 x0\n+3.4 x4\n-2.1 x8\n+1.1231237019273 x12\n",
                "+5.0 x1\n+3.4 x5\n-2.1 x9\n+1.1231237019273 x13\n",
                "+5.0 x2\n+3.4 x6\n-2.1 x10\n+1.1231237019273 x14\n",
                "+5.0 x3\n+3.4 x7\n-2.1 x11\n+1.1231237019273 x15\n",
            ],
        }
    )
    assert_frame_equal(
        result, expected_result, check_row_order=False, check_column_order=False
    )


if __name__ == "__main__":
    pytest.main(args=[__file__])
