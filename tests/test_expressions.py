import pandas as pd
import numpy as np
import convop


def test_drops_na():
    for na in [None, float("nan"), np.nan]:
        df = pd.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, na]}).set_index("dim1")[
            "value"
        ]
        constraint = 5 <= df.to_expr()

        expected_df = pd.DataFrame({"dim1": [1, 2], "value": [1, 2]}).set_index("dim1")[
            "value"
        ]
        expected_constraint = 5 <= expected_df.to_expr()
        assert constraint == expected_constraint
