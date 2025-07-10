# Copyright (c) 2023: Yue Yang
# https://github.com/metab0t/PyOptInterface_benchmark
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import pyoptinterface as poi
from pyoptinterface import gurobi, copt
import numpy as np


def solve(solver, size):
    model_constructor = {
        "gurobi": gurobi.Model,
        "copt": copt.Model,
    }.get(solver, None)
    if model_constructor is None:
        raise ValueError(f"Unknown solver {solver}")

    m = model_constructor()
    if type(size) == int:
        size = (size, size)
    G, F = size
    # Create variables
    y = add_ndarray_variable(m, (F, 2), lb=0.0, ub=1.0)
    s = add_ndarray_variable(m, (G + 1, G + 1, F), lb=0.0)
    z = add_ndarray_variable(m, (G + 1, G + 1, F), domain=poi.VariableDomain.Binary)
    r = add_ndarray_variable(m, (G + 1, G + 1, F, 2))
    d = m.add_variable()

    # Set objective
    m.set_objective(d * 1.0)

    # Add constraints
    for i in range(G + 1):
        for j in range(G + 1):
            expr = poi.quicksum(z[i, j, :])
            m.add_linear_constraint(expr, poi.Eq, 1.0)

    M = 2 * 1.414
    for i in range(G + 1):
        for j in range(G + 1):
            for f in range(F):
                expr = s[i, j, f] - d - M * (1 - z[i, j, f])
                m.add_linear_constraint(expr, poi.Eq, 0.0)
                expr = r[i, j, f, 0] - i / G + y[f, 0]
                m.add_linear_constraint(expr, poi.Eq, 0.0)
                expr = r[i, j, f, 1] - j / G + y[f, 1]
                m.add_linear_constraint(expr, poi.Eq, 0.0)
                m.add_second_order_cone_constraint(
                    [s[i, j, f], r[i, j, f, 0], r[i, j, f, 1]]
                )

    # Optimize model
    m.set_model_attribute(poi.ModelAttribute.Silent, True)
    m.set_model_attribute(poi.ModelAttribute.TimeLimitSec, 0.0)
    # close presolve for gurobi and open for copt
    solver_name = m.get_model_attribute(poi.ModelAttribute.SolverName)
    if solver_name.lower() == "gurobi":
        m.set_raw_parameter("Presolve", 0)
    elif solver_name.lower() == "copt":
        m.set_raw_parameter("Presolve", 1)
    m.optimize()


def add_ndarray_variable(m, shape, **kwargs):
    array = np.empty(shape, dtype=object)
    array_flat = array.flat
    for i in range(array.size):
        array_flat[i] = m.add_variable(**kwargs)
    return array


if __name__ == "__main__":
    solve("gurobi", 5)
