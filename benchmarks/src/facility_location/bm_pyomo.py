"""Pyomo implementation of the facility location benchmark.

Copyright (c) 2022: Miles Lubin and contributors

Use of this source code is governed by an MIT-style license that can be found
in the LICENSE.md file or at https://opensource.org/licenses/MIT.
See https://github.com/jump-dev/JuMPPaperBenchmarks
"""

import pyomo.environ as pyo
from benchmark_utils.pyomo import Benchmark


class Bench(Benchmark):
    def build(self):
        G = F = self.size

        model = pyo.ConcreteModel()
        model.Grid = pyo.RangeSet(0, G)
        model.Facs = pyo.RangeSet(1, F)
        model.Dims = pyo.RangeSet(1, 2)
        model.facility_position = pyo.Var(model.Facs, model.Dims, bounds=(0.0, 1.0))
        model.dist = pyo.Var(model.Grid, model.Grid, model.Facs, bounds=(0.0, None))
        model.is_closest = pyo.Var(
            model.Grid, model.Grid, model.Facs, within=pyo.Binary
        )
        model.r = pyo.Var(model.Grid, model.Grid, model.Facs, model.Dims)
        model.max_distance = pyo.Var()
        model.obj = pyo.Objective(expr=1.0 * model.max_distance)

        model.assmt = pyo.Constraint(
            model.Grid,
            model.Grid,
            rule=lambda mod, i, j: sum([mod.is_closest[i, j, f] for f in mod.Facs])
            == 1,
        )
        M = 2 * 1.414

        model.con_max_distance = pyo.Constraint(
            model.Grid,
            model.Grid,
            model.Facs,
            rule=lambda mod, i, j, f: mod.dist[i, j, f]
            == mod.max_distance + M * (1 - mod.is_closest[i, j, f]),
        )

        model.quaddistk1 = pyo.Constraint(
            model.Grid,
            model.Grid,
            model.Facs,
            rule=lambda mod, i, j, f: mod.r[i, j, f, 1]
            == (1.0 * i) / G - mod.facility_position[f, 1],
        )

        model.quaddistk2 = pyo.Constraint(
            model.Grid,
            model.Grid,
            model.Facs,
            rule=lambda mod, i, j, f: mod.r[i, j, f, 2]
            == (1.0 * j) / G - mod.facility_position[f, 2],
        )

        model.quaddist = pyo.Constraint(
            model.Grid,
            model.Grid,
            model.Facs,
            rule=lambda mod, i, j, f: mod.r[i, j, f, 1] ** 2 + mod.r[i, j, f, 2] ** 2
            <= mod.dist[i, j, f] ** 2,
        )

        return model


if __name__ == "__main__":
    Bench("gurobi", 5).run()
