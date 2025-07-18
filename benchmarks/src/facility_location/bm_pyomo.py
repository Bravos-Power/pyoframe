# Copyright (c) 2022: Miles Lubin and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.
# See https://github.com/jump-dev/JuMPPaperBenchmarks

import pyomo.environ as pyo

from benchmarks.utils import PyomoBenchmark


class Bench(PyomoBenchmark):
    def build(self):
        if isinstance(self.size, int):
            G = F = self.size
        else:
            G, F = self.size

        model = pyo.ConcreteModel()
        model.G = G
        model.F = F
        model.Grid = pyo.RangeSet(0, model.G)
        model.Facs = pyo.RangeSet(1, model.F)
        model.Dims = pyo.RangeSet(1, 2)
        model.y = pyo.Var(model.Facs, model.Dims, bounds=(0.0, 1.0))
        model.s = pyo.Var(model.Grid, model.Grid, model.Facs, bounds=(0.0, None))
        model.z = pyo.Var(model.Grid, model.Grid, model.Facs, within=pyo.Binary)
        model.r = pyo.Var(model.Grid, model.Grid, model.Facs, model.Dims)
        model.d = pyo.Var()
        model.obj = pyo.Objective(expr=1.0 * model.d)

        model.assmt = pyo.Constraint(
            model.Grid,
            model.Grid,
            rule=lambda mod, i, j: sum([mod.z[i, j, f] for f in mod.Facs]) == 1,
        )
        M = 2 * 1.414

        model.quadrhs = pyo.Constraint(
            model.Grid,
            model.Grid,
            model.Facs,
            rule=lambda mod, i, j, f: mod.s[i, j, f]
            == mod.d + M * (1 - mod.z[i, j, f]),
        )

        model.quaddistk1 = pyo.Constraint(
            model.Grid,
            model.Grid,
            model.Facs,
            rule=lambda mod, i, j, f: mod.r[i, j, f, 1]
            == (1.0 * i) / mod.G - mod.y[f, 1],
        )

        model.quaddistk2 = pyo.Constraint(
            model.Grid,
            model.Grid,
            model.Facs,
            rule=lambda mod, i, j, f: mod.r[i, j, f, 2]
            == (1.0 * j) / mod.G - mod.y[f, 2],
        )

        model.quaddist = pyo.Constraint(
            model.Grid,
            model.Grid,
            model.Facs,
            rule=lambda mod, i, j, f: mod.r[i, j, f, 1] ** 2 + mod.r[i, j, f, 2] ** 2
            <= mod.s[i, j, f] ** 2,
        )

        return model


if __name__ == "__main__":
    Bench("gurobi", 5).run()
