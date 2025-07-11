# Copyright (c) 2022: Miles Lubin and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.
# See https://github.com/jump-dev/JuMPPaperBenchmarks

import pyomo.environ as pyo

from benchmarks.util import PyomoBenchmark


class Bench(PyomoBenchmark):
    MAX_SIZE = 75  # After that it gets too slow

    def build(self):
        try:
            G, F = self.size
        except TypeError:
            G = F = self.size

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

        def assmt_rule(mod, i, j):
            return sum([mod.z[i, j, f] for f in mod.Facs]) == 1

        model.assmt = pyo.Constraint(model.Grid, model.Grid, rule=assmt_rule)
        M = 2 * 1.414

        def quadrhs_rule(mod, i, j, f):
            return mod.s[i, j, f] == mod.d + M * (1 - mod.z[i, j, f])

        model.quadrhs = pyo.Constraint(
            model.Grid, model.Grid, model.Facs, rule=quadrhs_rule
        )

        def quaddistk1_rule(mod, i, j, f):
            return mod.r[i, j, f, 1] == (1.0 * i) / mod.G - mod.y[f, 1]

        model.quaddistk1 = pyo.Constraint(
            model.Grid, model.Grid, model.Facs, rule=quaddistk1_rule
        )

        def quaddistk2_rule(mod, i, j, f):
            return mod.r[i, j, f, 2] == (1.0 * j) / mod.G - mod.y[f, 2]

        model.quaddistk2 = pyo.Constraint(
            model.Grid, model.Grid, model.Facs, rule=quaddistk2_rule
        )

        def quaddist_rule(mod, i, j, f):
            return (
                mod.r[i, j, f, 1] ** 2 + mod.r[i, j, f, 2] ** 2 <= mod.s[i, j, f] ** 2
            )

        model.quaddist = pyo.Constraint(
            model.Grid, model.Grid, model.Facs, rule=quaddist_rule
        )

        return model


if __name__ == "__main__":
    Bench("gurobi", 5).run()
