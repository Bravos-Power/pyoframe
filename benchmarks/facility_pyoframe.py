# Copyright (c) 2022: Miles Lubin and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import os
import time
import polars as pl
import pyoframe as pf
from pyoframe.constants import COEF_KEY

def solve_facility(solver, G, F):
    model = pf.Model("min")
    model.Grid = pf.Set(i=range(0, G))
    model.Grid_Matrix = model.Grid * model.Grid.rename({"i": "j"})
    model.Facs = pf.Set(f=range(1, F))
    model.Dims = pf.Set(d=[1, 2])
    model.y = pf.Variable(model.Facs, model.Dims, lb=0, ub=1)
    model.s = pf.Variable(model.Grid_Matrix, model.Facs, lb=0)
    model.z = pf.Variable(model.Grid_Matrix, model.Facs, vtype=pf.VType.BINARY)
    model.r = pf.Variable(model.Grid_Matrix, model.Facs, model.Dims)
    model.d = pf.Variable()
    model.objective = model.d
    model.assmt = pf.sum("f", model.z) == 1
    model.quadrhs = model.s == model.d.add_dim("i", "j", "f") + 2*(2**(1/2))*(1 - model.z)
    model.quaddistk1 = model.r.filter(d=1).drop("d") == (pl.DataFrame({"i": range(0, G), COEF_KEY: range(0, G)}).to_expr().add_dim("f") / G - model.y.filter(d=1).drop("d").add_dim("i")).add_dim("j")
    model.quaddistk2 = model.r.filter(d=2).drop("d") == (pl.DataFrame({"j": range(0, G), COEF_KEY: range(0, G)}).to_expr().add_dim("f") / G - model.y.filter(d=2).drop("d").add_dim("j")).add_dim("i")
    model.quaddist = model.r.filter(d=1).drop("d")**2 + model.r.filter(d=2).drop("d")**2 <= model.s**2 
    model.attr["timelimit"] = 0.0
    model.attr["presolve"] = False
    model.solve()
    return model

def main(Ns = [25, 50, 75, 100]):
    dir = os.path.realpath(os.path.dirname(__file__))
    for n in Ns:
        start = time.time()
        try:
            model = solve_facility('gurobi', n, n)
        except Exception as e:
            raise
        run_time = round(time.time() - start)
        with open(dir + "/benchmarks.csv", "a") as io:
            io.write("pyoframe fac-%i -1 %i\n" % (n, run_time))
    return

if __name__ == "__main__":
    main()
