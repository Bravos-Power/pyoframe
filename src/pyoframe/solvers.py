from pathlib import Path
from typing import Optional


def solve(m, solver, output_dir: Optional[Path] = None, **kwargs):
    if output_dir is not None and not output_dir.exists():
        output_dir.mkdir(parents=True)

    if solver == "gurobi":
        return gurobi_solve(m, dir_path=output_dir, **kwargs)
    else:
        raise ValueError(f"Solver {solver} not recognized or supported.")


def gurobi_solve(model, dir_path: Optional[Path] = None, use_var_names=True):
    import gurobipy as gp

    if dir_path is None:
        dir_path = Path.cwd()

    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    problem_file = dir_path / f"{model.name}.lp"
    model.to_file(problem_file, use_var_names=use_var_names)
    gurobi_model = gp.read(str(problem_file))
    gurobi_model.optimize()
    if gurobi_model.status != gp.GRB.OPTIMAL:
        raise Exception(f"Optimization failed with status {gurobi_model.status}")

    gurobi_model.write(str(dir_path / f"{model.name}.sol"))
    return gurobi_model
