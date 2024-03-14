from pathlib import Path


def solve(m, solver, *args, **kwargs):
    if solver == "gurobi":
        return gurobi_solve(m, *args, **kwargs)
    else:
        raise ValueError(f"Solver {solver} not recognized or supported.")


def gurobi_solve(model, dir_path: Path, use_var_names=True):
    import gurobipy as gp

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
