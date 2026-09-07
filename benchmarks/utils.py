"""Helper functions for the benchmarks."""

LIBRARY_NAMES = [
    "Pyoframe",
    "Pyomo",
    "PuLP",
    "CVXPY",
    "PyOptInterface",
    "Linopy",
    "JuMP",
    "AMPL",
]

LIBRARY_NAME_MAP = {n.lower(): n for n in LIBRARY_NAMES}

PROBLEM_NAME_MAP = {
    "facility_location": "Facility Location Problem (no data)",
    "simple_problem": "Trivial Data Problem",
    "energy_planning_security_constrained_dispatch": "Electrical Grid Dispatch Problem",
    "energy_planning_capacity_expansion": "Electrical Grid Capacity Expansion Problem",
}


def human_format(num):
    for unit in ["", "k", "M", "B", "T"]:
        if abs(num) < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000
    return f"{num:.0f}P"
