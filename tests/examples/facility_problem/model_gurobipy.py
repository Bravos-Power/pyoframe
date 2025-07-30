"""Gurobipy formulation of the facility problem.

Adapted from https://www.gurobi.com/documentation/current/examples/facility_py.html#subsubsection:facility.py
Copyright 2024, Gurobi Optimization, LLC

Facility location: a company currently ships its product from 5 plants
to 4 warehouses. It is considering closing some plants to reduce
costs. What plant(s) should the company close, in order to minimize
transportation and fixed costs?

Based on an example from Frontline Systems:
  http://www.solver.com/disfacility.htm
Used with permission.
"""

import os
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

_input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"


def main():
    # Warehouse demand in thousands of units
    demand = pd.read_csv(_input_dir / "wharehouses.csv")["demand"].to_list()

    # Plant capacity in thousands of units
    capacity = pd.read_csv(_input_dir / "plants.csv")["capacity"].to_list()

    # Fixed costs for each plant
    fixedCosts = pd.read_csv(_input_dir / "plants.csv")["fixed_cost"].to_list()

    # Transportation costs per thousand units
    transCosts = (
        pd.read_csv(_input_dir / "transport_costs.csv")
        .set_index("wharehouse")
        .values.tolist()
    )

    # Range of plants and warehouses
    plants = range(len(capacity))
    warehouses = range(len(demand))

    # Model
    m = gp.Model()

    # Plant open decision variables: open[p] == 1 if plant p is open.
    open = m.addVars(plants, vtype=GRB.BINARY, obj=fixedCosts, name="open")

    # Transportation decision variables: transport[w,p] captures the
    # optimal quantity to transport to warehouse w from plant p
    transport = m.addVars(warehouses, plants, obj=transCosts, name="transport")

    # The objective is to minimize the total fixed and variable costs
    m.ModelSense = GRB.MINIMIZE

    # Production constraints
    # Note that the right-hand limit sets the production to zero if the plant
    # is closed
    m.addConstrs(
        (transport.sum("*", p) <= capacity[p] * open[p] for p in plants), "Capacity"
    )

    # Demand constraints
    m.addConstrs((transport.sum(w) == demand[w] for w in warehouses), "Demand")

    # Use barrier to solve root relaxation
    m.params.Method = 2

    m.optimize()
    return m
