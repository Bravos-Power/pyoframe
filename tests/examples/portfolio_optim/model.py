"""Quadratic Portfolio Optimization Model (Markowitz Mean-Variance).

Minimize portfolio variance subject to achieving a minimum return.

Variables:
    w_i = fraction of portfolio invested in asset i

Parameters:
    r_i = expected return of asset i
    σ_ij = covariance between assets i and j
    r_min = minimum required return (10%)
    w_max = maximum weight per asset (50%)

Model:
    minimize    Σ_i Σ_j w_i * σ_ij * w_j        (portfolio variance)

    subject to  Σ_i w_i = 1                      (weights sum to 1)
                Σ_i r_i * w_i ≥ r_min            (minimum return)
                0 ≤ w_i ≤ w_max  ∀i              (weight bounds)

This is a convex quadratic program (QP) that tests IPOPT's ability to handle
nonlinear objectives. Both IPOPT and Gurobi should find the same optimal solution
since the problem has a unique global optimum.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import pyoframe as pf

_input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"


def solve_model(use_var_names=True):
    """Portfolio Optimization Test Case - Quadratic Programming with IPOPT.

    This example demonstrates IPOPT's capability to solve quadratic programming problems
    by implementing the classic Markowitz mean-variance portfolio optimization.

    Problem: Select portfolio weights to minimize risk (variance) while achieving a
    target return.

    Mathematical Formulation:

        minimize    sum_i sum_j (w_i * cov_ij * w_j)     [portfolio variance]

        subject to  sum_i w_i = 1                         [fully invested]
                    sum_i (r_i * w_i) >= r_min            [minimum return]
                    0 <= w_i <= w_max for all i           [position limits]

        where:
            w_i = weight of asset i in portfolio
            cov_ij = covariance between assets i and j
            r_i = expected return of asset i
            r_min = 0.10 (10% minimum required return)
            w_max = 0.50 (50% maximum weight per asset)

    Test Data:
        - 5 assets (A, B, C, D, E) with returns from 5% to 18%
        - Full 5x5 covariance matrix (symmetric, positive definite)
        - Convex problem guarantees unique global optimum

    Expected Result:
        Both Gurobi (QP solver) and IPOPT (nonlinear solver) should find the same
        optimal portfolio with minimum variance of ~0.0195 that achieves exactly
        10% expected return.

    This tests the quadratic objective support added to pyoframe for IPOPT integration.
    """
    # Read input data
    assets = pd.read_csv(_input_dir / "assets.csv").set_index("asset")
    covariance = pd.read_csv(_input_dir / "covariance.csv").set_index(
        ["asset_i", "asset_j"]
    )
    params = pd.read_csv(_input_dir / "portfolio_params.csv").set_index("param")[
        "value"
    ]

    min_return = params.loc["min_return"]
    max_weight = params.loc["max_weight"]

    # Create model
    m = pf.Model(solver_uses_variable_names=use_var_names)

    # Decision variables: portfolio weights
    m.weight = pf.Variable(assets.index, lb=0, ub=max_weight)

    # Constraint: weights must sum to 1
    m.con_weights_sum = m.weight.sum() == 1

    # Constraint: minimum expected return
    m.con_min_return = (m.weight * assets["expected_return"]).sum() >= min_return

    # Objective: minimize portfolio variance (quadratic)
    # Variance = sum over i,j of weight_i * cov_ij * weight_j
    # We need to match dimensions properly
    weight_i = m.weight.rename({"asset": "asset_i"})
    weight_j = m.weight.rename({"asset": "asset_j"})

    # Create the quadratic expression
    quad_expr = weight_i * covariance["covariance"] * weight_j

    m.minimize = quad_expr.sum()

    # Optimize
    m.optimize()

    return m


def print_results(model):
    print("\nOptimization Results:")
    print("=" * 50)

    # Print optimal weights
    print("\nOptimal Portfolio Weights:")
    weights = model.weight.solution

    # Weights is a polars DataFrame with 'asset' and 'solution' columns
    for asset, weight in weights.iter_rows():
        print(f"  {asset}: {weight:.4f}")

    # Calculate and print portfolio metrics
    assets = pd.read_csv(_input_dir / "assets.csv").set_index("asset")

    # Convert polars solution to dictionary for easier calculations
    weights_dict = {row[0]: row[1] for row in weights.iter_rows()}

    portfolio_return = sum(
        weights_dict[asset] * assets.loc[asset, "expected_return"]
        for asset in weights_dict
    )
    print(f"\nPortfolio Expected Return: {portfolio_return:.4f}")
    print(f"Portfolio Variance: {model.objective.value:.6f}")
    print(f"Portfolio Standard Deviation: {model.objective.value**0.5:.4f}")


if __name__ == "__main__":
    # Test with all three solvers that support quadratics
    print("Testing with Gurobi:")
    pf.Config.default_solver = "gurobi"
    model_gurobi = solve_model(use_var_names=True)
    print_results(model_gurobi)

    print("\n" + "=" * 70 + "\n")

    print("Testing with COPT:")
    pf.Config.default_solver = "copt"
    model_copt = solve_model(use_var_names=True)
    print_results(model_copt)

    print("\n" + "=" * 70 + "\n")

    print("Testing with IPOPT:")
    pf.Config.default_solver = "ipopt"
    model_ipopt = solve_model(use_var_names=True)
    print_results(model_ipopt)

    # Compare all three solutions
    print("\n" + "=" * 70 + "\n")
    print("Comparing solutions:")

    gurobi_weights = {
        row[0]: row[1] for row in model_gurobi.weight.solution.iter_rows()
    }
    copt_weights = {row[0]: row[1] for row in model_copt.weight.solution.iter_rows()}
    ipopt_weights = {row[0]: row[1] for row in model_ipopt.weight.solution.iter_rows()}

    max_diff = 0
    for asset in gurobi_weights:
        diff_copt = abs(gurobi_weights[asset] - copt_weights[asset])
        diff_ipopt = abs(gurobi_weights[asset] - ipopt_weights[asset])
        max_diff = max(max_diff, diff_copt, diff_ipopt)
        print(
            f"  {asset}: Gurobi={gurobi_weights[asset]:.6f}, "
            f"COPT={copt_weights[asset]:.6f}, "
            f"IPOPT={ipopt_weights[asset]:.6f}"
        )

    print(f"\nMaximum weight difference: {max_diff:.6f}")
    print(f"Gurobi objective: {model_gurobi.objective.value:.6f}")
    print(f"COPT objective: {model_copt.objective.value:.6f}")
    print(f"IPOPT objective: {model_ipopt.objective.value:.6f}")

    if max_diff < 1e-3:
        print("\n✓ All solvers found the same solution (within tolerance)")
    else:
        print("\n✗ Solutions differ significantly")

    # Manually calculate portfolio variance to verify
    print("\nManual variance calculation:")
    covariance_df = pd.read_csv(_input_dir / "covariance.csv")
    cov_matrix = covariance_df.pivot(
        index="asset_i", columns="asset_j", values="covariance"
    )

    # Get weights as a numpy array in the correct order
    assets_list = ["A", "B", "C", "D", "E"]
    w = np.array([gurobi_weights[asset] for asset in assets_list])

    # Calculate variance: w' * Σ * w
    variance_manual = np.dot(w, np.dot(cov_matrix.values, w))
    print(f"Manual calculation: {variance_manual:.6f}")
    print(f"Gurobi reported: {model_gurobi.objective.value:.6f}")
    print(f"COPT reported: {model_copt.objective.value:.6f}")
    print(f"IPOPT reported: {model_ipopt.objective.value:.6f}")

    if max_diff < 1e-3:
        print("\n✓ All solvers found the same solution (within tolerance)")
    else:
        print("\n✗ Solutions differ significantly")
