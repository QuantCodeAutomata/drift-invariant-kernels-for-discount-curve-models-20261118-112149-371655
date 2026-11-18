"""Tests for Experiment 2: the hypothesis space A as affine combinations of
flat discount curves (Section 3.3, Lemma 3.4) and the KRR estimate lying in A.
"""

from __future__ import annotations

import numpy as np

from src.config import EXP2_A, EXP2_B, EXP2_C, EXP2_GRID, EXP2_LAMBDA
from src.hypothesis_space import (
    basis_functions,
    basis_matrix_rank,
    decompose_into_flat_curves,
    eigenvalues,
    flat_curve_combination,
)
from src.krr import krr_fit


def test_basis_functions_full_rank() -> None:
    # Keep the evaluation grid on the same scale as exp/exp2_hypothesis_space.py's
    # X_EVAL (0.05 to 3.0): for large x the exponentials e^{lambda_j x} span many
    # orders of magnitude and numpy's rank tolerance mistakes the resulting
    # ill-conditioning for rank deficiency, which is a property of float64
    # evaluation, not of the linear independence claimed by the paper.
    x_eval = np.linspace(0.05, 3.0, 50)
    rank = basis_matrix_rank(x_eval, EXP2_GRID, EXP2_A, EXP2_B)
    assert rank == len(EXP2_GRID)


def test_affine_combination_reproduced_by_basis_functions() -> None:
    """Any f(x) = sum_j z_j e^{lambda_j x} with sum z_j = 1 should equal
    p_0(x) + sum_j Y_j phi_j(x) for some Y_j (Lemma 3.4)."""
    lambdas = eigenvalues(EXP2_GRID, EXP2_A, EXP2_B)
    rng = np.random.default_rng(0)
    z = rng.uniform(0.1, 1.0, size=len(lambdas))
    z /= z.sum()

    x_eval = np.linspace(0.05, 3.0, 40)
    target = flat_curve_combination(z, lambdas, x_eval)

    phi = basis_functions(x_eval, EXP2_GRID, EXP2_A, EXP2_B)
    p0 = np.exp(EXP2_B * x_eval)
    # Solve for Y such that target = p0 + phi @ Y (least squares, exact since phi is full rank)
    Y, *_ = np.linalg.lstsq(phi, target - p0, rcond=None)
    reconstructed = p0 + phi @ Y
    assert np.max(np.abs(reconstructed - target)) < 1e-8


def test_krr_estimate_satisfies_p0_equals_one() -> None:
    rng = np.random.default_rng(1)
    N = len(EXP2_GRID)
    m = 20
    cashflows = rng.uniform(0, 10, size=(m, N))
    prices = cashflows @ np.exp(EXP2_B * EXP2_GRID) * 0.95 + rng.normal(0, 0.01, size=m)
    weights = np.ones(m)

    fit = krr_fit(EXP2_GRID, cashflows, prices, weights, EXP2_A, EXP2_B, EXP2_C, EXP2_LAMBDA)
    p_at_zero = fit["hat_p"](0.0)
    assert abs(p_at_zero - 1.0) < 1e-8


def test_krr_estimate_lies_in_flat_curve_span() -> None:
    """hat_p should be expressible as sum_j z_j e^{lambda_j x} with sum z_j=1."""
    rng = np.random.default_rng(2)
    N = len(EXP2_GRID)
    m = 20
    cashflows = rng.uniform(0, 10, size=(m, N))
    prices = cashflows @ np.exp(EXP2_B * EXP2_GRID) * 0.95
    weights = np.ones(m)

    fit = krr_fit(EXP2_GRID, cashflows, prices, weights, EXP2_A, EXP2_B, EXP2_C, EXP2_LAMBDA)
    lambdas = eigenvalues(EXP2_GRID, EXP2_A, EXP2_B)
    x_fit = np.linspace(0.05, 3.0, 60)
    curve_values = fit["hat_p"](x_fit)
    z_hat = decompose_into_flat_curves(curve_values, x_fit, lambdas)

    assert abs(z_hat.sum() - 1.0) < 1e-6
    reconstructed = flat_curve_combination(z_hat, lambdas, x_fit)
    assert np.max(np.abs(reconstructed - curve_values)) < 1e-6


def test_run_experiment_returns_expected_keys() -> None:
    from exp.exp2_hypothesis_space import run_experiment

    results = run_experiment()
    assert isinstance(results, dict)
    assert results["basis_rank"] == len(EXP2_GRID)
    assert abs(results["hat_p0"] - 1.0) < 1e-8
    assert abs(results["z_hat_sum"] - 1.0) < 1e-6
