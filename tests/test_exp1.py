"""Tests for Experiment 1: drift-invariance of the kernel family
k(x,y) = e^{b(x+y)}(c e^{axy} + 1 - c), Section 3.2-3.3, Theorem 3.3.
"""

from __future__ import annotations

import numpy as np

from src.config import EXP1_FD_STEP, EXP1_GRIDS, EXP1_PARAM_SETS, EXP1_Y_EVAL
from src.drift import alpha_map, eigen_ratio
from src.kernels import (
    drift_invariant_kernel,
    flat_prior_curve,
    gaussian_rbf_kernel,
    schur_complement_k0_analytic,
    schur_complement_numeric,
)


def test_schur_complement_analytic_matches_numeric() -> None:
    a, b, c = 0.7, -0.01, 1.5
    grid = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
    x_grid, y_grid = np.meshgrid(grid, grid, indexing="ij")
    analytic = schur_complement_k0_analytic(x_grid, y_grid, a, b, c)
    numeric = schur_complement_numeric(drift_invariant_kernel, x_grid, y_grid, a=a, b=b, c=c)
    # Use a relative tolerance: for large a*x*y, c*e^{axy} reaches magnitudes
    # where adding/subtracting O(1) terms before the final multiplication
    # loses float64 precision through catastrophic cancellation, so a pure
    # absolute tolerance is not meaningful at the largest grid point (5, 5).
    abs_diff = np.abs(analytic - numeric)
    relative_diff = abs_diff / np.maximum(np.abs(analytic), 1.0)
    assert np.max(relative_diff) < 1e-6


def test_k00_equals_one() -> None:
    for a, b, c in [(1.0, 0.0, 1.0), (0.5, -0.02, 2.0), (2.0, 0.03, 0.5)]:
        k00 = drift_invariant_kernel(np.array(0.0), np.array(0.0), a, b, c)
        assert abs(float(k00) - 1.0) < 1e-10


def test_eigenfunction_property_holds_for_drift_invariant_kernel() -> None:
    param_set = EXP1_PARAM_SETS[0]
    a, b, c = param_set["a"], param_set["b"], param_set["c"]
    grid = EXP1_GRIDS[0]
    y_eval = EXP1_Y_EVAL[:4]
    p0_func = lambda y: flat_prior_curve(y, b)

    for x_j in grid:
        g_j = lambda y, x_j=x_j: schur_complement_k0_analytic(x_j, y, a, b, c)
        alpha_values = alpha_map(g_j, y_eval, p0_func=p0_func, h=EXP1_FD_STEP)
        g_values = g_j(y_eval)
        ratio = eigen_ratio(alpha_values, g_values)
        expected = a * float(x_j) + b
        valid = ~np.isnan(ratio)
        assert np.any(valid)
        assert np.max(np.abs(ratio[valid] - expected)) < 1e-3


def test_gaussian_kernel_ratio_is_not_constant() -> None:
    sigma = 2.0
    grid = EXP1_GRIDS[0]
    p0_func = lambda y: gaussian_rbf_kernel(np.zeros_like(np.asarray(y, dtype=float)), y, sigma)

    found_non_constant = False
    for x_j in grid:
        g_j = lambda y, x_j=x_j: schur_complement_numeric(gaussian_rbf_kernel, x_j, y, sigma=sigma)
        alpha_values = alpha_map(g_j, EXP1_Y_EVAL, p0_func=p0_func, h=EXP1_FD_STEP)
        g_values = g_j(EXP1_Y_EVAL)
        ratio = eigen_ratio(alpha_values, g_values)
        mean = np.nanmean(ratio)
        std = np.nanstd(ratio)
        if mean != 0.0 and std / abs(mean) > 0.02:
            found_non_constant = True
    assert found_non_constant


def test_run_experiment_returns_expected_keys() -> None:
    from exp.exp1_drift_invariance import run_experiment

    results = run_experiment()
    assert isinstance(results, dict)
    assert "drift_invariant_combinations" in results
    assert "gaussian_non_invariance" in results
    assert len(results["drift_invariant_combinations"]) == 4
    assert "per_grid_point" in results["gaussian_non_invariance"]
