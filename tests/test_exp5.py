"""Unit tests for the KRR estimator with the drift-invariant kernel on
synthetic bond data (Experiment 5, paper Sections 3.1-3.2).

Uses small synthetic fixtures (not market data); no live API calls.
"""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import generate_instruments, true_discount_curve_smooth
from src.config import EXP5_A, EXP5_B, EXP5_TRUE_SHORT_RATE, EXP5_TRUE_SLOPE
from src.kernels import flat_prior_curve
from src.krr import krr_fit


def _small_fixture(n_instruments: int = 10, seed: int = 0):
    x_grid = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    true_curve = true_discount_curve_smooth(EXP5_TRUE_SHORT_RATE, EXP5_TRUE_SLOPE)
    instruments = generate_instruments(x_grid, true_curve, n_instruments, seed)
    return x_grid, instruments


def test_hat_p_zero_equals_one():
    """hat_p(0) = 1 exactly, for any prices/weights (paper Section 3.2)."""
    x_grid, instruments = _small_fixture()
    prices = instruments["prices"]
    cashflows = instruments["cashflows"]
    weights = np.ones_like(prices)
    fit = krr_fit(x_grid, cashflows, prices, weights, EXP5_A, EXP5_B, 1.0, 0.1)
    assert abs(fit["hat_p"](np.array([0.0]))[0] - 1.0) < 1e-10


def test_c_ratio_invariance():
    """Only the ratio lambda/c determines beta_j * k_0(x_j, y) (paper Section 3.2,
    since replacing k_0 by c*k_0 scales K0 by c and beta by 1/c)."""
    x_grid, instruments = _small_fixture()
    prices = instruments["prices"]
    cashflows = instruments["cashflows"]
    weights = np.ones_like(prices)
    lam_over_c = 0.05
    y_eval = np.linspace(0.1, 9.0, 20)

    fit_c1 = krr_fit(x_grid, cashflows, prices, weights, EXP5_A, EXP5_B, 1.0, lam_over_c * 1.0)
    fit_c2 = krr_fit(x_grid, cashflows, prices, weights, EXP5_A, EXP5_B, 2.0, lam_over_c * 2.0)

    np.testing.assert_allclose(fit_c1["hat_p"](y_eval), fit_c2["hat_p"](y_eval), atol=1e-8)


def test_pricing_error_increases_with_regularization():
    """More regularization (larger lambda/c) should not decrease the weighted
    pricing error (paper Section 3.2 expected outcomes)."""
    x_grid, instruments = _small_fixture(n_instruments=15, seed=1)
    prices = instruments["prices"]
    cashflows = instruments["cashflows"]
    weights = np.ones_like(prices)

    errors = []
    for lam in (1e-4, 1e-2, 1.0, 1e2):
        fit = krr_fit(x_grid, cashflows, prices, weights, EXP5_A, EXP5_B, 1.0, lam)
        errors.append(fit["pricing_error"])
    assert errors[0] <= errors[-1] + 1e-6
    # monotone non-decreasing within numerical tolerance
    assert all(errors[i] <= errors[i + 1] + 1e-6 for i in range(len(errors) - 1))


def test_curve_lies_in_affine_space():
    """hat_p(y) = p_0(y) + sum_j beta_j k_0(x_j, y) must match the affine
    combination of flat discount curves (paper Section 3.1, Theorem 3.3)."""
    x_grid, instruments = _small_fixture()
    prices = instruments["prices"]
    cashflows = instruments["cashflows"]
    weights = np.ones_like(prices)
    fit = krr_fit(x_grid, cashflows, prices, weights, EXP5_A, EXP5_B, 1.0, 0.1)

    y_eval = np.linspace(0.1, 9.0, 15)
    direct = fit["hat_p"](y_eval)
    p0 = flat_prior_curve(y_eval, EXP5_B)
    from src.kernels import schur_complement_k0_analytic

    K0_y = schur_complement_k0_analytic(x_grid[:, None], y_eval[None, :], EXP5_A, EXP5_B, 1.0)
    reconstructed = p0 + fit["beta"] @ K0_y
    np.testing.assert_allclose(direct, reconstructed, atol=1e-10)


def test_input_validation():
    """krr_fit must reject a non-positive regularization or kernel parameter."""
    x_grid, instruments = _small_fixture()
    prices = instruments["prices"]
    cashflows = instruments["cashflows"]
    weights = np.ones_like(prices)
    try:
        krr_fit(x_grid, cashflows, prices, weights, EXP5_A, EXP5_B, 1.0, lam=-1.0)
        raised = False
    except AssertionError:
        raised = True
    assert raised
