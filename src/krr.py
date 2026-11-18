"""Kernel ridge regression estimator of the discount curve, equations
(3.3)-(3.4). Custom implementation (paper Eq. 3.3-3.4, Filipovic et al. 2025,
Theorem A.1 as cited in Section 3.1): this closed-form KRR solution is
specific to the discount-curve setting of this paper and is not provided by
a general-purpose library such as pyportfolioopt or skfolio.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.kernels import flat_prior_curve, schur_complement_k0_analytic


def krr_fit(
    x_grid: np.ndarray,
    cashflows: np.ndarray,
    prices: np.ndarray,
    weights: np.ndarray,
    a: float,
    b: float,
    c: float,
    lam: float,
) -> dict:
    """Fit the KRR discount curve estimator (3.4).

    Parameters
    ----------
    x_grid : (N,) cash flow dates x_1 < ... < x_N.
    cashflows : (m, N) matrix C = (c_ij), cash flow of instrument i at x_j.
    prices : (m,) vector of observed prices P.
    weights : (m,) vector of weights w_i > 0.
    a, b, c : kernel parameters of Theorem 3.3(1).
    lam : regularization parameter lambda > 0.

    Returns
    -------
    dict with keys:
      beta       : (N,) coefficients of (3.4), hat_h = sum_j beta_j k_0(x_j, .)
      Y          : (N,) coefficients Y = c * beta (Section 4, eq. above (4.1))
      K0         : (N, N) kernel matrix (k_0(x_i, x_j))
      hat_p      : callable, the estimated discount curve y -> p_0(y) + sum_j beta_j k_0(x_j, y)
      pricing_error : float, sum_i w_i (P_i - sum_j c_ij hat_p(x_j))^2
    """
    x_grid = np.asarray(x_grid, dtype=float)
    cashflows = np.asarray(cashflows, dtype=float)
    prices = np.asarray(prices, dtype=float)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), prices.shape).copy()
    assert lam > 0 and a > 0 and c > 0
    assert cashflows.shape[1] == len(x_grid)
    assert cashflows.shape[0] == len(prices) == len(weights)

    N = len(x_grid)
    K0 = schur_complement_k0_analytic(x_grid[:, None], x_grid[None, :], a, b, c)
    C = cashflows
    Lambda = np.diag(lam / weights)
    p0_x = flat_prior_curve(x_grid, b)

    M = C @ K0 @ C.T + Lambda
    rhs = prices - C @ p0_x
    # M = C K0 C^T + Lambda is symmetric positive definite: K0 is PSD (it is a
    # Schur complement of the PSD kernel k, Lemma/Theorem 3.3) so C K0 C^T is
    # PSD, and Lambda = diag(lambda/w_i) has a strictly positive diagonal;
    # hence every eigenvalue of M is at least min_i(lambda/w_i) > 0. A direct
    # solve (Cholesky via np.linalg.solve) is used rather than a least-squares
    # pseudo-inverse, so that the regularization strength lambda is always
    # respected rather than being swamped by an SVD truncation cutoff.
    alpha_coeffs = np.linalg.solve(M, rhs)
    beta = C.T @ alpha_coeffs
    Y = c * beta  # Section 4: "The KRR estimate (3.4) corresponds to Y = c*beta."

    def hat_p(y) -> np.ndarray:
        """Estimated discount curve hat_p(y) = p_0(y) + sum_j beta_j k_0(x_j, y)."""
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        k0_vals = schur_complement_k0_analytic(x_grid[:, None], y_arr[None, :], a, b, c)  # (N, len(y))
        result = flat_prior_curve(y_arr, b) + k0_vals.T @ beta
        return float(result[0]) if np.isscalar(y) or np.asarray(y).ndim == 0 else result

    fitted_prices = C @ hat_p(x_grid)
    pricing_error = float(np.sum(weights * (prices - fitted_prices) ** 2))

    return {
        "beta": beta,
        "Y": Y,
        "K0": K0,
        "hat_p": hat_p,
        "pricing_error": pricing_error,
        "p0_x": p0_x,
    }
