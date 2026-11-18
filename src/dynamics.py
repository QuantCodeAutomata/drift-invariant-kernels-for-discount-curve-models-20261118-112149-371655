"""Risk-neutral coefficient dynamics in the hypothesis space A (Section 4):
the short rate (4.3), the derivative identity (4.4), the drift equivalence
of Theorem 4.1, and the augmented linear system of Remark 4.2.

Custom implementation (paper Eq. 4.1-4.6, Theorem 4.1, Remark 4.2): this is
the paper's own term-structure dynamics, not a feature of any general SDE or
term-structure library (the `arch`, `hmmlearn` and similar libraries in this
project do not cover affine/linear-rational term-structure simulation).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from src.hypothesis_space import basis_functions, eigenvalues


def build_G_matrix(x_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    """Matrix G in R^{(N+1)x(N+1)} of Remark 4.2:
    G_00 = b, G_0j = a*x_j*e^{b*x_j}, G_jj = lambda_j for j=1,...,N, else 0.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    N = len(x_grid)
    G = np.zeros((N + 1, N + 1))
    G[0, 0] = b
    lambdas = a * x_grid + b  # lambda_1,...,lambda_N
    for j in range(1, N + 1):
        G[0, j] = a * x_grid[j - 1] * np.exp(b * x_grid[j - 1])
        G[j, j] = lambdas[j - 1]
    return G


def simulate_Ytilde(
    x_grid: np.ndarray,
    a: float,
    b: float,
    ytilde0_init: float,
    nu: np.ndarray,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> dict:
    """Simulate the discounted coefficients Ytilde = (Ytilde^0,...,Ytilde^N).

    Ytilde^j for j=1,...,N follows the geometric Brownian motion
    d Ytilde_t^j = lambda_j Ytilde_t^j dt + Ytilde_t^j nu_j^T dW_t  (4.6, GBM case),
    simulated by Euler-Maruyama. Ytilde^0 = 1/B_t follows the linear ODE
    d Ytilde_t^0 = (b Ytilde_t^0 + a sum_j x_j e^{b x_j} Ytilde_t^j) dt  (Remark 4.2),
    integrated along each simulated path by the forward Euler scheme.

    Returns a dict with Ytilde_paths (n_steps+1, n_paths, N+1), the time grid,
    and the money-market account B = 1/Ytilde^0 and coefficients Y = Ytilde/Ytilde^0.
    """
    rng = np.random.default_rng(seed)
    x_grid = np.asarray(x_grid, dtype=float)
    N = len(x_grid)
    lambdas = a * x_grid + b
    d_factors = nu.shape[1]
    c_coef = a * x_grid * np.exp(b * x_grid)  # G_{0,j} of Remark 4.2
    nu_sq = np.sum(nu**2, axis=1)  # |nu_j|^2, j=1,...,N

    Ytilde = np.zeros((n_steps + 1, n_paths, N + 1))
    Ytilde[0, :, 1:] = ytilde0_init
    Ytilde[0, :, 0] = 1.0  # Ytilde_0^0 = 1/B_0 = 1

    sqrt_dt = np.sqrt(dt)
    # Each step is simulated with the explicit solution of Remark 4.2 rather
    # than forward Euler: Ytilde^j (j=1,...,N) is an exact geometric Brownian
    # motion step (eliminating the drift-discretization bias of Euler for the
    # fast-growing factors with lambda_j*dt not negligible), and Ytilde^0 is
    # obtained by exactly integrating its linear ODE over [t,t+dt], assuming
    # each Ytilde^j grows at its own rate lambda_j within the step (the
    # deterministic skeleton of the explicit solution
    # Ytilde_t = exp(G t) Ytilde_0 + integral exp(G(t-s)) sigma_tilde_s dW_s).
    for t in range(n_steps):
        dW = rng.normal(size=(n_paths, d_factors)) * sqrt_dt
        Yt_j = Ytilde[t, :, 1:]  # (n_paths, N)
        log_increment = (lambdas[None, :] - 0.5 * nu_sq[None, :]) * dt + dW @ nu.T
        Ytilde_j_next = Yt_j * np.exp(log_increment)

        Yt_0 = Ytilde[t, :, 0]
        exp_b_dt = np.exp(b * dt)
        exp_lam_dt = np.exp(lambdas * dt)  # (N,)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(
                np.abs(lambdas - b) > 1e-10,
                (exp_lam_dt - exp_b_dt) / (lambdas - b),
                dt * exp_b_dt,  # limit lambda_j -> b
            )
        Ytilde_0_next = Yt_0 * exp_b_dt + (Yt_j * (c_coef * weight)[None, :]).sum(axis=1)

        Ytilde[t + 1, :, 0] = Ytilde_0_next
        Ytilde[t + 1, :, 1:] = Ytilde_j_next

    B = 1.0 / Ytilde[:, :, 0]
    Y = Ytilde[:, :, 1:] * Ytilde[:, :, [0]] ** -1  # Y^j = Ytilde^j / Ytilde^0

    return {
        "Ytilde": Ytilde,
        "B": B,
        "Y": Y,
        "time_grid": np.arange(n_steps + 1) * dt,
        "lambdas": lambdas,
    }


def short_rate(Y: np.ndarray, x_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    """r_t = -b - a sum_j x_j e^{b x_j} Y_t^j, equation (4.3)."""
    x_grid = np.asarray(x_grid, dtype=float)
    weight = x_grid * np.exp(b * x_grid)
    return -b - a * (Y * weight) @ np.ones(len(x_grid)) if Y.ndim == 1 else -b - a * np.einsum(
        "...j,j->...", Y, weight
    )


def bond_price(Y: np.ndarray, tau: np.ndarray, x_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    """P(t,T) = p_t(T-t) = e^{b tau} + sum_j Y_t^j phi_j(tau), tau = T - t, eq. (4.1)."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    phi = basis_functions(tau, x_grid, a, b)  # (len(tau), N)
    flat = np.exp(b * tau)
    # Y has shape (..., N); result shape (..., len(tau))
    return flat + np.einsum("...j,tj->...t", Y, phi)
