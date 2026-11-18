"""Example 3.7: the drift-invariant space W = span{e^x - 1, x e^x} for
a=1, b=0 (so p_0 = 1, k_0(x,y) = e^{xy} - 1), which is not of the sub-grid
form of Proposition 3.5 for any grid.

Custom implementation (paper Example 3.7): specific worked example.
"""

from __future__ import annotations

import numpy as np

from src.drift import alpha_map


def h1(x: np.ndarray) -> np.ndarray:
    """h1(x) = e^x - 1."""
    return np.exp(np.asarray(x, dtype=float)) - 1.0


def h2(x: np.ndarray) -> np.ndarray:
    """h2(x) = x * e^x (generalized eigenfunction of alpha, Example 3.7)."""
    x = np.asarray(x, dtype=float)
    return x * np.exp(x)


def p0_const(x: np.ndarray) -> np.ndarray:
    """p_0(x) = 1 for a=1, b=0."""
    return np.ones_like(np.asarray(x, dtype=float))


def alpha_h1(y: np.ndarray, fd_step: float = 1e-5) -> np.ndarray:
    """Numerically evaluated alpha(h1)(y), expected to equal h1(y) (eigenvalue 1)."""
    return alpha_map(h1, y, p0_const, fd_step)


def alpha_h2(y: np.ndarray, fd_step: float = 1e-5) -> np.ndarray:
    """Numerically evaluated alpha(h2)(y), expected to equal h1(y) + h2(y)."""
    return alpha_map(h2, y, p0_const, fd_step)


def jordan_block_matrix(y_eval: np.ndarray, fd_step: float = 1e-5) -> np.ndarray:
    """Matrix representation of alpha restricted to W = span{h1, h2} in the
    basis {h1, h2}, obtained by least-squares projection of alpha(h1),
    alpha(h2) onto {h1, h2}. Expected to equal the Jordan block [[1,1],[0,1]].
    """
    basis = np.stack([h1(y_eval), h2(y_eval)], axis=1)  # (len(y_eval), 2)
    a_h1 = alpha_h1(y_eval, fd_step)
    a_h2 = alpha_h2(y_eval, fd_step)
    col1, *_ = np.linalg.lstsq(basis, a_h1, rcond=None)
    col2, *_ = np.linalg.lstsq(basis, a_h2, rcond=None)
    return np.column_stack([col1, col2])


def taylor_coefficients_h2(orders: list[int]) -> np.ndarray:
    """Taylor coefficients of x*e^x = x + x^2 + x^3/2! + x^4/3! + ... at 0,
    i.e. coefficient of x^n is 1/(n-1)!.
    """
    from math import factorial

    return np.array([1.0 / factorial(n - 1) for n in orders])


def taylor_coefficients_exp_minus_one(x_scale: float, orders: list[int]) -> np.ndarray:
    """Taylor coefficients of e^{x_scale * x} - 1 at 0: coefficient of x^n is
    x_scale^n / n!.
    """
    from math import factorial

    return np.array([x_scale ** n / factorial(n) for n in orders])


def taylor_match_residual(x1: float, x2: float, c1: float, c2: float, orders: list[int]) -> float:
    """Residual of matching the Taylor coefficients of x*e^x (orders>=1) with
    those of c1*(e^{x1 x}-1) + c2*(e^{x2 x}-1) (methodology step 6). A
    least-squares (rather than exact) comparison, since the system is shown
    to be inconsistent for any x1, x2, c1, c2.
    """
    target = taylor_coefficients_h2(orders)
    combo = c1 * taylor_coefficients_exp_minus_one(x1, orders) + c2 * taylor_coefficients_exp_minus_one(
        x2, orders
    )
    return float(np.linalg.norm(target - combo))


def fit_two_exponential_combination(x_grid: np.ndarray, n_restarts: int, seed: int) -> dict:
    """Numerically minimize ||x e^x - (c1*(e^{x1 x}-1) + c2*(e^{x2 x}-1))||^2
    over (x1, x2, c1, c2) on ``x_grid`` (methodology step 7), using multiple
    random restarts of scipy.optimize.minimize.

    x1, x2 are confined to disjoint intervals (genuinely distinct cash-flow
    dates, x1 < 1.4 < x2) and |c1|, |c2| are bounded. Without these bounds an
    unconstrained optimizer can let x1 -> x2 with c1 -> -c2 -> infinity, which
    is exactly the confluent limit by which x*e^x arises as a generalized
    eigenfunction (Example 3.7): it then approximates the target arbitrarily
    closely without ever representing it exactly for a fixed pair x1 != x2,
    so an unconstrained search would report a spurious near-zero residual.
    Example 3.7 predicts a strictly positive minimum under these bounds.
    """
    from scipy.optimize import minimize

    target = h2(x_grid)
    rng = np.random.default_rng(seed)
    bounds = [(0.1, 1.4), (1.4, 4.0), (-20.0, 20.0), (-20.0, 20.0)]

    def objective(theta):
        x1, x2, c1, c2 = theta
        model = c1 * (np.exp(x1 * x_grid) - 1.0) + c2 * (np.exp(x2 * x_grid) - 1.0)
        return float(np.sum((target - model) ** 2))

    best = None
    for _ in range(n_restarts):
        theta0 = [
            rng.uniform(0.1, 1.4),
            rng.uniform(1.4, 4.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
        ]
        res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
                        options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res
    return {"residual": float(best.fun), "params": best.x, "n_grid_points": len(x_grid)}
