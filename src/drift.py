"""The no-arbitrage drift map eta and the linear map alpha of Section 2
(Definition 2.1, equations (2.1)-(2.2), Proposition 2.2), and generic
finite-difference tools to evaluate them for an arbitrary function h in
C^1(R_+) without assuming a closed-form derivative.

Custom implementation (paper Eq. 2.1-2.2): these maps are specific to this
paper and are not provided by any standard numerical library.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def central_derivative(
    func: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """Central finite-difference derivative f'(y) = (f(y+h)-f(y-h))/(2h).

    All functions handled in this repository (flat curves, Schur-complement
    basis functions, eigenfunctions of alpha) extend analytically from R_+ to
    all of R, so the central difference at y=0 is well defined and matches
    the one-sided derivative f'(0) required by (2.1)-(2.2).
    """
    y = np.asarray(y, dtype=float)
    return (func(y + h) - func(y - h)) / (2.0 * h)


def eta_map(
    func: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """No-arbitrage drift eta(f) = f' - f'(0) f, equation (2.1)."""
    y = np.asarray(y, dtype=float)
    f_prime = central_derivative(func, y, h)
    # Evaluate f'(0) with a 1-element 1-d array (not a 0-d scalar array) so
    # that callables relying on np.stack(..., axis=1) over their input shape
    # (e.g. PCA eigenfunctions in src/subspaces.py) also accept y=0.
    f_prime_0 = float(central_derivative(func, np.array([0.0]), h).reshape(-1)[0])
    return f_prime - f_prime_0 * func(y)


def alpha_map(
    func: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    p0_func: Callable[[np.ndarray], np.ndarray],
    h: float = 1e-5,
) -> np.ndarray:
    """Linear map alpha(h) = h' - h'(0) p_0, equation (2.2), evaluated on
    functions h in the direction space V (h(0) = 0), with p_0 the base point.
    """
    y = np.asarray(y, dtype=float)
    f_prime = central_derivative(func, y, h)
    # See the comment in eta_map: use a 1-d array of length 1, not a 0-d array.
    f_prime_0 = float(central_derivative(func, np.array([0.0]), h).reshape(-1)[0])
    return f_prime - f_prime_0 * p0_func(y)


def eigen_ratio(
    alpha_values: np.ndarray, func_values: np.ndarray, tol_zero: float = 1e-8
) -> np.ndarray:
    """Pointwise ratio alpha(g)(y) / g(y) used to test the eigenfunction
    property alpha(g) = lambda * g (3.7). Points where |g(y)| < tol_zero are
    excluded to avoid division by (near) zero, as noted in the paper's
    implementation notes for experiment 1.
    """
    alpha_values = np.asarray(alpha_values, dtype=float)
    func_values = np.asarray(func_values, dtype=float)
    mask = np.abs(func_values) > tol_zero
    ratio = np.full_like(func_values, np.nan)
    ratio[mask] = alpha_values[mask] / func_values[mask]
    return ratio
