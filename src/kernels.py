"""Kernel family of Theorem 3.3(1) and its Schur complement (Section 3.1-3.3),
plus a Sobolev-type smoothness kernel used as a non-drift-invariant comparison.

All formulas are vectorized with numpy broadcasting; each function accepts
scalars or arrays of the same shape for x and y.
"""

from __future__ import annotations

import numpy as np


def flat_prior_curve(x: np.ndarray, b: float) -> np.ndarray:
    """Prior curve p_0(x) = e^{bx} (Lemma 3.1: p_0 = k(0, .))."""
    return np.exp(b * np.asarray(x, dtype=float))


def drift_invariant_kernel(
    x: np.ndarray, y: np.ndarray, a: float, b: float, c: float
) -> np.ndarray:
    """Kernel family k(x,y) = e^{b(x+y)}(c e^{axy} + 1 - c), Theorem 3.3(1).

    Requires a, c > 0 for k to be a valid (positive semidefinite) kernel,
    Lemma B.2. Satisfies the normalization k(0,0) = 1 (Section 3).
    """
    assert a > 0 and c > 0, "Theorem 3.3(1) requires a > 0 and c > 0"
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.exp(b * (x + y)) * (c * np.exp(a * x * y) + 1.0 - c)


def schur_complement_k0_analytic(
    x: np.ndarray, y: np.ndarray, a: float, b: float, c: float
) -> np.ndarray:
    """Analytical Schur complement k_0(x,y) = c e^{b(x+y)}(e^{axy} - 1),
    equation (3.6), derived from k_0 = k(x,y) - k(x,0)k(0,y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return c * np.exp(b * (x + y)) * (np.exp(a * x * y) - 1.0)


def schur_complement_numeric(kernel_func, x: np.ndarray, y: np.ndarray, **kernel_kwargs):
    """Generic Schur complement k_0(x,y) = k(x,y) - k(x,0) k(0,y) (3.1),
    evaluated directly from any kernel function k(x, y, **kwargs).

    Used to cross-check the analytical formula above and to compute k_0 for
    kernels without a closed form (e.g. the Sobolev-type proxy kernel).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k_xy = kernel_func(x, y, **kernel_kwargs)
    k_x0 = kernel_func(x, np.zeros_like(x), **kernel_kwargs)
    k_0y = kernel_func(np.zeros_like(y), y, **kernel_kwargs)
    return k_xy - k_x0 * k_0y


def gaussian_rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian radial-basis-function kernel k(x,y) = exp(-(x-y)^2/(2 sigma^2)),
    used as a proxy for the Sobolev-type smoothness-rewarding kernels of
    Filipovic et al. (2025); Camenzind and Filipovic (2024), which the present
    paper does not reproduce in closed form (Section 3.3, p.7): "Kernels that
    reward smoothness of the curve ... are not real analytic across the
    diagonal." The Gaussian kernel shares this property (k(0,0)=1 holds, but
    k is not real analytic in the sense required by Theorem 3.3) and is used
    here only to demonstrate the failure of drift-invariance for a
    smoothness-rewarding kernel (see Deviations from the paper).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert sigma > 0
    return np.exp(-((x - y) ** 2) / (2.0 * sigma ** 2))
