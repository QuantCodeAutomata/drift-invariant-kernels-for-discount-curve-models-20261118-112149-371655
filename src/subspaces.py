"""Drift-invariant subspaces of V (Proposition 3.5), the failure of
PCA-based dimension reduction (Corollary 3.6), and pivoted Cholesky
sub-grid selection (Section 3.3, p.8, citing Filipovic and Schneider 2025).

Custom implementation (paper Eq. 3.7, Proposition 3.5, Corollary 3.6): the
drift-invariance tests are specific to this paper. The pivoted Cholesky
decomposition itself is a standard numerical-linear-algebra algorithm
(greedy maximum-diagonal-pivot Cholesky for symmetric positive
semidefinite matrices); no higher-level quantitative-finance library
(skfolio, pyportfolioopt, scikit-learn) exposes a pivoted Cholesky
routine, so it is implemented directly from its textbook definition.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from src.drift import alpha_map
from src.kernels import flat_prior_curve, schur_complement_k0_analytic


def _safe_norm(v: np.ndarray) -> float:
    """Euclidean norm that avoids overflow when entries of ``v`` are as large
    as exp(400) (~5e173): squaring such an entry directly (~2.7e347) exceeds
    the float64 range (~1.8e308), even though the entry itself does not. The
    vector is rescaled by its largest magnitude before squaring, then the
    scale is multiplied back in.
    """
    scale = np.max(np.abs(v))
    if scale == 0 or not np.isfinite(scale):
        return float(np.linalg.norm(v))
    return float(scale * np.linalg.norm(v / scale))


def kernel_matrix(x_grid: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """K_0 = (k_0(x_i, x_j))_{i,j=1}^N, equation below (3.5)."""
    x_grid = np.asarray(x_grid, dtype=float)
    return schur_complement_k0_analytic(x_grid[:, None], x_grid[None, :], a, b, c)


def subgrid_basis_function(y: np.ndarray, x_j: float, a: float, b: float, c: float) -> np.ndarray:
    """k_0(x_j, y) as a function of y, for a single fixed grid point x_j."""
    return schur_complement_k0_analytic(np.full_like(np.asarray(y, dtype=float), x_j), y, a, b, c)


def subgrid_invariance_residual(
    x_grid: np.ndarray,
    subgrid_idx: Sequence[int],
    a: float,
    b: float,
    c: float,
    y_eval: np.ndarray,
    h: float = 1e-5,
) -> float:
    """Residual of alpha(W) subset W for W = span{k_0(x_j,.) : j in subgrid_idx}
    (Proposition 3.5).

    By (3.7), each basis function g_j = k_0(x_j,.) is individually a (closed
    form) eigenfunction of alpha with eigenvalue lambda_j = a*x_j + b, so
    membership in W is tested pointwise, alpha(g_j)(y) == lambda_j * g_j(y)
    for every y, rather than by a least-squares projection onto the other
    members of the sub-grid. The latter would require solving a design
    matrix whose columns k_0(x_i, y_eval) grow at exponentially different
    rates over x_i in the sub-grid, which is spuriously ill-conditioned at
    large y even though each column already satisfies the eigenfunction
    identity on its own; the pointwise test avoids this conditioning
    artifact and is the direct numerical content of equation (3.7).
    """
    x_grid = np.asarray(x_grid, dtype=float)
    p0_func = lambda y: flat_prior_curve(y, b)
    max_residual = 0.0
    for j in subgrid_idx:
        g_j = lambda y, j=j: subgrid_basis_function(y, x_grid[j], a, b, c)
        g_vals = g_j(y_eval)
        alpha_vals = alpha_map(g_j, y_eval, p0_func, h)
        lambda_j = a * x_grid[j] + b
        residual = _safe_norm(alpha_vals - lambda_j * g_vals)
        scale = _safe_norm(alpha_vals) + 1e-12
        max_residual = max(max_residual, residual / scale)
    return float(max_residual)


def pca_subspace_residual(
    x_grid: np.ndarray, a: float, b: float, c: float, d: int, y_eval: np.ndarray, h: float = 1e-5
) -> float:
    """Residual of alpha(W) subset W for W = span of the top-d eigenvectors
    of K_0 (Corollary 3.6). Returns the relative least-squares residual of
    projecting alpha(f_l) onto span{f_1,...,f_d}; Corollary 3.6 predicts this
    residual is strictly positive (drift-invariance fails) whenever 1<=d<N.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    N = len(x_grid)
    assert 1 <= d < N
    K0 = kernel_matrix(x_grid, a, b, c)
    eigvals, eigvecs = np.linalg.eigh(K0)  # ascending order
    top_idx = np.argsort(eigvals)[::-1][:d]
    U = eigvecs[:, top_idx]  # (N, d), columns u_1,...,u_d

    def f_l(y, coeffs):
        y = np.asarray(y, dtype=float)
        vals = np.stack(
            [subgrid_basis_function(y, x_grid[j], a, b, c) for j in range(N)], axis=1
        )  # (len(y), N)
        return vals @ coeffs

    p0_func = lambda y: flat_prior_curve(y, b)
    f_vals = np.stack([f_l(y_eval, U[:, l]) for l in range(d)], axis=1)  # (len(y_eval), d)
    # Column-normalize before solving: f_1,...,f_d combine basis functions k_0(x_j,.)
    # that grow at very different exponential rates over the grid, which otherwise
    # ill-conditions the least-squares projection without changing the subspace it
    # tests (dividing a column by its norm leaves span{f_1,...,f_d} unchanged).
    col_norms = np.array([_safe_norm(f_vals[:, j]) for j in range(f_vals.shape[1])])
    col_norms = np.where(col_norms < 1e-300, 1.0, col_norms)
    f_vals_scaled = f_vals / col_norms

    max_residual = 0.0
    for l in range(d):
        func_l = lambda y, l=l: f_l(y, U[:, l])
        alpha_vals = alpha_map(func_l, y_eval, p0_func, h)
        # alpha_vals and the columns of f_vals_scaled span magnitudes up to
        # ~exp(400); rescale the whole least-squares system by its largest
        # entry so that neither lstsq's internal SVD nor the residual norm
        # below forms a squared value beyond the float64 range.
        system_scale = max(np.max(np.abs(f_vals_scaled)), np.max(np.abs(alpha_vals)), 1e-300)
        coeffs, *_ = np.linalg.lstsq(
            f_vals_scaled / system_scale, alpha_vals / system_scale, rcond=None
        )
        residual = _safe_norm(alpha_vals - f_vals_scaled @ coeffs)
        scale = _safe_norm(alpha_vals) + 1e-12
        max_residual = max(max_residual, residual / scale)
    return float(max_residual)


def pivoted_cholesky(K: np.ndarray, d: int, tol: float = 1e-12) -> list[int]:
    """Greedy maximum-diagonal-pivot (pivoted) Cholesky decomposition of a
    symmetric positive semidefinite matrix K, returning the first ``d``
    selected pivot indices (Filipovic and Schneider 2025, cited in Section
    3.3, p.8, for sub-grid selection; the specific algorithm of that
    reference is not detailed in the present paper, see the uncertainties
    of experiment 3). This is the textbook pivoted Cholesky algorithm: at
    each step, select the remaining index with the largest Schur-complement
    diagonal entry and eliminate it.
    """
    K = np.array(K, dtype=float)
    N = K.shape[0]
    assert K.shape == (N, N)
    remaining = list(range(N))
    residual = K.copy()
    pivots: list[int] = []
    for _ in range(min(d, N)):
        diag = np.diag(residual)
        local_idx = int(np.argmax(diag))
        if diag[local_idx] <= tol:
            break
        global_idx = remaining[local_idx]
        pivots.append(global_idx)
        pivot_col = residual[:, local_idx]
        residual = residual - np.outer(pivot_col, pivot_col) / diag[local_idx]
        remaining.pop(local_idx)
        residual = np.delete(np.delete(residual, local_idx, axis=0), local_idx, axis=1)
    return pivots
