"""Structure of the hypothesis space A = p_0 + V (Section 3.3): the
eigenvalues lambda_j = a x_j + b (3.8), the normalized basis functions
phi_j(x) = k_0(x_j, x)/c (3.9), and the flat-discount-curve representation
of A (Lemma 3.4).

Custom implementation (paper Eq. 3.6-3.9, Lemma 3.4): specific to this paper.
"""

from __future__ import annotations

import numpy as np


def eigenvalues(x_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    """lambda_j = a*x_j + b for j=0,...,N with x_0 = 0, so lambda_0 = b (3.8).

    Returns an array of length N+1, index 0 is lambda_0 = b.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    x_full = np.concatenate(([0.0], x_grid))
    return a * x_full + b


def basis_functions(x_eval: np.ndarray, x_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    """phi_j(x) = e^{b x_j} (e^{lambda_j x} - e^{b x}), j=1,...,N (3.9).

    Returns a matrix of shape (len(x_eval), N) with columns phi_1,...,phi_N.
    """
    x_eval = np.asarray(x_eval, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    lambdas = a * x_grid + b  # lambda_1,...,lambda_N
    # broadcasting: rows = x_eval, columns = grid index j
    exp_lx = np.exp(np.outer(x_eval, lambdas))           # e^{lambda_j x}
    exp_bx = np.exp(b * x_eval)[:, None]                  # e^{b x}, broadcast over columns
    prefactor = np.exp(b * x_grid)[None, :]               # e^{b x_j}
    return prefactor * (exp_lx - exp_bx)


def flat_curve_combination(z: np.ndarray, lambdas: np.ndarray, x: np.ndarray) -> np.ndarray:
    """f(x) = sum_{j=0}^N z_j e^{lambda_j x}, Lemma 3.4, requires sum(z) = 1."""
    z = np.asarray(z, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    x = np.asarray(x, dtype=float)
    assert abs(z.sum() - 1.0) < 1e-8, "Lemma 3.4 requires sum_j z_j = 1"
    return np.exp(np.outer(x, lambdas)) @ z


def decompose_into_flat_curves(
    curve_values: np.ndarray, x_fit: np.ndarray, lambdas: np.ndarray
) -> np.ndarray:
    """Solve for z in R^{N+1} such that sum_j z_j e^{lambda_j x} matches
    ``curve_values`` on the grid ``x_fit``, under the constraint sum z_j = 1
    (Lemma 3.4).

    The constraint z_0 = 1 - sum_{j=1}^N z_j is substituted directly rather
    than enforced through a penalised row, which would mix terms of very
    different magnitude (e^{lambda_j x} ranges over several orders of
    magnitude across j) into one ill-conditioned least-squares problem. After
    substitution the regression is

        curve_values - e^{lambda_0 x} = sum_{j=1}^N z_j (e^{lambda_j x} - e^{lambda_0 x}),

    an ordinary least-squares fit in z_1,...,z_N with the same well-scaled
    design as the basis functions phi_j (3.9); z_0 is recovered from the
    constraint afterwards.
    """
    x_fit = np.asarray(x_fit, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    curve_values = np.asarray(curve_values, dtype=float)
    lambda0, lambdas_rest = lambdas[0], lambdas[1:]
    baseline = np.exp(lambda0 * x_fit)
    design = np.exp(np.outer(x_fit, lambdas_rest)) - baseline[:, None]
    z_rest, *_ = np.linalg.lstsq(design, curve_values - baseline, rcond=None)
    z0 = 1.0 - z_rest.sum()
    return np.concatenate([[z0], z_rest])


def basis_matrix_rank(x_eval: np.ndarray, x_grid: np.ndarray, a: float, b: float) -> int:
    """Rank of the matrix Phi = (phi_j(x_i)) used to test linear independence
    of the basis functions phi_1,...,phi_N (methodology step 5 of exp_2).
    """
    phi = basis_functions(x_eval, x_grid, a, b)
    return int(np.linalg.matrix_rank(phi))
