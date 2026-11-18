"""Experiment 2: structure of the hypothesis space A = p_0 + V as affine
combinations of flat discount curves (Celary, Eksi Altay, Filipovic and
Kruhner, "Drift-invariant kernels for discount curve models", Section 3.3,
Lemma 3.4, equations (3.8)-(3.9)).

For the drift-invariant kernel k(x,y) = e^{b(x+y)}(c e^{axy} + 1 - c):
  1. the basis functions phi_j(x) = e^{b x_j}(e^{lambda_j x} - e^{b x}),
     lambda_j = a x_j + b, are linearly independent (rank(Phi) = N);
  2. A = {x -> sum_{j=0}^N z_j e^{lambda_j x} : sum_j z_j = 1} (Lemma 3.4);
  3. the KRR estimate (3.4) lies in A for arbitrary synthetic instrument
     cash flows and prices, i.e. it can be re-expressed exactly as an affine
     combination of the flat discount curves e^{lambda_j x}.

Run as: ``python -m exp.exp2_hypothesis_space``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP2_A,
    EXP2_B,
    EXP2_C,
    EXP2_GRID,
    EXP2_LAMBDA,
    EXP2_N_INSTRUMENTS,
    EXP2_SEED,
)
from src.hypothesis_space import (
    basis_functions,
    basis_matrix_rank,
    decompose_into_flat_curves,
    eigenvalues,
    flat_curve_combination,
)
from src.krr import krr_fit
from data.generate_synthetic import generate_instruments, true_discount_curve_smooth

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_JSON = RESULTS_DIR / "exp2_hypothesis_space_results.json"
RESULTS_PLOT = RESULTS_DIR / "exp2_flat_curve_decomposition.png"

X_EVAL = np.linspace(0.05, 3.0, 200)  # fine grid for the function-space checks
# Kept within the bulk of the grid (x up to 10): e^{lambda_j x} spans about 15
# orders of magnitude across j already at x=3 (lambda_N ~ 5), which is enough
# to resolve N=5 linearly independent directions in double precision; evaluating
# up to x=15 makes the largest-lambda column overwhelm the others by more than
# 1e30, collapsing the numerical rank well below the tolerance of
# numpy.linalg.matrix_rank regardless of true linear independence.


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    N = len(EXP2_GRID)
    lambdas = eigenvalues(EXP2_GRID, EXP2_A, EXP2_B)  # length N+1, lambda_0 = b

    # Step 5: linear independence of phi_1,...,phi_N (methodology step 5).
    rank = basis_matrix_rank(X_EVAL, EXP2_GRID, EXP2_A, EXP2_B)
    assert rank == N, f"basis functions must be linearly independent: rank={rank}, N={N}"

    # Step 6: verify any f in A = p_0 + V can be written as an affine
    # combination of flat discount curves, sum_j z_j = 1 (Lemma 3.4).
    rng = np.random.default_rng(EXP2_SEED)
    z_random = rng.dirichlet(np.ones(N + 1))  # nonnegative weights summing to 1
    f_values = flat_curve_combination(z_random, lambdas, X_EVAL)
    basis_vals = basis_functions(X_EVAL, EXP2_GRID, EXP2_A, EXP2_B)  # (len(X_EVAL), N)
    p0_vals = np.exp(EXP2_B * X_EVAL)
    Y_from_z = z_random[1:] / np.exp(EXP2_B * EXP2_GRID)  # Lemma 3.4: z_j = Y_j e^{-b x_j} for j>=1
    f_reconstructed = p0_vals + basis_vals @ Y_from_z
    affine_combination_max_abs_error = float(np.max(np.abs(f_values - f_reconstructed)))
    assert affine_combination_max_abs_error < 1e-8, (
        "f = p_0 + sum_j Y_j phi_j must equal the flat-curve combination sum_j z_j e^{lambda_j x}"
    )

    # Step 7-9: synthetic instruments and the KRR estimate (3.4). Instruments
    # are priced from a smooth target yield curve (see
    # data/generate_synthetic.py:true_discount_curve_smooth) rather than from
    # an arbitrary affine combination of flat curves, because a combination
    # with nonzero weight on the largest eigenvalue lambda_N (here 4.97 at
    # x=10) blows up to economically meaningless prices; the property under
    # test here is that the *estimate* hat_p lies in A, not the shape of the
    # data-generating curve.
    true_curve = true_discount_curve_smooth(short_rate=0.03, slope=0.0015)
    instruments = generate_instruments(EXP2_GRID, true_curve, EXP2_N_INSTRUMENTS, EXP2_SEED)
    weights = np.ones(EXP2_N_INSTRUMENTS)

    fit = krr_fit(
        EXP2_GRID,
        instruments["cashflows"],
        instruments["prices"],
        weights,
        EXP2_A,
        EXP2_B,
        EXP2_C,
        EXP2_LAMBDA,
    )
    hat_p_grid = fit["hat_p"](EXP2_GRID)
    hat_p0 = float(fit["hat_p"](np.array([0.0]))[0])
    assert abs(hat_p0 - 1.0) < 1e-8, "the KRR estimate must satisfy hat_p(0) = 1"

    # Step 9 (end): re-express hat_p in the flat-discount-curve basis
    # (methodology step 9 of experiment 2 / step 9 of experiment 5) and
    # verify the two representations agree on a fine grid.
    hat_p_fine = fit["hat_p"](X_EVAL)
    z_hat = decompose_into_flat_curves(hat_p_fine, X_EVAL, lambdas)
    z_hat_sum = float(z_hat.sum())
    assert abs(z_hat_sum - 1.0) < 1e-6, f"decomposed weights must sum to 1, got {z_hat_sum}"
    hat_p_from_flat_curves = flat_curve_combination(z_hat, lambdas, X_EVAL)
    flat_curve_decomposition_max_abs_error = float(np.max(np.abs(hat_p_fine - hat_p_from_flat_curves)))
    assert flat_curve_decomposition_max_abs_error < 1e-6, (
        "the KRR estimate must be representable as an affine combination of flat discount curves"
    )

    results: dict[str, Any] = {
        "basis_rank": rank,
        "N": N,
        "lambdas": lambdas.tolist(),
        "affine_combination_max_abs_error": affine_combination_max_abs_error,
        "hat_p0": hat_p0,
        "z_hat_sum": z_hat_sum,
        "flat_curve_decomposition_max_abs_error": flat_curve_decomposition_max_abs_error,
        "pricing_error": fit["pricing_error"],
        "n_instruments": EXP2_N_INSTRUMENTS,
        "instrument_types": instruments["instrument_types"],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_left.plot(X_EVAL, f_values, label="direct affine combination", linewidth=2)
    ax_left.plot(X_EVAL, f_reconstructed, "--", label="p_0 + sum Y_j phi_j", linewidth=2)
    ax_left.set_xlabel("x (years)")
    ax_left.set_ylabel("f(x)")
    ax_left.set_title("Lemma 3.4: A = affine combinations of flat discount curves")
    ax_left.legend(fontsize=8)

    ax_right.plot(X_EVAL, hat_p_fine, label="KRR estimate hat_p", linewidth=2)
    ax_right.plot(X_EVAL, hat_p_from_flat_curves, "--", label="flat-curve reconstruction", linewidth=2)
    ax_right.scatter(EXP2_GRID, hat_p_grid, color="black", zorder=5, label="grid points x_1..x_N")
    ax_right.set_xlabel("x (years)")
    ax_right.set_ylabel("discount factor")
    ax_right.set_title("KRR estimate lies in A (Section 3.2, eq. 3.4)")
    ax_right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_PLOT, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("Experiment 2: structure of the hypothesis space")
    print("=" * 60)
    print(f"rank(Phi) = {rank} (N = {N}): basis functions linearly independent = {rank == N}")
    print(f"affine-combination max abs error: {affine_combination_max_abs_error:.3e}")
    print(f"KRR estimate hat_p(0) = {hat_p0:.10f} (expected 1)")
    print(f"flat-curve decomposition: sum(z_hat) = {z_hat_sum:.8f} (expected 1)")
    print(f"flat-curve decomposition max abs error: {flat_curve_decomposition_max_abs_error:.3e}")
    print(f"Results written to {RESULTS_JSON}")
    print(f"Figure written to {RESULTS_PLOT}")

    return results


if __name__ == "__main__":
    run_experiment()
