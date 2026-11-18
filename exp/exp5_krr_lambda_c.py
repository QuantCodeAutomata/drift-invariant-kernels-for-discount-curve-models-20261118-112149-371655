"""Experiment 5: kernel ridge regression estimation with the drift-invariant
kernel on synthetic bond data (Celary, Eksi Altay, Filipovic and Kruhner,
"Drift-invariant kernels for discount curve models", Sections 3.1-3.2,
equations (3.3)-(3.4)).

Verifies:
  1. the KRR estimate hat_p satisfies hat_p(0) = 1 for arbitrary synthetic
     instrument cash flows and prices;
  2. the estimate lies in A = p_0 + span{k_0(x_1,.),...,k_0(x_N,.)}, i.e. it
     is representable as an affine combination of flat discount curves
     (Lemma 3.4);
  3. the estimator depends on c only through the ratio lambda/c: replacing
     k_0 by c*k_0 rescales K_0 by c and beta by 1/c, leaving beta_j*k_0(x_j,y)
     invariant (methodology step 11, implementation_notes);
  4. the pricing error sum_i w_i (P_i - sum_j c_ij hat_p(x_j))^2 decreases
     monotonically as lambda/c decreases (less regularization).

Run as: ``python -m exp.exp5_krr_lambda_c``.
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
    EXP5_A,
    EXP5_B,
    EXP5_C_ALT,
    EXP5_C_BASE,
    EXP5_GRID,
    EXP5_LAMBDAS_FOR_C1,
    EXP5_LAMBDAS_FOR_C_ALT,
    EXP5_N_INSTRUMENTS,
    EXP5_SEED,
    EXP5_TRUE_SHORT_RATE,
    EXP5_TRUE_SLOPE,
    EXP5_WEIGHTS,
)
from src.hypothesis_space import decompose_into_flat_curves, eigenvalues, flat_curve_combination
from src.krr import krr_fit
from data.generate_synthetic import generate_instruments, true_discount_curve_smooth

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_JSON = RESULTS_DIR / "exp5_krr_lambda_c_results.json"
RESULTS_PLOT = RESULTS_DIR / "exp5_krr_curves.png"

X_EVAL = np.linspace(0.05, 10.0, 200)  # fine grid for the function-space checks,
# kept within the bulk of the x<=10 region of the N=11 grid (up to x=30) for the
# same conditioning reason documented in exp2_hypothesis_space.py (the
# flat-curve basis e^{lambda_j x} spans many orders of magnitude at large x).


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    N = len(EXP5_GRID)
    lambdas = eigenvalues(EXP5_GRID, EXP5_A, EXP5_B)

    true_curve = true_discount_curve_smooth(EXP5_TRUE_SHORT_RATE, EXP5_TRUE_SLOPE)
    instruments = generate_instruments(EXP5_GRID, true_curve, EXP5_N_INSTRUMENTS, EXP5_SEED)
    weights = np.full(EXP5_N_INSTRUMENTS, EXP5_WEIGHTS)

    def fit_and_check(c: float, lam: float) -> dict[str, Any]:
        fit = krr_fit(
            EXP5_GRID, instruments["cashflows"], instruments["prices"], weights,
            EXP5_A, EXP5_B, c, lam,
        )
        hat_p0 = float(fit["hat_p"](np.array([0.0]))[0])
        hat_p_fine = fit["hat_p"](X_EVAL)
        z_hat = decompose_into_flat_curves(hat_p_fine, X_EVAL, lambdas)
        hat_p_from_flat = flat_curve_combination(z_hat, lambdas, X_EVAL)
        decomposition_error = float(np.max(np.abs(hat_p_fine - hat_p_from_flat)))
        return {
            "c": c, "lam": lam, "lam_over_c": lam / c,
            "hat_p0": hat_p0,
            "z_hat_sum": float(z_hat.sum()),
            "flat_curve_decomposition_max_abs_error": decomposition_error,
            "pricing_error": fit["pricing_error"],
            "hat_p_fine": hat_p_fine,
        }

    # Step 11: varying lambda/c at c=1 and at c=2 with matching ratios.
    runs_c1 = [fit_and_check(EXP5_C_BASE, lam) for lam in EXP5_LAMBDAS_FOR_C1]
    runs_c2 = [fit_and_check(EXP5_C_ALT, lam) for lam in EXP5_LAMBDAS_FOR_C_ALT]

    for r in runs_c1 + runs_c2:
        assert abs(r["hat_p0"] - 1.0) < 1e-8, "hat_p(0) must equal 1 for every lambda/c"
        assert r["flat_curve_decomposition_max_abs_error"] < 1e-5, (
            "the KRR estimate must lie in A for every lambda/c"
        )

    # Step 11 (invariance in c): curves with the same lambda/c ratio must match.
    c_invariance_max_abs_error = float(max(
        np.max(np.abs(r1["hat_p_fine"] - r2["hat_p_fine"]))
        for r1, r2 in zip(runs_c1, runs_c2)
    ))
    assert c_invariance_max_abs_error < 1e-6, (
        "the estimator must depend on c only through the ratio lambda/c"
    )

    # Step 12: pricing error should increase with lambda/c (monotonicity, up
    # to numerical ties at very small lambda/c where the system is already
    # near-interpolating).
    pricing_errors_c1 = [r["pricing_error"] for r in runs_c1]
    monotone_nondecreasing = all(
        pricing_errors_c1[i] <= pricing_errors_c1[i + 1] + 1e-6
        for i in range(len(pricing_errors_c1) - 1)
    )

    results: dict[str, Any] = {
        "N": N,
        "lambdas": lambdas.tolist(),
        "runs_c1": [
            {k: v for k, v in r.items() if k != "hat_p_fine"} for r in runs_c1
        ],
        "runs_c2": [
            {k: v for k, v in r.items() if k != "hat_p_fine"} for r in runs_c2
        ],
        "c_invariance_max_abs_error": c_invariance_max_abs_error,
        "pricing_error_monotone_nondecreasing_in_lambda_over_c": monotone_nondecreasing,
        "n_instruments": EXP5_N_INSTRUMENTS,
        "instrument_types": instruments["instrument_types"],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))
    for r in runs_c1:
        ax_left.plot(X_EVAL, r["hat_p_fine"], label=f"lambda/c={r['lam_over_c']:g}")
    ax_left.plot(X_EVAL, true_curve(X_EVAL), "k--", label="true curve", linewidth=2)
    ax_left.set_xlabel("x (years)")
    ax_left.set_ylabel("discount factor")
    ax_left.set_title("Effect of lambda/c on the KRR estimate (c=1)")
    ax_left.legend(fontsize=7)

    ax_right.plot(EXP5_LAMBDAS_FOR_C1, pricing_errors_c1, "o-", label="c=1")
    ax_right.plot(
        [lam / EXP5_C_ALT * EXP5_C_BASE for lam in EXP5_LAMBDAS_FOR_C_ALT],
        [r["pricing_error"] for r in runs_c2], "x--", label="c=2 (lambda rescaled)"
    )
    ax_right.set_xscale("log")
    ax_right.set_xlabel("lambda/c")
    ax_right.set_ylabel("weighted pricing error")
    ax_right.set_title("Pricing error increases with regularization lambda/c")
    ax_right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_PLOT, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("Experiment 5: KRR estimation with the drift-invariant kernel")
    print("=" * 60)
    for r in runs_c1:
        print(
            f"c=1, lambda={r['lam']:<8g} hat_p(0)={r['hat_p0']:.10f} "
            f"decomp_err={r['flat_curve_decomposition_max_abs_error']:.2e} "
            f"pricing_err={r['pricing_error']:.4f}"
        )
    print(f"max |curve(c=1) - curve(c=2, matched lambda/c)| = {c_invariance_max_abs_error:.3e}")
    print(f"pricing error monotone nondecreasing in lambda/c: {monotone_nondecreasing}")
    print(f"Results written to {RESULTS_JSON}")
    print(f"Figure written to {RESULTS_PLOT}")

    return results


if __name__ == "__main__":
    run_experiment()
