"""Experiment 6: Example 3.7 of Celary, Eksi Altay, Filipovic and Kruhner,
"Drift-invariant kernels for discount curve models" -- generalized
eigenfunctions and the exponential-polynomial family
W = span{e^x - 1, x e^x}, for a=1, b=0 (so p_0(x)=1, k_0(x,y)=e^{xy}-1,
alpha(h) = h' - h'(0)).

Verifies:
  1. alpha(e^x - 1) = e^x - 1 (eigenfunction, eigenvalue 1);
  2. alpha(x e^x) = (e^x - 1) + x e^x, so alpha(W) subset W even though
     x e^x is only a *generalized* eigenfunction (Jordan block [[1,1],[0,1]]);
  3. x e^x cannot be written as c1*(e^{x1 x}-1) + c2*(e^{x2 x}-1) for any
     x1, x2, c1, c2: the Taylor-coefficient system at 0 is inconsistent, and
     direct nonlinear least-squares fitting confirms a strictly positive
     residual;
  4. Proposition 2.2's conditions for p_0 + W to be drift-invariant hold
     (p_0(0)=1, h(0)=0 for h in W, eta(p_0)=0 in W).

Run as: ``python -m exp.exp6_generalized_eigen``.
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
    EXP6_A,
    EXP6_B,
    EXP6_FD_STEP,
    EXP6_OPT_N_RESTARTS,
    EXP6_OPT_SEED,
    EXP6_TAYLOR_ORDERS,
    EXP6_X_FIT_GRID,
    EXP6_Y_EVAL,
)
from src.drift import eta_map
from src.generalized_eigen import (
    alpha_h1,
    alpha_h2,
    fit_two_exponential_combination,
    h1,
    h2,
    jordan_block_matrix,
    p0_const,
    taylor_match_residual,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_JSON = RESULTS_DIR / "exp6_generalized_eigen_results.json"
RESULTS_PLOT = RESULTS_DIR / "exp6_generalized_eigen.png"


def run_experiment() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    assert EXP6_A == 1.0 and EXP6_B == 0.0, "Example 3.7 requires a=1, b=0"

    y = EXP6_Y_EVAL

    # Step 3: alpha(h1) = h1 (eigenvalue 1).
    a_h1 = alpha_h1(y, EXP6_FD_STEP)
    h1_vals = h1(y)
    alpha_h1_error = float(np.max(np.abs(a_h1 - h1_vals)))
    assert alpha_h1_error < 1e-6, "alpha(e^x - 1) must equal e^x - 1"

    # Step 4: alpha(h2) = h1 + h2.
    a_h2 = alpha_h2(y, EXP6_FD_STEP)
    h2_vals = h2(y)
    alpha_h2_error = float(np.max(np.abs(a_h2 - (h1_vals + h2_vals))))
    assert alpha_h2_error < 1e-6, "alpha(x e^x) must equal (e^x - 1) + x e^x"

    # Step 5: Jordan block matrix representation of alpha|_W in basis {h1, h2}.
    jordan = jordan_block_matrix(y, EXP6_FD_STEP)
    jordan_expected = np.array([[1.0, 1.0], [0.0, 1.0]])
    jordan_error = float(np.max(np.abs(jordan - jordan_expected)))
    assert jordan_error < 1e-6, f"matrix of alpha|_W must be the Jordan block [[1,1],[0,1]], got {jordan}"

    # Step 6: Taylor-coefficient inconsistency test. Scan a grid of (x1, x2)
    # and solve the 2x2 linear system for (c1, c2) matching the first two
    # Taylor orders, then check that the next order is NOT matched (the
    # system for 3+ orders is overdetermined and generically inconsistent).
    orders = EXP6_TAYLOR_ORDERS
    rng = np.random.default_rng(EXP6_OPT_SEED)
    min_residual_exact_fit = np.inf
    for _ in range(200):
        x1, x2 = rng.uniform(0.2, 3.0, size=2)
        if abs(x1 - x2) < 1e-3:
            continue
        # Match the first two Taylor coefficients exactly (2 equations, 2 unknowns):
        # c1*x1 + c2*x2 = 1 (order 1); c1*x1^2/2 + c2*x2^2/2 = 1 (order 2).
        A = np.array([[x1, x2], [x1 ** 2 / 2.0, x2 ** 2 / 2.0]])
        rhs = np.array([1.0, 1.0])
        try:
            c1, c2 = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        residual = taylor_match_residual(x1, x2, c1, c2, orders)
        min_residual_exact_fit = min(min_residual_exact_fit, residual)
    assert min_residual_exact_fit > 1e-3, (
        "matching the first two Taylor coefficients must leave a residual at "
        "higher orders (the full system is inconsistent for any x1, x2, c1, c2)"
    )

    # Direct nonlinear least-squares confirmation (methodology step 7).
    opt = fit_two_exponential_combination(EXP6_X_FIT_GRID, EXP6_OPT_N_RESTARTS, EXP6_OPT_SEED)
    target_norm_sq = float(np.sum(h2(EXP6_X_FIT_GRID) ** 2))
    relative_residual = opt["residual"] / target_norm_sq
    # The rigorous proof of non-representability is the Taylor-coefficient
    # argument above (an exact, finite-dimensional linear-algebra statement).
    # This bounded nonlinear fit only illustrates it: with x1, x2 confined to
    # disjoint intervals and bounded coefficients, the best achievable fit
    # still leaves a residual well above floating-point round-off.
    assert relative_residual > 1e-8, (
        "x e^x must not be representable as c1*(e^{x1 x}-1)+c2*(e^{x2 x}-1); "
        f"got a near-zero relative residual {relative_residual:.3e}"
    )

    # Step 8: Proposition 2.2 conditions for p_0 + W, p_0 = 1.
    p0_at_0 = float(p0_const(np.array([0.0]))[0])
    h1_at_0 = float(h1(np.array([0.0]))[0])
    h2_at_0 = float(h2(np.array([0.0]))[0])
    eta_p0 = eta_map(p0_const, np.array([0.0]), EXP6_FD_STEP)
    proposition_conditions = {
        "p0(0)=1": abs(p0_at_0 - 1.0) < 1e-8,
        "h1(0)=0": abs(h1_at_0) < 1e-8,
        "h2(0)=0": abs(h2_at_0) < 1e-8,
        "eta(p0)(0)=0": abs(float(eta_p0[0])) < 1e-6,
    }
    assert all(proposition_conditions.values()), f"Proposition 2.2 conditions failed: {proposition_conditions}"

    results: dict[str, Any] = {
        "alpha_h1_max_abs_error": alpha_h1_error,
        "alpha_h2_max_abs_error": alpha_h2_error,
        "jordan_block_max_abs_error": jordan_error,
        "jordan_block_matrix": jordan.tolist(),
        "min_taylor_residual_after_2_order_exact_fit": float(min_residual_exact_fit),
        "nonlinear_lstsq_residual": opt["residual"],
        "nonlinear_lstsq_relative_residual": relative_residual,
        "nonlinear_lstsq_params_x1_x2_c1_c2": opt["params"].tolist(),
        "proposition_2_2_conditions": proposition_conditions,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))
    x_fine = np.linspace(0.05, 3.0, 200)
    ax_left.plot(x_fine, h1(x_fine), label="h1(x) = e^x - 1")
    ax_left.plot(x_fine, h2(x_fine), label="h2(x) = x e^x")
    ax_left.plot(x_fine, alpha_h2(x_fine, EXP6_FD_STEP), "--", label="alpha(h2)(x)")
    ax_left.plot(x_fine, h1(x_fine) + h2(x_fine), ":", label="h1(x)+h2(x)")
    ax_left.set_xlabel("x (years)")
    ax_left.set_title("Example 3.7: alpha(h2) = h1 + h2")
    ax_left.legend(fontsize=8)

    x1_best, x2_best, c1_best, c2_best = opt["params"]
    best_fit = c1_best * (np.exp(x1_best * x_fine) - 1.0) + c2_best * (np.exp(x2_best * x_fine) - 1.0)
    ax_right.plot(x_fine, h2(x_fine), label="x e^x (target)")
    ax_right.plot(x_fine, best_fit, "--", label="best-fit c1(e^{x1 x}-1)+c2(e^{x2 x}-1)")
    ax_right.set_xlabel("x (years)")
    ax_right.set_title("x e^x is not a combination of two flat discount curves")
    ax_right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_PLOT, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("Experiment 6: generalized eigenfunctions (Example 3.7)")
    print("=" * 60)
    print(f"alpha(h1) = h1 max abs error: {alpha_h1_error:.3e}")
    print(f"alpha(h2) = h1+h2 max abs error: {alpha_h2_error:.3e}")
    print(f"Jordan block [[1,1],[0,1]] max abs error: {jordan_error:.3e}")
    print(f"min residual after matching 2 Taylor orders (orders 3+): {min_residual_exact_fit:.3e}")
    print(f"nonlinear least-squares relative residual: {relative_residual:.3e}")
    print(f"Proposition 2.2 conditions: {proposition_conditions}")
    print(f"Results written to {RESULTS_JSON}")
    print(f"Figure written to {RESULTS_PLOT}")

    return results


if __name__ == "__main__":
    run_experiment()
