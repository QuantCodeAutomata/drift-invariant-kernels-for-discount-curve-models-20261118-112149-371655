"""Experiment 4: risk-neutral coefficient dynamics (Section 4 of Celary,
Eksi Altay, Filipovic and Kruhner, "Drift-invariant kernels for discount
curve models").

The hypothesis space's coefficient process Y has risk-neutral drift
mu_t^j = (lambda_j + r_t) Y_t^j, which is quadratic in the state because the
short rate r_t = -b - a sum_j x_j e^{b x_j} Y_t^j (eq. 4.3) is itself linear
in Y. Theorem 4.1 shows that after discounting by the money-market account
B_t, the resulting coefficients Ytilde = Y / B instead satisfy the LINEAR
dynamics

    d Ytilde_t^j = lambda_j Ytilde_t^j dt + Ytilde_t^j nu_j^T dW_t,  j = 1,...,N

(the "linear-rational term structure" phenomenon, eq. 4.6), while
Ytilde^0 = 1/B_t follows the linear ODE of Remark 4.2 driven by the same
state. This module empirically verifies:

  1. the linear discounted drift lambda_j of Theorem 4.1 (eq. 4.6);
  2. approximate consistency between the simulated B and the discrete
     compounding of the short rate r_t (eq. 4.3);
  3. the discounted-bond-price martingale property implied by risk-neutral
     pricing, P(0,T) = E[P(t,T)/B_t] with P(T,T) = 1 (eq. 4.1);
  4. the structure of the augmented matrix G of Remark 4.2, whose diagonal
     collects the eigenvalues lambda_0,...,lambda_N (eq. 3.8).

Run as: ``python -m exp.exp4_dynamics``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP4_A,
    EXP4_B,
    EXP4_BOND_MATURITY_T,
    EXP4_D_FACTORS,
    EXP4_DT,
    EXP4_GRID,
    EXP4_N_PATHS_RUN,
    EXP4_N_STEPS,
    EXP4_NU_SCALE,
    EXP4_NU_SEED,
    EXP4_SIM_SEED,
    EXP4_YTILDE0_INIT,
)
from src.dynamics import bond_price, build_G_matrix, short_rate, simulate_Ytilde
from src.hypothesis_space import eigenvalues

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RESULTS_JSON_PATH = os.path.join(RESULTS_DIR, "exp4_dynamics_results.json")
RESULTS_PLOT_PATH = os.path.join(RESULTS_DIR, "exp4_dynamics_plots.png")


def _fit_log_linear_slope(time_grid: np.ndarray, mean_path: np.ndarray) -> float:
    """Fit the slope of log(mean_path) against time_grid via degree-1 polyfit."""
    log_mean = np.log(mean_path)
    slope, _intercept = np.polyfit(time_grid, log_mean, deg=1)
    return float(slope)


def run_experiment() -> dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Volatility vectors nu_j, j=1,...,N, each with EXP4_D_FACTORS factor loadings.
    rng = np.random.default_rng(EXP4_NU_SEED)
    nu = EXP4_NU_SCALE * rng.standard_normal((len(EXP4_GRID), EXP4_D_FACTORS))

    # 2. Euler-Maruyama simulation of the discounted coefficients Ytilde.
    sim = simulate_Ytilde(
        EXP4_GRID,
        EXP4_A,
        EXP4_B,
        EXP4_YTILDE0_INIT,
        nu,
        EXP4_DT,
        EXP4_N_STEPS,
        EXP4_N_PATHS_RUN,
        EXP4_SIM_SEED,
    )
    time_grid = sim["time_grid"]
    lambdas = sim["lambdas"]
    N = len(EXP4_GRID)

    # 3. Verify the linear discounted drift of Theorem 4.1: E[Ytilde^j_t] ~
    #    Ytilde^j_0 * exp(lambda_j t), i.e. log of the cross-sectional mean is
    #    (approximately) linear in time with slope lambda_j.
    fitted_slopes = np.empty(N)
    mean_paths = np.empty((len(time_grid), N))
    for j in range(N):
        mean_paths[:, j] = sim["Ytilde"][:, :, j + 1].mean(axis=1)
        fitted_slopes[j] = _fit_log_linear_slope(time_grid, mean_paths[:, j])
    slope_deviation = np.abs(fitted_slopes - lambdas)
    for j in range(N):
        assert slope_deviation[j] < 0.1, (
            f"factor {j}: fitted slope {fitted_slopes[j]:.5f} deviates from "
            f"theoretical lambda_j {lambdas[j]:.5f} by {slope_deviation[j]:.5f} (tolerance 0.1)"
        )

    # 4. Short-rate / money-market account consistency (approximate check only:
    #    the simulation integrates the exact linear ODE for Ytilde^0, not a
    #    discrete compounding of the instantaneous short rate; the two schemes
    #    agree only up to Euler discretization error).
    r = short_rate(sim["Y"], EXP4_GRID, EXP4_A, EXP4_B)  # shape (n_steps+1, n_paths)
    log_B_from_r = np.cumsum(r[:-1, :] * EXP4_DT, axis=0)  # log(B_from_short_rate[t+1])
    log_B_from_r = np.vstack([np.zeros((1, r.shape[1])), log_B_from_r])
    log_B_sim = np.log(sim["B"])
    short_rate_log_B_deviation = float(np.mean(np.abs(log_B_from_r[-1, :] - log_B_sim[-1, :])))
    assert short_rate_log_B_deviation < 0.2, (
        f"log(B) consistency deviation {short_rate_log_B_deviation:.5f} exceeds the loose "
        "sanity bound of 0.2"
    )

    # 5. Discounted-bond-price martingale test for a bond of fixed maturity
    #    T = EXP4_BOND_MATURITY_T (measured from t=0).
    tau = EXP4_BOND_MATURITY_T - time_grid
    valid = tau >= 0
    tau = tau[valid]
    t_T = min(int(round(EXP4_BOND_MATURITY_T / EXP4_DT)), EXP4_N_STEPS)
    tau_T = EXP4_BOND_MATURITY_T - time_grid[t_T]

    P0 = float(bond_price(sim["Y"][0, 0], tau[0], EXP4_GRID, EXP4_A, EXP4_B)[0])
    P_T = bond_price(sim["Y"][t_T], tau_T, EXP4_GRID, EXP4_A, EXP4_B)[:, 0]  # (n_paths,)
    P_T_over_B_T = P_T / sim["B"][t_T]
    mc_average = float(P_T_over_B_T.mean())
    mc_std_error = float(P_T_over_B_T.std(ddof=1) / np.sqrt(len(P_T_over_B_T)))
    relative_difference = float((mc_average - P0) / P0)
    within_3_se = bool(abs(mc_average - P0) < 3 * mc_std_error)
    if not within_3_se:
        print(
            f"[exp4] WARNING: martingale test deviation {mc_average - P0:.6f} exceeds "
            f"3 Monte Carlo standard errors ({3 * mc_std_error:.6f}); reporting, not failing."
        )

    # 6. Structure of the augmented matrix G (Remark 4.2).
    G = build_G_matrix(EXP4_GRID, EXP4_A, EXP4_B)
    eigvals = eigenvalues(EXP4_GRID, EXP4_A, EXP4_B)
    assert G[0, 0] == EXP4_B
    g_diagonal_matches = bool(np.allclose(np.diag(G), eigvals))
    assert g_diagonal_matches

    # 7. Save a JSON summary.
    results: dict[str, Any] = {
        "lambda_fit": {
            "fitted_slopes": fitted_slopes.tolist(),
            "theoretical_lambdas": lambdas.tolist(),
            "abs_deviation": slope_deviation.tolist(),
        },
        "short_rate_consistency": {
            "mean_abs_log_B_deviation_at_final_step": short_rate_log_B_deviation,
        },
        "martingale_test": {
            "bond_maturity_T": EXP4_BOND_MATURITY_T,
            "t_T_step": t_T,
            "tau_T": float(tau_T),
            "P0": P0,
            "mc_average_P_T_over_B_T": mc_average,
            "relative_difference": relative_difference,
            "mc_standard_error": mc_std_error,
            "within_3_standard_errors": within_3_se,
        },
        "G_matrix_check": {
            "G_00": float(G[0, 0]),
            "expected_b": EXP4_B,
            "diagonal_matches_eigenvalues": g_diagonal_matches,
        },
    }
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 8. Plot: left = cross-sectional mean Ytilde^j paths (log scale) vs theory;
    #    right = histogram of terminal P_T/B_T values.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    for j in range(N):
        ax.plot(time_grid, mean_paths[:, j], color=colors[j], label=f"mean Ytilde^{j + 1} (MC)")
        theory = EXP4_YTILDE0_INIT * np.exp(lambdas[j] * time_grid)
        ax.plot(time_grid, theory, "--", color=colors[j], alpha=0.7, label=f"theory lambda_{j + 1}")
    ax.set_yscale("log")
    ax.set_xlabel("time t (years)")
    ax.set_ylabel("cross-sectional mean of Ytilde^j_t (log scale)")
    ax.set_title("Linear discounted drift (Theorem 4.1)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ax.hist(P_T_over_B_T, bins=60, color="tab:blue", alpha=0.75, label="P(t_T,T)/B_{t_T} (MC)")
    ax.axvline(P0, color="black", linestyle="--", label=f"P(0,T) = {P0:.4f}")
    ax.set_xlabel("P(t_T,T) / B_{t_T}")
    ax.set_ylabel("count")
    ax.set_title("Discounted bond price martingale test")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 9. Print and return.
    print("[exp4] Fitted vs theoretical lambda_j:")
    for j in range(N):
        print(
            f"  j={j + 1}: fitted={fitted_slopes[j]:.5f}, theoretical={lambdas[j]:.5f}, "
            f"abs_dev={slope_deviation[j]:.5f}"
        )
    print(f"[exp4] Short-rate/log(B) consistency deviation: {short_rate_log_B_deviation:.5f}")
    print(
        f"[exp4] Martingale test: P0={P0:.6f}, MC average={mc_average:.6f}, "
        f"relative diff={relative_difference:.6f}, MC std error={mc_std_error:.6f}, "
        f"within 3 SE={within_3_se}"
    )
    print(f"[exp4] G matrix diagonal matches eigenvalues: {g_diagonal_matches}")
    print(f"[exp4] Results written to {RESULTS_JSON_PATH}")
    print(f"[exp4] Plots written to {RESULTS_PLOT_PATH}")

    return results


if __name__ == "__main__":
    run_experiment()
