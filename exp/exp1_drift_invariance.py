"""Experiment 1: drift-invariance of the kernel family k(x,y) = e^{b(x+y)}
(c e^{axy} + 1 - c), replicating the eigenfunction property of Theorem 3.3
(Sections 3.2-3.3) of Celary, Eksi Altay, Filipovic and Kruhner,
"Drift-invariant kernels for discount curve models" (arXiv:2611.09751).

For each parameter set (a, b, c) and cash-flow grid, this script:
  1. checks the normalization k(0,0) = 1 (Section 3);
  2. cross-checks the analytical Schur complement k_0 (eq. 3.6) against a
     generic numerical Schur complement k_0 = k(x,.) - k(x,0)k(0,.) (eq. 3.1);
  3. verifies that, for every grid point x_j, g_j(y) = k_0(x_j, y) is an
     eigenfunction of the linear map alpha (eq. 2.2) with eigenvalue
     a*x_j + b, i.e. alpha(g_j) = (a*x_j+b) g_j (eq. 3.7, Theorem 3.3(1)).

It then contrasts this with a Gaussian/Sobolev-type kernel, for which the
analogous ratio alpha(g_j)(y)/g_j(y) is *not* constant in y, demonstrating
the failure of drift-invariance outside the real-analytic kernel family of
Theorem 3.3 (Section 3.3, p. 7).

Run as: python -m exp.exp1_drift_invariance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import (  # noqa: E402
    EXP1_FD_STEP,
    EXP1_GAUSSIAN_SIGMA,
    EXP1_GRIDS,
    EXP1_PARAM_SETS,
    EXP1_Y_EVAL,
)
from src.drift import alpha_map, eigen_ratio  # noqa: E402
from src.kernels import (  # noqa: E402
    drift_invariant_kernel,
    flat_prior_curve,
    gaussian_rbf_kernel,
    schur_complement_k0_analytic,
    schur_complement_numeric,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSON_PATH = RESULTS_DIR / "exp1_drift_invariance_results.json"
PNG_PATH = RESULTS_DIR / "exp1_eigenfunction_ratios.png"

EIGEN_FD_TOL = 1e-4
GAUSSIAN_REL_STD_THRESHOLD = 0.05


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars (and dict/list containers)
    into plain Python types suitable for `json.dump`.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _check_drift_invariant_combination(
    a: float, b: float, c: float, grid: np.ndarray
) -> Dict[str, Any]:
    """Run checks 1-3 of the methodology for one (param_set, grid) pair."""
    k00 = drift_invariant_kernel(np.array(0.0), np.array(0.0), a, b, c)
    assert abs(float(k00) - 1.0) < 1e-10, "k(0,0) must equal 1 (normalization)"

    sample_pairs = [
        (float(grid[0]), float(grid[0])),
        (float(grid[0]), float(grid[-1])),
        (float(grid[-1]), float(grid[-1])),
    ]
    sample_values = [
        float(drift_invariant_kernel(np.array(x), np.array(y), a, b, c))
        for x, y in sample_pairs
    ]

    x_grid, y_grid = np.meshgrid(grid, grid, indexing="ij")
    k0_analytic = schur_complement_k0_analytic(x_grid, y_grid, a, b, c)
    k0_numeric = schur_complement_numeric(
        drift_invariant_kernel, x_grid, y_grid, a=a, b=b, c=c
    )
    abs_diff = np.abs(k0_analytic - k0_numeric)
    max_schur_mismatch = float(np.max(abs_diff))
    # For large a*x*y the intermediate value c*e^{axy} can reach ~1e21, so the
    # numeric formula (which adds and subtracts O(1) terms next to that huge
    # value before multiplying by e^{b(x+y)}) loses float64 precision through
    # catastrophic cancellation; a pure absolute tolerance is then physically
    # unachievable. We therefore require the *relative* mismatch (mismatch
    # scaled by the magnitude of the analytic value, floored at 1.0) to be at
    # machine-precision level, which is the correct notion of "agreement" here
    # and still catches genuine implementation bugs.
    max_relative_mismatch = float(np.max(abs_diff / np.maximum(np.abs(k0_analytic), 1.0)))
    assert max_relative_mismatch < 1e-6, (
        f"Analytic and numeric Schur complements disagree: "
        f"max abs diff={max_schur_mismatch}, max relative diff={max_relative_mismatch}"
    )

    p0_func: Callable[[np.ndarray], np.ndarray] = lambda y: flat_prior_curve(y, b)

    per_grid_point: List[Dict[str, Any]] = []
    for x_j in grid:
        g_j: Callable[[np.ndarray], np.ndarray] = (
            lambda y, x_j=x_j: schur_complement_k0_analytic(x_j, y, a, b, c)
        )
        alpha_values = alpha_map(g_j, EXP1_Y_EVAL, p0_func=p0_func, h=EXP1_FD_STEP)
        g_values = g_j(EXP1_Y_EVAL)
        ratio = eigen_ratio(alpha_values, g_values)
        expected_eigenvalue = a * float(x_j) + b

        valid = ~np.isnan(ratio)
        if np.any(valid):
            max_abs_deviation = float(np.max(np.abs(ratio[valid] - expected_eigenvalue)))
        else:
            max_abs_deviation = float("nan")
        assert max_abs_deviation < EIGEN_FD_TOL, (
            f"Eigenfunction property violated at x_j={x_j}: "
            f"max abs deviation {max_abs_deviation}"
        )

        per_grid_point.append(
            {
                "x_j": float(x_j),
                "expected_eigenvalue": expected_eigenvalue,
                "ratio_values": ratio,
                "ratio_mean": float(np.nanmean(ratio)),
                "ratio_std": float(np.nanstd(ratio)),
                "max_abs_deviation": max_abs_deviation,
                "is_eigenfunction": bool(max_abs_deviation < EIGEN_FD_TOL),
            }
        )

    return {
        "a": a,
        "b": b,
        "c": c,
        "grid": grid,
        "k00": float(k00),
        "sample_pairs": sample_pairs,
        "sample_values": sample_values,
        "max_schur_mismatch": max_schur_mismatch,
        "per_grid_point": per_grid_point,
    }


def _check_gaussian_non_invariance(sigma: float, grid: np.ndarray) -> Dict[str, Any]:
    """Methodology step 5: show the eigenfunction property fails for the
    Gaussian/Sobolev-type kernel, for which p_0(y) = k(0,y) = e^{-y^2/(2 sigma^2)}.
    """
    p0_func: Callable[[np.ndarray], np.ndarray] = lambda y: gaussian_rbf_kernel(
        np.zeros_like(np.asarray(y, dtype=float)), y, sigma
    )

    per_grid_point: List[Dict[str, Any]] = []
    for x_j in grid:
        g_j: Callable[[np.ndarray], np.ndarray] = (
            lambda y, x_j=x_j: schur_complement_numeric(
                gaussian_rbf_kernel, x_j, y, sigma=sigma
            )
        )
        alpha_values = alpha_map(g_j, EXP1_Y_EVAL, p0_func=p0_func, h=EXP1_FD_STEP)
        g_values = g_j(EXP1_Y_EVAL)
        ratio = eigen_ratio(alpha_values, g_values)

        ratio_mean = float(np.nanmean(ratio))
        ratio_std = float(np.nanstd(ratio))
        relative_std = ratio_std / abs(ratio_mean) if ratio_mean != 0.0 else float("inf")

        per_grid_point.append(
            {
                "x_j": float(x_j),
                "ratio_values": ratio,
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
                "relative_std": relative_std,
                "is_not_constant": bool(relative_std > GAUSSIAN_REL_STD_THRESHOLD),
            }
        )

    return {"sigma": sigma, "grid": grid, "per_grid_point": per_grid_point}


def _plot_eigenfunction_ratios(
    drift_invariant_result: Dict[str, Any], gaussian_result: Dict[str, Any]
) -> None:
    """Methodology step 7: two-panel figure of alpha(g_j)(y)/g_j(y) vs y."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(drift_invariant_result["per_grid_point"])))
    for point, color in zip(drift_invariant_result["per_grid_point"], colors):
        label = f"x_j={point['x_j']:g}"
        ax_left.plot(EXP1_Y_EVAL, point["ratio_values"], "o-", color=color, label=label)
        ax_left.axhline(point["expected_eigenvalue"], color=color, linestyle="--", linewidth=1)
    ax_left.set_xlabel("y")
    ax_left.set_ylabel("alpha(g_j)(y) / g_j(y)")
    ax_left.set_title("Drift-invariant kernel: constant ratio (eigenfunction holds)")
    ax_left.legend(fontsize=8)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(gaussian_result["per_grid_point"])))
    for point, color in zip(gaussian_result["per_grid_point"], colors):
        label = f"x_j={point['x_j']:g}"
        ax_right.plot(EXP1_Y_EVAL, point["ratio_values"], "o-", color=color, label=label)
    ax_right.set_xlabel("y")
    ax_right.set_ylabel("alpha(g_j)(y) / g_j(y)")
    ax_right.set_title("Gaussian kernel: non-constant ratio (eigenfunction fails)")
    ax_right.legend(fontsize=8)

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_experiment() -> Dict[str, Any]:
    """Run Experiment 1 end-to-end and return the full results dict."""
    combination_results: List[Dict[str, Any]] = []
    for param_set in EXP1_PARAM_SETS:
        for grid in EXP1_GRIDS:
            result = _check_drift_invariant_combination(
                a=param_set["a"], b=param_set["b"], c=param_set["c"], grid=grid
            )
            combination_results.append(result)

    gaussian_result = _check_gaussian_non_invariance(
        sigma=EXP1_GAUSSIAN_SIGMA, grid=EXP1_GRIDS[0]
    )

    results: Dict[str, Any] = {
        "drift_invariant_combinations": combination_results,
        "gaussian_non_invariance": gaussian_result,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=True)

    _plot_eigenfunction_ratios(combination_results[0], gaussian_result)

    print("Experiment 1: drift-invariance of the kernel family")
    print("=" * 60)
    for result in combination_results:
        n_points = len(result["grid"])
        max_dev = max(p["max_abs_deviation"] for p in result["per_grid_point"])
        print(
            f"a={result['a']}, b={result['b']}, c={result['c']}, "
            f"N={n_points}: Schur mismatch={result['max_schur_mismatch']:.2e}, "
            f"max eigenfunction deviation={max_dev:.2e}"
        )
    print("-" * 60)
    print("Gaussian kernel (non-drift-invariant comparison):")
    for point in gaussian_result["per_grid_point"]:
        print(
            f"  x_j={point['x_j']:g}: ratio mean={point['ratio_mean']:.4f}, "
            f"std={point['ratio_std']:.4f}, relative_std={point['relative_std']:.4f}"
        )
    print(f"Results written to {JSON_PATH}")
    print(f"Figure written to {PNG_PATH}")

    return results


if __name__ == "__main__":
    run_experiment()
