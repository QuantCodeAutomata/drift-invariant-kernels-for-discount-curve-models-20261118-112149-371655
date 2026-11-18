"""Synthetic instrument generation for the experiments that require it
(experiment 2 and experiment 5): random cash-flow instruments priced from a
known discount curve, used only because the paper's experiments state "No
external data required" and specify synthetic grids, cash flows and prices
(see each experiment's data_requirements and methodology).

This module is the only place in the repository that generates synthetic
data; src/ and exp/ call it rather than sampling data themselves.
"""

from __future__ import annotations

import numpy as np

from src.hypothesis_space import flat_curve_combination, decompose_into_flat_curves


def true_discount_curve_smooth(short_rate: float, slope: float):
    """Return a callable smooth target discount curve
    p_true(x) = exp(-(short_rate + slope*x) * x), used as ground truth to
    price the synthetic instruments of experiment 5 (methodology step 3,
    "Generate a true discount curve").

    The paper's methodology illustrates p_true as an affine combination of
    flat discount curves with an example set of weights ("e.g. z_0=0.3,
    z_1=0.15, ..."), but does not fix exact values. Representing a bounded,
    monotonically decaying curve exactly in the flat-curve basis
    e^{lambda_j x} on a wide grid (up to x=30 years) requires weights that
    differ by many orders of magnitude between the shortest and longest grid
    dates, which is not required by the experiment's research objectives
    (those concern the KRR *estimate* hat_p lying in the hypothesis space,
    verified directly from its beta/k_0 representation in krr_fit, not the
    data-generating curve). A smooth target yield curve is used instead so
    that synthetic instrument prices stay at economically plausible
    magnitudes.
    """

    def curve(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.exp(-(short_rate + slope * x) * x)

    return curve


def true_discount_curve(z: np.ndarray, lambdas: np.ndarray):
    """Return a callable true discount curve p_true(x) = sum_j z_j e^{lambda_j x},
    with sum_j z_j = 1 (Lemma 3.4). ``z`` and ``lambdas`` must have equal length.
    Used directly only where the grid is short enough to stay well-conditioned
    (e.g. experiment 2); see ``true_discount_curve_smooth`` for experiment 5.
    """
    z = np.asarray(z, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    assert abs(z.sum() - 1.0) < 1e-6, "z must sum to 1 to define a valid discount curve (Lemma 3.4)"

    def curve(x):
        return flat_curve_combination(z, lambdas, np.atleast_1d(np.asarray(x, dtype=float)))

    return curve


def generate_instruments(
    x_grid: np.ndarray, true_curve, m: int, seed: int
) -> dict:
    """Generate ``m`` synthetic fixed-income instruments with cash flows on
    the grid ``x_grid`` and prices consistent with ``true_curve``.

    Each instrument is either a zero-coupon bond (cash flow 100 at one random
    grid date) or a coupon bond (a random annual coupon at every grid date up
    to a randomly chosen maturity index, plus a face value of 100 at that
    maturity), reflecting standard fixed-income instrument types.

    Returns a dict with cashflows (m, N), prices (m,), and instrument_types.
    """
    rng = np.random.default_rng(seed)
    x_grid = np.asarray(x_grid, dtype=float)
    N = len(x_grid)
    cashflows = np.zeros((m, N))
    instrument_types = []
    for i in range(m):
        if rng.random() < 0.4:
            # zero-coupon bond: 100 at a single random grid date
            j = rng.integers(0, N)
            cashflows[i, j] = 100.0
            instrument_types.append(f"zero@{x_grid[j]:g}")
        else:
            # coupon bond: random coupon rate at every date up to a random
            # maturity index, face value 100 at maturity
            maturity_idx = rng.integers(1, N)
            coupon = rng.uniform(1.0, 6.0)
            cashflows[i, : maturity_idx + 1] = coupon
            cashflows[i, maturity_idx] += 100.0
            instrument_types.append(f"coupon{coupon:.2f}@{x_grid[maturity_idx]:g}")

    prices = cashflows @ true_curve(x_grid).reshape(-1)
    return {"cashflows": cashflows, "prices": prices, "instrument_types": instrument_types}
