"""Tests for exp/exp4_dynamics.py and the underlying src.dynamics /
src.hypothesis_space functions used by Experiment 4 (Section 4 of
Celary, Eksi Altay, Filipovic and Kruhner, "Drift-invariant kernels for
discount curve models").
"""

from __future__ import annotations

import numpy as np

from src.config import EXP4_A, EXP4_B, EXP4_D_FACTORS, EXP4_GRID, EXP4_YTILDE0_INIT
from src.dynamics import bond_price, build_G_matrix, short_rate, simulate_Ytilde
from src.hypothesis_space import eigenvalues


def test_build_G_matrix_shape_and_diagonal() -> None:
    G = build_G_matrix(EXP4_GRID, EXP4_A, EXP4_B)
    N = len(EXP4_GRID)
    assert G.shape == (N + 1, N + 1)
    assert G[0, 0] == EXP4_B
    eigvals = eigenvalues(EXP4_GRID, EXP4_A, EXP4_B)
    assert np.allclose(np.diag(G), eigvals)


def test_simulate_Ytilde_small_run_is_well_behaved() -> None:
    rng = np.random.default_rng(0)
    nu = 0.02 * rng.standard_normal((len(EXP4_GRID), EXP4_D_FACTORS))
    n_steps = 50
    n_paths = 200
    dt = 1.0 / 252
    sim = simulate_Ytilde(EXP4_GRID, EXP4_A, EXP4_B, EXP4_YTILDE0_INIT, nu, dt, n_steps, n_paths, seed=1)

    N = len(EXP4_GRID)
    assert sim["Ytilde"].shape == (n_steps + 1, n_paths, N + 1)
    assert sim["Y"].shape == (n_steps + 1, n_paths, N)
    assert sim["B"].shape == (n_steps + 1, n_paths)
    assert sim["time_grid"].shape == (n_steps + 1,)
    assert sim["lambdas"].shape == (N,)

    assert not np.any(np.isnan(sim["Ytilde"]))
    assert not np.any(np.isnan(sim["Y"]))
    assert not np.any(np.isnan(sim["B"]))
    assert np.all(sim["Ytilde"][:, :, 1:] >= 0.0)
    assert np.all(sim["Ytilde"][:, :, 0] > 0.0)


def test_short_rate_matches_analytical_formula_at_t0() -> None:
    rng = np.random.default_rng(2)
    n_paths = 5
    N = len(EXP4_GRID)
    Y0_single = 0.01 * rng.standard_normal(N) ** 2  # arbitrary positive-ish values
    Y0 = np.tile(Y0_single, (n_paths, 1))  # identical across paths

    r = short_rate(Y0, EXP4_GRID, EXP4_A, EXP4_B)
    expected = -EXP4_B - EXP4_A * np.sum(EXP4_GRID * np.exp(EXP4_B * EXP4_GRID) * Y0_single)

    assert r.shape == (n_paths,)
    assert np.allclose(r, expected)
    assert np.allclose(r, r[0])  # identical initial Y across paths -> identical repeated value


def test_bond_price_at_tau_zero_equals_one() -> None:
    rng = np.random.default_rng(3)
    N = len(EXP4_GRID)
    Y = 0.02 * rng.standard_normal((7, N))  # representative random Y values, several "paths"

    price = bond_price(Y, 0.0, EXP4_GRID, EXP4_A, EXP4_B)
    assert price.shape == (7, 1)
    assert np.allclose(price, 1.0)


def test_run_experiment_smoke_with_reduced_simulation() -> None:
    """Smoke test that exercises the same src functions used by run_experiment,
    but with a reduced number of paths/steps to keep the test fast (the full
    configuration uses EXP4_N_PATHS_RUN=20000 paths over 252 steps, which is
    kept untouched in src/config.py).
    """
    import exp.exp4_dynamics as exp4_module

    rng = np.random.default_rng(exp4_module.EXP4_NU_SEED)
    nu = exp4_module.EXP4_NU_SCALE * rng.standard_normal((len(exp4_module.EXP4_GRID), exp4_module.EXP4_D_FACTORS))

    sim = simulate_Ytilde(
        exp4_module.EXP4_GRID,
        exp4_module.EXP4_A,
        exp4_module.EXP4_B,
        exp4_module.EXP4_YTILDE0_INIT,
        nu,
        exp4_module.EXP4_DT,
        n_steps=30,
        n_paths=500,
        seed=exp4_module.EXP4_SIM_SEED,
    )
    N = len(exp4_module.EXP4_GRID)
    assert sim["Ytilde"].shape == (31, 500, N + 1)
    assert not np.any(np.isnan(sim["Ytilde"]))

    r = short_rate(sim["Y"], exp4_module.EXP4_GRID, exp4_module.EXP4_A, exp4_module.EXP4_B)
    assert r.shape == (31, 500)
    assert not np.any(np.isnan(r))
