"""Unit tests for Example 3.7: generalized eigenfunctions and the
exponential-polynomial family W = span{e^x - 1, x*e^x} (paper Example 3.7,
Section 2).
"""

from __future__ import annotations

import numpy as np

from src.config import EXP6_FD_STEP, EXP6_TAYLOR_ORDERS, EXP6_Y_EVAL
from src.generalized_eigen import (
    alpha_h1,
    alpha_h2,
    fit_two_exponential_combination,
    h1,
    h2,
    jordan_block_matrix,
    taylor_match_residual,
)


def test_alpha_h1_is_eigenfunction_eigenvalue_one():
    """alpha(e^x - 1) = e^x - 1 exactly (eigenvalue 1)."""
    y = EXP6_Y_EVAL
    np.testing.assert_allclose(alpha_h1(y, EXP6_FD_STEP), h1(y), atol=1e-5)


def test_alpha_h2_equals_h1_plus_h2():
    """alpha(x*e^x) = (e^x - 1) + x*e^x, i.e. a generalized eigenfunction."""
    y = EXP6_Y_EVAL
    expected = h1(y) + h2(y)
    np.testing.assert_allclose(alpha_h2(y, EXP6_FD_STEP), expected, atol=1e-4)


def test_jordan_block_structure():
    """The matrix representation of alpha in basis {h1, h2} is the Jordan
    block [[1, 1], [0, 1]] (paper Example 3.7)."""
    J = jordan_block_matrix(EXP6_Y_EVAL, EXP6_FD_STEP)
    expected = np.array([[1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(J, expected, atol=5e-3)


def test_taylor_system_inconsistent():
    """No x1, x2, c1, c2 reproduce the Taylor coefficients of x*e^x via
    c1*(e^{x1 x}-1) + c2*(e^{x2 x}-1); the residual must be bounded away
    from zero for generic (x1, x2)."""
    residual = taylor_match_residual(1.0, 2.0, 1.0, 1.0, EXP6_TAYLOR_ORDERS)
    assert residual > 1e-6


def test_numerical_fit_has_nonzero_residual():
    """The best nonlinear least-squares fit of x*e^x by two flat-curve
    exponentials has a strictly positive residual, confirming x*e^x is not
    expressible in that two-dimensional exponential family."""
    result = fit_two_exponential_combination(
        np.linspace(0.05, 3.0, 60), n_restarts=10, seed=0
    )
    assert result["residual"] > 1e-3


def test_h_functions_vanish_at_zero():
    """h1(0) = h2(0) = 0, consistent with Proposition 2.2 (h(0) = 0 for h in W)."""
    assert abs(h1(np.array([0.0]))[0]) < 1e-12
    assert abs(h2(np.array([0.0]))[0]) < 1e-12
