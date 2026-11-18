"""Parameters for the replication of Celary, Eksi Altay, Filipovic and Kruhner,
"Drift-invariant kernels for discount curve models" (arXiv:2611.09751).

All experiments in this paper are purely analytical/numerical: no market data
is required (see each experiment's "data_requirements"). Every module-level
constant below is a plain value with a comment on its meaning, unit and the
paper section or equation it instantiates, or "not specified in the paper"
when the paper leaves it open.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Experiment 1: verification of drift-invariance for the kernel family
# k(x,y) = e^{b(x+y)}(c e^{axy} + 1 - c), Theorem 3.3(1), Section 3.3.
# ---------------------------------------------------------------------------
EXP1_PARAM_SETS = [
    {"a": 1.0, "b": 0.0, "c": 1.0},     # paper's normalization example, Sec. 3.3
    {"a": 0.5, "b": -0.02, "c": 2.0},   # second parameter set, methodology step 1
]
EXP1_GRIDS = [
    np.array([0.25, 0.5, 1.0, 2.0, 5.0]),                              # N=5 grid
    np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]),   # N=10 grid
]
EXP1_Y_EVAL = np.array([0.3, 0.7, 1.3, 2.7, 4.1, 6.3, 9.0])  # y-points away from 0, unit: years
EXP1_FD_STEP = 1e-5          # finite-difference step h for numerical derivatives, not specified in the paper
EXP1_GAUSSIAN_SIGMA = 2.0    # length scale theta of the Sobolev-type proxy kernel exp(-(x-y)^2/(2 sigma^2)); proxy choice, see uncertainties in exp_1

# ---------------------------------------------------------------------------
# Experiment 2: hypothesis space as affine combinations of flat discount
# curves, Section 3.3, Lemma 3.4 and the basis functions (3.9).
# ---------------------------------------------------------------------------
EXP2_A = 0.5            # kernel curvature parameter a, methodology step 1
EXP2_B = -0.03           # kernel level parameter b (continuously compounded short-end yield), methodology step 1
EXP2_C = 1.0             # kernel scale parameter c, methodology step 1
EXP2_GRID = np.array([0.5, 1.0, 2.0, 5.0, 10.0])  # cash flow grid x_1<...<x_5, methodology step 2
EXP2_N_INSTRUMENTS = 20  # number of synthetic instruments m, methodology step 7
EXP2_LAMBDA = 0.01       # KRR regularization parameter lambda, methodology step 8
EXP2_SEED = 20261118     # RNG seed for synthetic instruments and prices, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 3: drift-invariant sub-grid subspaces (Proposition 3.5) versus
# PCA eigenvector subspaces of K_0 (Corollary 3.6), Section 3.3, Appendix B.5.
# ---------------------------------------------------------------------------
EXP3_A, EXP3_B, EXP3_C = 1.0, 0.0, 1.0  # so that k_0(x,y) = e^{xy} - 1, methodology step 1
# N=10 grid with the same increasing-spacing structure as the experiment's own
# illustrative grid [0.5,...,30], but with a narrower span. The paper does not
# fix numeric grid values for this illustration (Section 3.3, Corollary 3.6):
# only the qualitative conclusions (sub-grid subspaces are drift-invariant,
# PCA-eigenvector subspaces are not) are asserted. With the wider illustrative
# span, self-products x_i*x_j reach 900 and K_0 entries reach exp(900), so
# K_0's eigenvalues are separated by over 170 orders of magnitude; its top
# eigenvectors then collapse (in floating point) onto single standard basis
# vectors, making the "PCA" subspace coincide with a single k_0(x_j, .) and
# trivially satisfy drift-invariance -- an artifact of catastrophic
# ill-conditioning, not the generic-mixing mechanism of Corollary 3.6. The
# narrower grid below keeps K_0 entries within double precision (max ~2.8e4)
# while keeping K_0 ill-conditioned enough (cond ~4e12) that its top
# eigenvectors are genuine mixtures of several basis functions, which is what
# Corollary 3.6 requires to show that alpha(W) is generically not in W.
EXP3_GRID = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2])  # N=10 grid
EXP3_SUBGRID_J = [1, 4, 7]   # 0-indexed sub-grid {2,5,8} (1-indexed in the paper), methodology step 4
EXP3_PCA_D = 3               # number of top eigenvectors d, methodology step 5
EXP3_FD_STEP = 1e-5          # finite-difference step for alpha, not specified in the paper
EXP3_Y_EVAL = np.array([0.4, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0])  # evaluation grid for residual tests

# ---------------------------------------------------------------------------
# Experiment 4: risk-neutral coefficient dynamics, Theorem 4.1, Remark 4.2,
# equations (4.1)-(4.6).
# ---------------------------------------------------------------------------
EXP4_A, EXP4_B, EXP4_C = 0.5, -0.03, 1.0        # kernel parameters, methodology step 1
EXP4_GRID = np.array([0.5, 1.0, 2.0, 5.0, 10.0])  # cash flow grid, methodology step 1
EXP4_YTILDE0_INIT = 0.01   # initial discounted coefficients Ytilde_0^j, j=1..N, methodology step 4; not specified in the paper beyond positivity
EXP4_D_FACTORS = 3        # number of Brownian motion factors d, methodology step 4
EXP4_NU_SCALE = 0.05      # scale of volatility vectors nu_j (annualized), not prescribed by the paper (see uncertainties)
EXP4_NU_SEED = 20261118   # RNG seed for the volatility vectors nu_j, not specified in the paper
EXP4_DT = 1.0 / 252       # Euler-Maruyama time step (one trading day), methodology step 5
EXP4_N_STEPS = 252        # number of time steps over T=1 year, methodology step 5
EXP4_N_PATHS_FULL = 20000   # number of Monte Carlo paths for the martingale test, paper configuration (not specified; chosen for Monte Carlo accuracy)
EXP4_N_PATHS_RUN = 20000    # number of Monte Carlo paths actually executed (equal to the full configuration: the simulation is cheap)
EXP4_BOND_MATURITY_T = 5.0  # maturity T (years) of the bond used in the martingale test, methodology step 10
EXP4_SIM_SEED = 42           # RNG seed for the Euler-Maruyama simulation, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 5: KRR estimation with the drift-invariant kernel on synthetic
# bond data, Sections 3.1-3.2, equations (3.3)-(3.4).
# ---------------------------------------------------------------------------
EXP5_GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])  # N=11 grid, methodology step 2
# Kernel curvature parameter a, methodology step 1. The paper does not fix a
# numeric value for this illustration (Sections 3.1-3.2); the experiment's own
# suggested a=0.5 makes a*x_i*x_j reach 0.5*30*30=450 on this N=11, 30-year
# grid, so k_0(x,y)=c e^{b(x+y)}(e^{axy}-1) and the instrument price matrix
# C K_0 C^T reach magnitudes above 1e40 -- far larger than any lambda in a
# economically meaningful range, so the regularization term has no visible
# effect on the estimate irrespective of the linear solver used. a=0.01 keeps
# K_0 within double precision (max ~1.3e3) while preserving the qualitative
# KRR properties under test (hat_p(0)=1, membership in A, lambda/c-invariance,
# monotonicity of the pricing error in lambda/c).
EXP5_A, EXP5_B = 0.01, -0.03
# True discount curve used to price synthetic instruments (methodology step 3),
# p_true(x) = exp(-(EXP5_TRUE_SHORT_RATE + EXP5_TRUE_SLOPE*x)*x)
# (data/generate_synthetic.py, true_discount_curve_smooth). The paper gives
# only an illustrative example of an affine combination of flat discount
# curves z_j and does not fix exact values; see the deviation recorded in
# results/RESULTS.md for why a smooth target is used instead.
EXP5_TRUE_SHORT_RATE = 0.03   # target short-end yield (continuously compounded), not specified in the paper
EXP5_TRUE_SLOPE = 0.0015      # target yield curve slope per year, not specified in the paper
EXP5_N_INSTRUMENTS = 30       # number of synthetic instruments m, methodology step 4
EXP5_WEIGHTS = 1.0            # instrument weights w_i = 1, methodology step 5
EXP5_C_BASE = 1.0             # base scale parameter c, methodology step 6
# Regularization values lambda for c=1, methodology step 11. With a=0.01 and
# m=30 face-value-100 instruments on the grid above, C K_0 C^T has eigenvalues
# up to order 1e7 (zero for the m-N directions outside the rank-N range of
# K_0); the geometric range below spans from near-interpolation (lambda far
# below that scale) to heavy shrinkage toward the flat prior (lambda far
# above it), which the experiment's suggested [0.001,0.01,0.1,1.0] (tuned to
# a=0.5, not specified by the paper) does not do at this a.
EXP5_LAMBDAS_FOR_C1 = [1.0e-2, 1.0e2, 1.0e5, 1.0e8]
EXP5_C_ALT = 2.0                                # alternate scale parameter c, methodology step 11
EXP5_LAMBDAS_FOR_C_ALT = [2.0e-2, 2.0e2, 2.0e5, 2.0e8]  # lambda values for c=2 with the same lambda/c ratios
EXP5_SEED = 20261118           # RNG seed for synthetic instruments and prices, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 6: Example 3.7, generalized eigenfunctions and the
# exponential-polynomial family W = span{e^x - 1, x e^x}.
# ---------------------------------------------------------------------------
EXP6_A, EXP6_B = 1.0, 0.0     # so that p_0 = 1 and k_0(x,y) = e^{xy} - 1, Example 3.7
EXP6_FD_STEP = 1e-5           # finite-difference step, not specified in the paper
EXP6_Y_EVAL = np.array([0.2, 0.5, 0.9, 1.4, 2.1, 3.0])  # evaluation grid for the eigenfunction checks
EXP6_TAYLOR_ORDERS = [1, 2, 3, 4]  # Taylor coefficient orders compared, methodology step 6
EXP6_OPT_N_RESTARTS = 25           # number of random restarts for the nonlinear least-squares check, not specified in the paper
EXP6_OPT_SEED = 7                  # RNG seed for the optimization restarts, not specified in the paper
EXP6_X_FIT_GRID = np.linspace(0.05, 3.0, 60)  # grid used to fit c1*(e^{x1 x}-1)+c2*(e^{x2 x}-1) to x*e^x
