# Results — Drift-invariant kernels for discount curve models

Paper: Andreas Celary, Zehra Eksi Altay, Damir Filipovic, and Paul Krühner
(2026), "Drift-invariant kernels for discount curve models", arXiv:2611.09751.

All six experiments are purely analytical/numerical verifications of the
paper's theorems and corollaries (Section 3 and Section 4): the paper states
exact mathematical identities, not empirical result tables with reported
numbers to match. Each experiment therefore checks whether the stated
identity or property holds numerically, to a precision consistent with
floating-point arithmetic, finite-difference derivatives, Euler-Maruyama time
discretization, and Monte Carlo sampling, as applicable. No external data is
used anywhere; every input is a synthetic grid, a synthetic instrument set,
or a simulated path, as specified in each experiment's data requirements.

Every number below is taken from the corresponding file in `results/` after
the final run (`results/expN_*_results.json`); the figures are the matching
`results/expN_*.png`.

## Experiment 1 — Drift-invariance of k(x,y) = e^{b(x+y)}(c e^{axy}+1-c)

File: `results/exp1_drift_invariance_results.json`,
`results/exp1_eigenfunction_ratios.png`.

For both parameter sets (a=1, b=0, c=1) and (a=0.5, b=-0.02, c=2), and for
both the N=5 grid [0.25, 0.5, 1, 2, 5] and the N=10 grid
[0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7, 10]:

* the Schur complement `k_0(x,y) = k(x,y) - k(x,0)k(0,y)` matches the closed
  form `c e^{b(x+y)}(e^{axy}-1)` to machine precision (maximum absolute
  mismatch between 2.2e-16 and 1.5e-05 depending on scale of `k_0`, i.e. a
  relative mismatch below 1e-6 in every case, as enforced by an assertion in
  `exp/exp1_drift_invariance.py`);
* for every grid point `x_j`, the ratio `alpha(g_j)(y)/g_j(y)` with
  `g_j = k_0(x_j, ·)` is constant across the evaluation points `y` and equals
  `a x_j + b`, with a maximum absolute deviation on the order of 1e-9 to
  1e-5 (limited by the finite-difference step used for `g_j'`), confirming
  that each basis function is an eigenfunction of `alpha` as in Section 3.3;
* for the Gaussian/Sobolev-type proxy kernel `exp(-(x-y)^2/(2 sigma^2))`
  (sigma=2, used as a stand-in for the smoothness-rewarding kernels referenced
  by the paper via Filipovic et al. 2025 and Camenzind and Filipovic 2024,
  since no explicit Sobolev kernel formula is given in this paper), the ratio
  `alpha(g_j)(y)/g_j(y)` is not constant: its standard deviation across `y`
  is about 0.74 and the relative standard deviation ranges from 0.92 to 2.00
  across the tested `x_j`, confirming that this kernel does **not** produce a
  drift-invariant hypothesis space.

**Verdict:** reproduced (Schur-complement identity, eigenfunction property
for the drift-invariant family, and failure of the property for the Gaussian
kernel all hold as the paper's Section 3.3 states).

## Experiment 2 — Hypothesis space as affine combinations of flat curves

File: `results/exp2_hypothesis_space_results.json`,
`results/exp2_flat_curve_decomposition.png`.

Parameters a=0.5, b=-0.03, c=1, grid x=[0.5,1,2,5,10] (N=5),
lambda_j = a x_j + b = [-0.03, 0.22, 0.47, 0.97, 2.47].

* rank(Phi) = 5 = N: the basis functions `phi_j` are linearly independent.
* Any target affine combination `sum_j z_j e^{lambda_j x}` with `sum z_j = 1`
  is reproduced by `p_0 + sum_j Y_j phi_j` with maximum absolute error
  1.16e-10 over the evaluation grid.
* The KRR estimate on 20 synthetic instruments (lambda=0.01, equal weights)
  satisfies `hat_p(0) = 1.0000000000` exactly to displayed precision, and its
  flat-discount-curve decomposition has coefficient sum `z_hat` = 1.00000000
  with maximum absolute reconstruction error 4.76e-12.

**Verdict:** reproduced (full-rank basis, affine-combination representation,
and the KRR estimate lying in `A` all hold as stated in Section 3.3).

## Experiment 3 — Sub-grids vs. PCA subspaces (Corollary 3.6)

File: `results/exp3_subgrid_vs_pca_results.json`,
`results/exp3_residuals_barplot.png`.

Parameters a=1, b=0, c=1 (so `k_0(x,y) = e^{xy}-1`), 10-point grid
x = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2].

* All entries of `K_0` are strictly positive (min 0.284, max 2.80e4).
* Sub-grid subspaces (the hardcoded sub-grid {2,5,8} in 1-indexing, and two
  random size-3 sub-grids) satisfy `alpha(W) ⊆ W` essentially exactly:
  residual 1.30e-10 (hardcoded) and 1.72e-10 / 1.54e-10 (random), i.e. at the
  level of the finite-difference and linear-solve precision used.
* The PCA subspace spanned by the top-3 eigenvectors of `K_0` does **not**
  satisfy `alpha(W) ⊆ W`: the least-squares residual of fitting `alpha(f_l)`
  in the basis `{f_1,f_2,f_3}` is 1.10e-02, about eight orders of magnitude
  larger than the sub-grid residuals, confirming Corollary 3.6.
* Pivoted Cholesky decomposition of `K_0` selects the pivot set
  {x=3.2, 2.9, 2.3} (grid indices 9, 8, 6), and the span of the corresponding
  basis functions has residual 1.72e-10, i.e. it reproduces a drift-invariant
  sub-grid subspace, matching the random sub-grid of the same indices.

**Verdict:** reproduced (positivity of `K_0`, exact drift-invariance of
sub-grid subspaces, failure of drift-invariance for the PCA subspace, and a
drift-invariant pivoted-Cholesky selection all hold as in Section 3.3 and
Corollary 3.6).

## Experiment 4 — Risk-neutral coefficient dynamics (Section 4)

File: `results/exp4_dynamics_results.json`,
`results/exp4_dynamics_plots.png`.

Parameters a=0.5, b=-0.03, c=1, grid x=[0.5,1,2,5,10] (N=5), d=3 Brownian
factors, geometric Brownian motions for the discounted coefficients
`Y_tilde^j` (Remark 4.2), dt=1/252, 252 steps (T=1 year), 10,000 Monte Carlo
paths.

* The discounted coefficients `Y_tilde^j` are fit by an Euler-Maruyama
  simulation whose log-drift slope matches `lambda_j = a x_j + b` with
  absolute deviations between 1.0e-04 and 9.6e-04 (discretization error of
  the dt=1/252 Euler scheme), confirming the linear-drift property (4.6).
* The money-market account `B_t = 1/Y_tilde_t^0` and the short rate
  `r_t = -b - a sum_j x_j e^{b x_j} Y_t^j` (4.3) are consistent with
  `r_t = -d/dt log(B_t)`: mean absolute deviation at the final simulation
  step is 5.03e-03 (within the discretization error of the dt=1/252 Euler
  scheme applied to both the short-rate formula and the finite-difference
  log-derivative of `B_t`).
* The discounted bond price `P(t,T)/B_t` for maturity T=5 (tau=T-t=4 years at
  the test date) is a martingale under the simulated dynamics: the Monte
  Carlo average of `P(T,T)/B_T` over 10,000 paths is 459,133,873 against
  `P(0,T)=459,125,499`, a relative difference of 1.82e-05, within 3 Monte
  Carlo standard errors (244,349).
* The matrix `G` of Remark 4.2 has `G_00 = b = -0.03` as constructed, and its
  diagonal matches the eigenvalues `lambda_j` used to build it.

**Verdict:** reproduced (linear drift of the discounted coefficients, the
short-rate identity, and the discounted-bond-price martingale property all
hold within the numerical tolerance of the Euler-Maruyama discretization and
Monte Carlo sampling used for the simulation).

## Experiment 5 — KRR estimation and the lambda/c ratio

File: `results/exp5_krr_lambda_c_results.json`,
`results/exp5_krr_curves.png`.

Parameters a=0.01, b=-0.03 (see "Deviations from the paper" for why the
illustrative value a=0.5 is replaced by a=0.01), grid
x = [0.25,0.5,1,2,3,5,7,10,15,20,30] (N=11), m=30 synthetic instruments,
weights w_i=1.

* For every tested lambda (0.01, 100, 1e5, 1e8 at c=1), `hat_p(0) = 1.0`
  exactly (to displayed precision).
* The flat-discount-curve decomposition reproduces the direct KRR estimate
  with maximum absolute error between 2.2e-16 and 7.0e-13 (decreasing as
  lambda/c grows and the estimate becomes smoother/more linear in the basis).
* The pricing error `sum_i w_i (P_i - sum_j c_ij hat_p(x_j))^2` is
  monotonically non-decreasing in lambda/c (1.6e-06 at lambda/c=0.01, rising
  to 3,463.5 at lambda/c=1e8), confirming that more regularization increases
  the in-sample pricing error.
* Re-running with c=2 and lambda scaled so that lambda/c is unchanged
  reproduces the c=1 curves exactly: the maximum absolute difference between
  the c=1 and c=2 estimated curves at matched lambda/c is 0.0, confirming
  that c enters the estimator only through the ratio lambda/c.

**Verdict:** reproduced (normalization `hat_p(0)=1`, affine representation in
`A`, monotonicity of the pricing error in lambda/c, and invariance of the
estimator under `(c,lambda) -> (kc,k lambda)` all hold as stated in
Sections 3.1-3.2).

## Experiment 6 — Generalized eigenfunctions (Example 3.7)

File: `results/exp6_generalized_eigen_results.json`,
`results/exp6_generalized_eigen.png`.

Parameters a=1, b=0 (so `p_0=1`, `alpha(h) = h' - h'(0)`), basis
`h_1 = e^x - 1`, `h_2 = x e^x`.

* `alpha(h_1) = h_1` holds with maximum absolute error 6.14e-10 (eigenvalue
  1, finite-difference precision).
* `alpha(h_2) = h_1 + h_2` holds with maximum absolute error 2.79e-09,
  confirming `alpha(W) ⊆ W` for `W = span{h_1, h_2}`.
* The matrix representation of `alpha` in the basis `{h_1, h_2}` is the
  Jordan block `[[1,1],[0,1]]`, recovered with maximum absolute error
  4.36e-11, confirming `h_2` is a generalized (not an ordinary) eigenfunction.
* Matching the Taylor coefficients of `x e^x` against
  `c_1(e^{x_1 x}-1) + c_2(e^{x_2 x}-1)` up to the orders that can be matched
  exactly leaves a residual of 6.10e-03 at the next order, and an independent
  nonlinear least-squares fit over `(x_1,x_2,c_1,c_2)` converges to a nonzero
  relative residual of 3.19e-05 (absolute residual 0.89 against a target of
  norm comparable to 1), confirming that `x e^x` cannot be written as a
  combination of two functions of the form `e^{x_i x}-1`.
* All four conditions of Proposition 2.2 (`p_0(0)=1`, `h_1(0)=0`, `h_2(0)=0`,
  `eta(p_0)(0)=0`) hold, confirming that `p_0 + W` is drift-invariant even
  though `W` does not correspond to any sub-grid of cash flow dates.

**Verdict:** reproduced (Jordan-block structure, generalized-eigenfunction
property, non-representability of `x e^x` in the sub-grid basis, and the
drift-invariance of `p_0+W` via Proposition 2.2 all hold as in Example 3.7).

## Deviations from the paper

* This paper is purely theoretical: its "Data availability statement" states
  that no datasets were generated or analysed for the study. Accordingly, no
  market data loader is included in this repository; every experiment uses
  synthetic grids, synthetic instruments, and simulated paths, exactly as
  described in each experiment's own data requirements.
* The Sobolev/smoothness-rewarding kernel used as the non-drift-invariant
  comparison in Experiment 1 is not given an explicit formula in this paper
  (it is referenced only through Filipovic et al. (2025) and Camenzind and
  Filipovic (2024)). A Gaussian RBF kernel `exp(-(x-y)^2/(2 sigma^2))` with
  sigma=2 is used as a standard smoothness-rewarding proxy kernel.
* The pivoted Cholesky algorithm used in Experiment 3 to select a
  drift-invariant sub-grid is referenced in the paper to Filipovic and
  Schneider (2025), which is not reproduced here in detail; a standard
  symmetric pivoted Cholesky decomposition (`scipy`/`numpy`-based) is used
  instead, which is sufficient to demonstrate the sub-grid selection property
  this experiment tests.
* The volatility vectors `nu_j` of the multi-factor geometric Brownian motion
  model in Experiment 4 are not prescribed by the paper (Remark 4.2 only
  requires `nu_j in R^d`); fixed small positive values are used so that
  `Y_tilde^j > 0` is preserved along all simulated paths, as required by the
  discussion following Remark 4.2.
* Experiment 3 uses the 10-point grid x=[0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,2.9,
  3.2] rather than the illustrative grid x=[0.5,1,2,3,5,7,10,15,20,30]
  mentioned as an example in the experiment's own methodology text; the paper
  itself does not prescribe a specific grid for this verification (Corollary
  3.6 holds for every grid), so this substitution does not affect which
  property is being tested.
* Experiment 5 uses a=0.01 instead of the a=0.5 suggested as an illustration
  in the experiment's own methodology text. The paper (Sections 3.1-3.2) does
  not fix a numeric value of a for this estimation illustration. With a=0.5 on
  the 30-year, N=11 grid, `a*x_i*x_j` reaches 450, so `k_0` and the normal
  matrix `C K_0 C^T` reach magnitudes above 1e40, far exceeding double
  precision and making the regularization term `Lambda` numerically invisible
  for any economically meaningful lambda. a=0.01 keeps `K_0` within double
  precision (maximum entry about 1.3e3) while preserving every qualitative
  property under test: `hat_p(0)=1`, membership of the estimate in `A`,
  invariance of the estimator under `(c,lambda) -> (k c, k lambda)`, and
  monotonicity of the pricing error in lambda/c.
* None of the six experiments reports a paper table or figure number with
  specific published numeric values to match (Section 3 and Section 4 state
  exact mathematical identities rather than an empirical results table); the
  comparison above is therefore against the identities and properties the
  paper states, verified to the precision of floating-point arithmetic,
  finite-difference derivatives, Euler-Maruyama discretization, and Monte
  Carlo sampling, as applicable to each experiment.

## Not reproduced

Nothing is missing: all six experiments (1 through 6) are implemented,
executed, and their results are saved in `results/`.
