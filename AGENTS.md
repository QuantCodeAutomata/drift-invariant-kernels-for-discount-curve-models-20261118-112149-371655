# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of six numerical experiments from Celary, Eksi Altay,
Filipovic and Kruhner, "Drift-invariant kernels for discount curve models"
(arXiv:2611.09751):

1. `exp1_drift_invariance` — verification of the drift-invariance property of
   `k(x,y) = e^{b(x+y)}(c e^{axy} + 1 - c)` (Section 3.3) and of its failure
   for a Gaussian/Sobolev-type kernel.
2. `exp2_hypothesis_space` — the hypothesis space `A` as affine combinations
   of flat discount curves, linear independence of the basis functions, and
   a KRR fit lying in `A` (Section 3.3).
3. `exp3_subgrid_vs_pca` — drift-invariant subspaces of `V` correspond to
   sub-grids of cash flow dates; PCA/eigenvector-based dimension reduction of
   the kernel matrix breaks drift-invariance (Section 3.3, Corollary 3.6).
4. `exp4_dynamics` — risk-neutral dynamics of the coefficient process in `A`:
   quadratic drift of `Y`, linear drift of the discounted coefficients
   `Y_tilde`, the augmented matrix `G` of Remark 4.2, and the discounted bond
   price martingale property (Section 4, Theorem 4.1).
5. `exp5_krr_lambda_c` — the full KRR estimator (3.3)-(3.4) on synthetic
   instruments, the normalization `hat_p(0)=1`, and the invariance of the
   estimator under `(c, lambda) -> (kc, k*lambda)` (Sections 3.1-3.2).
6. `exp6_generalized_eigen` — Example 3.7: the generalized eigenfunction
   `x e^x` of the drift map `alpha`, the Jordan-block structure, and the
   impossibility of representing `x e^x` as a combination of two
   exponential-affine basis functions.

All experiments are purely analytical/numerical; no market data is used in
this paper, so there is no `data/data_loader.py` and no API keys are
required.

## Layout

```
src/     flat modules: kernels.py, hypothesis_space.py, subspaces.py,
         dynamics.py, krr.py, generalized_eigen.py, drift.py, config.py
exp/     one script per experiment (exp1_... .. exp6_...), run as a module
tests/   pytest tests of the src/ modules and of the experiment scripts
data/    generate_synthetic.py builds the synthetic grids, cash-flow
         matrices, prices and Brownian paths used by the experiments
results/ JSON metrics and PNG figures written by the experiment scripts,
         plus RESULTS.md with the comparison against the paper
```

## Conventions

* All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` constants, each commented with its meaning, unit, and the
  paper section/equation it instantiates (or "not specified in the paper").
  No parameter is hardcoded inside `src/` or `exp/` modules.
* Synthetic data generation lives only in `data/generate_synthetic.py`; `src/`
  and `exp/` call it rather than generating their own ad hoc inputs.
* Run experiments from the repository root as `python -m exp.<name>`; run
  tests as `python -m pytest`. Do not edit `sys.path` in code.
* Deviations from the paper and methodology interpretations are recorded only
  in `results/RESULTS.md`, never silently in code.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                      # unit tests (no market_data marker: no live data is used)
python -m exp.exp1_drift_invariance
python -m exp.exp2_hypothesis_space
python -m exp.exp3_subgrid_vs_pca
python -m exp.exp4_dynamics
python -m exp.exp5_krr_lambda_c
python -m exp.exp6_generalized_eigen
./reproduce.sh                           # tests + all six experiments
```

## Notes for future changes

After rerunning an experiment, its command overwrites the corresponding files
in `results/`. The numbers and verdicts in `results/RESULTS.md` are not
regenerated automatically: re-check them against the new result files and
update the table by hand before committing.
