# Drift-invariant kernels for discount curve models — replication

This repository reproduces the numerical experiments of:

> Andreas Celary, Zehra Eksi Altay, Damir Filipovic, and Paul Krühner (2026),
> "Drift-invariant kernels for discount curve models", arXiv:2611.09751.

The paper studies kernel ridge regression (KRR) estimators of the discount
curve. By the representer theorem the KRR estimate lies in a finite
dimensional affine space `A = p_0 + V` determined by the kernel and by the
grid of cash flow dates. The paper asks when `A` is invariant under the
risk-neutral drift of the discount curve, for every grid, and shows that this
holds if and only if the kernel has the form

```
k(x, y) = e^{b(x+y)} (c e^{a x y} + 1 - c),   a, c > 0, b in R        (1.2)
```

built from the Segal-Bargmann kernel. For this kernel family the drift-
invariant subspaces of `A` correspond to sub-grids of cash flow dates, and the
discounted coefficients of a risk-neutral model built in `A` follow linear
dynamics, placing the model in the linear-rational term structure class.

All six experiments are purely analytical/numerical (Section 3 and Section 4
of the paper): no market data is used anywhere in this repository.

## Repository layout

```
src/            flat library modules (kernels, hypothesis space, subspaces,
                dynamics, KRR estimator, generalized eigenfunctions, config)
exp/            one script per experiment, run as python -m exp.<name>
tests/          pytest unit tests of the src/ modules and experiment scripts
data/           data/generate_synthetic.py generates the synthetic cash-flow
                grids, instruments and prices used by the experiments
                (no market data loader is needed for this paper)
results/        tables, metrics (JSON) and figures saved by the experiments,
                plus results/RESULTS.md with the comparison to the paper
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys are required: every experiment in this replication uses only
synthetic, analytically generated data (grids of cash flow dates, synthetic
instruments and simulated coefficient processes), as stated in each
experiment's data requirements in the paper.

## Running the tests

```bash
python -m pytest -q
```

(There is no `market_data` marker in this repository because no experiment
loads external market data.)

## Running the experiments

Each experiment writes its metrics (JSON) and figures (PNG) directly into
`results/`.

```bash
python -m exp.exp1_drift_invariance    # Experiment 1: drift-invariance of k(x,y), Section 3.3
python -m exp.exp2_hypothesis_space    # Experiment 2: A as affine combinations of flat curves, Section 3.3
python -m exp.exp3_subgrid_vs_pca      # Experiment 3: sub-grids vs. PCA subspaces, Section 3.3, Corollary 3.6
python -m exp.exp4_dynamics            # Experiment 4: risk-neutral dynamics in A, Section 4
python -m exp.exp5_krr_lambda_c        # Experiment 5: KRR estimation and the lambda/c ratio, Section 3.1-3.2
python -m exp.exp6_generalized_eigen   # Experiment 6: generalized eigenfunctions, Example 3.7
```

Or run everything (tests, then all six experiments) with:

```bash
./reproduce.sh
```

All experiment parameters (kernel parameters `a, b, c`, grids of cash flow
dates, regularization values, simulation horizons, number of Monte Carlo
paths, etc.) are collected as module-level constants in `src/config.py`, each
annotated with the paper section or equation it instantiates, or marked "not
specified in the paper" when the paper leaves the value open.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the numerical comparison
against the paper's claims, the deviations from the paper, and the claims that
are not reproduced.
