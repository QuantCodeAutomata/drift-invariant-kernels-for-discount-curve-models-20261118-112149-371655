#!/usr/bin/env bash
# Reproduces the experiments of Celary, Eksi Altay, Filipovic and Kruhner,
# "Drift-invariant kernels for discount curve models" (arXiv:2611.09751).
set -uo pipefail

cd "$(dirname "$0")"

echo "=== Running tests ==="
python -m pytest -q || exit 1

echo "=== Experiment 1: drift-invariance of k(x,y) ==="
python -m exp.exp1_drift_invariance

echo "=== Experiment 2: structure of the hypothesis space ==="
python -m exp.exp2_hypothesis_space

echo "=== Experiment 3: sub-grids vs. PCA subspaces ==="
python -m exp.exp3_subgrid_vs_pca

echo "=== Experiment 4: risk-neutral coefficient dynamics ==="
python -m exp.exp4_dynamics

echo "=== Experiment 5: KRR estimation and the lambda/c ratio ==="
python -m exp.exp5_krr_lambda_c

echo "=== Experiment 6: generalized eigenfunctions (Example 3.7) ==="
python -m exp.exp6_generalized_eigen

echo "=== Done. Results are in results/ ==="
