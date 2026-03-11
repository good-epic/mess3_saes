#!/usr/bin/env bash
# kNN sparsification sweep for co-occurrence clustering.
# knn_k caps each node's degree after NC backbone, breaking the giant cluster.
# Sweep knn_k x z_threshold x gamma.

set -euo pipefail
export PYTHONPATH=.

for Z_THRESH in 1.96 3.0 5.0; do
  for KNN_K in 5 10 20 50; do
    for GAMMA in 0.01 0.05 0.1; do
      python -u clustering/cooc_leiden.py \
        --cooc_path   outputs/cooc_comparison/cooc_stats.npz \
        --output_dir  outputs/cooc_leiden_knn_sweep \
        --z_threshold "${Z_THRESH}" \
        --min_weight  5 \
        --knn_k       "${KNN_K}" \
        --gamma       "${GAMMA}" \
        --weight_type raw \
        --seed 42
    done
  done
done
