#!/bin/bash

# KL divergence distributions across the simplex (Validation 2A)
#
# Computes three distributions of pairwise divergences per cluster:
#   1. Same-vertex pairs
#   2. Cross-vertex pairs
#   3. Within-simplex pairs (using collected simplex samples)
#
# Reports sym_kl and JS divergence, plus head/tail diagnostic.
# Requires GPU (model forward passes).
#
# Run AFTER run_collect_simplex_samples.sh
# Run from project root: ./run_scripts/run_kl_divergence.sh

set -e

export PYTHONPATH=.

# All broad_2 priority clusters
PRIORITY_CLUSTERS="512_17,512_22,512_67,512_181,512_229,512_261,512_471,512_504,768_114,768_140,768_210,768_306,768_570,768_581,768_596,768_672"

VERTEX_SAMPLES_DIR="/workspace/outputs/selected_clusters_broad_2"
SIMPLEX_DIR="/workspace/outputs/simplex_samples"

echo "============================================================"
echo "KL DIVERGENCE ANALYSIS — broad_2 clusters (2A)"
echo "============================================================"
echo "Priority clusters:   ${PRIORITY_CLUSTERS}"
echo "Vertex samples dir:  ${VERTEX_SAMPLES_DIR}"
echo "Simplex samples dir: ${SIMPLEX_DIR}"

python -u validation/kl_divergence_simplex.py \
    --clusters "${PRIORITY_CLUSTERS}" \
    --vertex_samples_dir "${VERTEX_SAMPLES_DIR}" \
    --simplex_samples_dir "${SIMPLEX_DIR}" \
    --output_dir "/workspace/outputs/validation/kl_divergence_broad_2" \
    --batch_size 256 \
    --n_pairs_per_sample 20 \
    --max_samples_per_vertex 200 \
    --max_simplex_samples 5000 \
    --n_diagnostic_pairs 1000 \
    --seed 42 \
    --model_name "gemma-2-9b" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN}

echo ""
echo "============================================================"
echo "Done. Results in /workspace/outputs/validation/kl_divergence_broad_2/"
echo "============================================================"
