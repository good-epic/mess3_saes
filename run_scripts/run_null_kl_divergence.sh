#!/bin/bash

# KL divergence distributions across the simplex — NULL cluster control (2A)
#
# Same analysis as run_kl_divergence.sh but on null AANet clusters
# (random latent groupings with fitted AANets).
# Used to verify that null clusters do NOT show the cross > same-vertex
# KL separation seen in real clusters.
#
# Priority null clusters: 512_138, 512_345, 768_310
# (the only three with both vertex samples and simplex samples collected)
#
# Run from project root: ./run_scripts/run_null_kl_divergence.sh

set -e

export PYTHONPATH=.

NULL_CLUSTERS="512_138,512_345,768_310"

VERTEX_SAMPLES_DIR="/workspace/outputs/selected_null_clusters"
SIMPLEX_DIR="/workspace/outputs/null_simplex_samples"

echo "============================================================"
echo "KL DIVERGENCE ANALYSIS — null clusters (2A control)"
echo "============================================================"
echo "Null clusters:       ${NULL_CLUSTERS}"
echo "Vertex samples dir:  ${VERTEX_SAMPLES_DIR}"
echo "Simplex samples dir: ${SIMPLEX_DIR}"

python -u validation/kl_divergence_simplex.py \
    --clusters "${NULL_CLUSTERS}" \
    --vertex_samples_dir "${VERTEX_SAMPLES_DIR}" \
    --simplex_samples_dir "${SIMPLEX_DIR}" \
    --output_dir "/workspace/outputs/validation/kl_divergence_null" \
    --batch_size 16 \
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
echo "Done. Results in /workspace/outputs/validation/kl_divergence_null/"
echo "============================================================"
