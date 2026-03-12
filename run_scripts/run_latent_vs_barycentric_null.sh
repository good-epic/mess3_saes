#!/bin/bash

# 2Bb (null): Barycentric coordinate vs. single-latent predictive power — NULL clusters
#
# GPU required. Runs vertex samples through Gemma-2-9b + SAE + AANet and
# compares how well individual cluster latents vs. the full barycentric
# coordinate predict:
#   1. Vertex membership (K-class classification)
#   2. Next-token log-probabilities (top-50 tokens by log-prob variance)
#
# No simplex interior samples exist for null clusters, so only vertex split is run.
#
# Run from project root: ./run_scripts/run_latent_vs_barycentric_null.sh
#
# Prerequisites:
#   - AANet models in CSV_DIR (aanet_cluster_138/345/310_k*.pt)
#   - Prepared vertex samples in PREPARED_SAMPLES_DIR

set -e

export PYTHONPATH=.

PREPARED_SAMPLES_DIR="/workspace/outputs/interpretations/prepared_samples_null"
CSV_DIR="/workspace/outputs/null_clusters"
OUTPUT_DIR="/workspace/outputs/validation/latent_vs_barycentric_null"

CLUSTERS="512_138,512_345,768_310"

echo "============================================================"
echo "2Bb (NULL): LATENT VS. BARYCENTRIC REGRESSION"
echo "============================================================"
echo "Prepared samples:  ${PREPARED_SAMPLES_DIR}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Clusters:          ${CLUSTERS}"
echo ""

python -u validation/latent_vs_barycentric.py \
    --prepared_samples_dir "${PREPARED_SAMPLES_DIR}" \
    --source_dir "${CSV_DIR}" \
    --csv_dir "${CSV_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${CLUSTERS}" \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token "${HF_TOKEN}" \
    --batch_size 8 \
    --n_top_tokens 50 \
    --cv_folds 5 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05 \
    --max_samples_per_vertex 200

echo ""
echo "============================================================"
echo "Done. Results in: ${OUTPUT_DIR}/"
echo "  - all_results.json (combined)"
echo "  - cluster_*/  (per-cluster JSON + PNG plots)"
echo "============================================================"
