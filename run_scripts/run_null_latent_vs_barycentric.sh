#!/bin/bash

# 2Bb (null control): Barycentric coordinate vs. single-latent predictive power
# on null AANet clusters (random latent groupings).
#
# Step 1 (CPU): prepare vertex sample JSON files from the null cluster manifest.
# Step 2 (GPU): run latent_vs_barycentric.py on the prepared samples.
#
# Priority null clusters: 512_138, 512_345, 768_310
#
# Run from project root: ./run_scripts/run_null_latent_vs_barycentric.sh
#
# Prerequisites:
#   - selected_null_clusters/ must exist with vertex_samples.jsonl files
#   - null_clusters/ must contain AANet .pt models and consolidated_metrics CSVs

set -e

export PYTHONPATH=.

NULL_CLUSTERS="512_138,512_345,768_310"

MANIFEST="/workspace/outputs/selected_null_clusters/manifest.json"
PREPARED_SAMPLES_DIR="/workspace/outputs/interpretations/prepared_samples_null"
SOURCE_DIR="/workspace/outputs/selected_null_clusters"
CSV_DIR="/workspace/outputs/null_clusters"
OUTPUT_DIR="/workspace/outputs/validation/latent_vs_barycentric_null"

echo "============================================================"
echo "2Bb NULL CONTROL: LATENT VS. BARYCENTRIC REGRESSION"
echo "============================================================"
echo "Null clusters:      ${NULL_CLUSTERS}"
echo "Manifest:           ${MANIFEST}"
echo "Prepared samples:   ${PREPARED_SAMPLES_DIR}_no_whitespace"
echo "AANet/CSV dir:      ${CSV_DIR}"
echo "Output dir:         ${OUTPUT_DIR}"
echo ""

# Step 1: prepare vertex sample JSON files (CPU-only)
echo "--- Step 1: preparing vertex samples ---"
python -u interpretation/prepare_vertex_samples.py \
    --manifest "${MANIFEST}" \
    --output_dir "${PREPARED_SAMPLES_DIR}" \
    --max_samples_per_vertex 200 \
    --min_samples_per_vertex 1

echo ""
echo "--- Step 2: latent vs. barycentric regression ---"
python -u validation/latent_vs_barycentric.py \
    --prepared_samples_dir "${PREPARED_SAMPLES_DIR}_no_whitespace" \
    --source_dir "${SOURCE_DIR}" \
    --csv_dir "${CSV_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${NULL_CLUSTERS}" \
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
echo "============================================================"
