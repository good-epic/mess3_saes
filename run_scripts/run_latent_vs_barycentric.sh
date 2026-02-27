#!/bin/bash

# 2Bb: Barycentric coordinate vs. single-latent predictive power
#
# GPU required. Runs vertex samples through Gemma-2-9b + SAE + AANet and
# compares how well individual cluster latents vs. the full barycentric
# coordinate predict:
#   1. Vertex membership (K-class classification)
#   2. Primary linguistic feature from 1c (spacy)
#   3. Next-token log-probabilities (top-50 tokens by log-prob variance)
#
# Run from project root: ./run_scripts/run_latent_vs_barycentric.sh
#
# Prerequisites:
#   - Refitted AANet models in SOURCE_DIR (from run_null_refit / refit_selected_clusters)
#   - Prepared vertex samples in PREPARED_SAMPLES_DIR (already exist for broad_2)
#   - spacy: pip install spacy && python -m spacy download en_core_web_sm

set -e

export PYTHONPATH=.

PREPARED_SAMPLES_DIR="outputs/interpretations/prepared_samples_broad_2_no_whitespace"
SOURCE_DIR="/workspace/outputs/selected_clusters_broad_2"
CSV_DIR="/workspace/outputs/real_data_analysis_canonical"
OUTPUT_DIR="/workspace/outputs/validation/latent_vs_barycentric"

# Priority clusters with defined feature extractors
CLUSTERS="512_17,512_22,512_67,512_181,512_229,512_261,512_471,512_504,\
768_140,768_210,768_306,768_581,768_596"

echo "============================================================"
echo "2Bb: LATENT VS. BARYCENTRIC REGRESSION"
echo "============================================================"
echo "Prepared samples:  ${PREPARED_SAMPLES_DIR}"
echo "AANet source dir:  ${SOURCE_DIR}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Clusters:          ${CLUSTERS}"
echo ""

python -u validation/latent_vs_barycentric.py \
    --prepared_samples_dir "${PREPARED_SAMPLES_DIR}" \
    --source_dir "${SOURCE_DIR}" \
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
    --aanet_noise 0.05

echo ""
echo "============================================================"
echo "Done. Results in: ${OUTPUT_DIR}/"
echo "  - all_results.json (combined)"
echo "  - cluster_*/  (per-cluster JSON + PNG plots)"
echo "============================================================"
