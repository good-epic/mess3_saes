#!/bin/bash

# Phase 3a (null control): Causal Steering via AANet Subspace Patching
#
# Same analysis as run_causal_steering.sh but on null AANet clusters
# (random latent groupings with fitted AANets).
# Used to verify that causal steering on null clusters does not produce
# coherent or interpretable behavioral changes.
#
# Priority null clusters: 512_138, 512_345, 768_310
#
# GPU required. Run from project root:
#   ./run_scripts/run_null_causal_steering.sh
#
# Prerequisites:
#   - AANet models in NULL_SOURCE_DIR (from run_null_refit_512/768)
#   - Vertex samples in NULL_SOURCE_DIR (from run_null_collect_simplex_samples)
#   - Consolidated metrics CSVs in NULL_CSV_DIR

set -e

export PYTHONPATH=.

SOURCE_DIR="/workspace/outputs/selected_null_clusters"
CSV_DIR="/workspace/outputs/null_clusters"
OUTPUT_DIR="/workspace/outputs/validation/causal_steering_null"

# Priority null clusters
CLUSTERS="512_138,512_345,768_310"

echo "============================================================"
echo "Phase 3a NULL CONTROL: CAUSAL STEERING"
echo "============================================================"
echo "AANet source dir:  ${SOURCE_DIR}"
echo "CSV dir:           ${CSV_DIR}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Clusters:          ${CLUSTERS}"
echo ""

python -u validation/causal_steering.py \
    --source_dir "${SOURCE_DIR}" \
    --csv_dir "${CSV_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${CLUSTERS}" \
    --scales 1 5 20 \
    --n_gen_tokens 50 \
    --max_examples_per_vertex 30 \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token "${HF_TOKEN}" \
    --k_sustain 12 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05

echo ""
echo "============================================================"
echo "Done. Results in: ${OUTPUT_DIR}/"
echo "  - cluster_{key}/steering_results.jsonl"
echo "  - cluster_{key}/logprobs.npz"
echo "============================================================"
