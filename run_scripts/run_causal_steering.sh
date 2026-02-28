#!/bin/bash

# Phase 3a: Causal Steering via AANet Subspace Patching
#
# For each near-vertex example, subtracts the cluster's current AANet
# reconstruction from the layer-20 residual at the trigger position and adds
# back a scaled version of the target vertex's reconstruction. Records the
# full next-token distribution and greedy continuation (steered and unsteered)
# for downstream LLM auto-interpretation (3b) and distribution-shift analysis.
#
# Three steering types are run per example:
#   type1 — patch only trigger position T (delta persists via attention)
#   type2 — patch T + next k_sustain generated positions, then stop
#   type3 — patch T + every generated position throughout full continuation
#
# GPU required. Run from project root:
#   ./run_scripts/run_causal_steering.sh
#
# Prerequisites:
#   - Refitted AANet models in SOURCE_DIR (from refit_selected_clusters)
#   - Vertex samples in SOURCE_DIR (from select_clusters / refit pipeline)
#   - Consolidated metrics CSVs in CSV_DIR

set -e

export PYTHONPATH=.

SOURCE_DIR="/workspace/outputs/selected_clusters_broad_2"
CSV_DIR="/workspace/outputs/real_data_analysis_canonical"
OUTPUT_DIR="/workspace/outputs/validation/causal_steering"

# All 13 priority clusters (4 HIGH + 9 MEDIUM confidence)
CLUSTERS="512_17,512_22,512_67,512_181,512_229,512_261,512_471,512_504,\
768_140,768_210,768_306,768_581,768_596"

echo "============================================================"
echo "Phase 3a: CAUSAL STEERING"
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
