#!/bin/bash

# Phase 3b (null): Causal Steering Auto-Interpretation — NULL clusters
#
# Same pipeline as run_causal_steering_autointerp.sh but for the three null
# clusters (random, non-simplex-structured). Uses sonnet_null synthesis and
# prepared_samples_null for grounding.
#
# Three test types:
#   document  — original post-trigger text from the source document
#   baseline  — unsteered (greedy) continuation
#   steered   — continuations steered at scales [1, 5, 20] × 3 types × (k-1)
#               directions per vertex
#
# GPU required. Run from project root:
#   ./run_scripts/run_null_causal_steering_autointerp.sh
#
# Prerequisites:
#   - Phase 3a null steering outputs in STEERING_DIR (run_causal_steering.sh)
#   - Synthesis JSONs in SYNTHESIS_DIR (sonnet_null/synthesis)
#   - vllm: pip install vllm

set -e

export PYTHONPATH=.

STEERING_DIR="/workspace/outputs/validation/causal_steering_null"
SYNTHESIS_DIR="/workspace/outputs/interpretations/sonnet_null/synthesis"
PREPARED_SAMPLES_DIR="/workspace/outputs/interpretations/prepared_samples_null"
OUTPUT_DIR="/workspace/outputs/validation/causal_steering_autointerp_null"

# Null clusters
CLUSTERS="512_138,512_345,768_310"

echo "============================================================"
echo "Phase 3b (null): CAUSAL STEERING AUTO-INTERPRETATION"
echo "============================================================"
echo "Steering results:  ${STEERING_DIR}"
echo "Synthesis dir:     ${SYNTHESIS_DIR}"
echo "Prepared samples:  ${PREPARED_SAMPLES_DIR}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Clusters:          ${CLUSTERS}"
echo ""

python -u validation/causal_steering_autointerp.py \
    --steering_dir "${STEERING_DIR}" \
    --synthesis_dir "${SYNTHESIS_DIR}" \
    --prepared_samples_dir "${PREPARED_SAMPLES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${CLUSTERS}" \
    --scales 1 5 20 \
    --n_samples 30 \
    --n_exemplars 5 \
    --model_name "Qwen/Qwen2.5-72B-Instruct-AWQ" \
    --quantization "awq_marlin" \
    --gpu_memory_utilization 0.85 \
    --max_model_len 8192 \
    --gemma_tokenizer "google/gemma-2-9b" \
    --vllm_batch_size 256 \
    --cache_dir "/workspace/hf_cache" \
    --hf_token "${HF_TOKEN}" \
    --seed 42

echo ""
echo "============================================================"
echo "Done. Results in: ${OUTPUT_DIR}/"
echo "  - all_results.json (combined metrics)"
echo "  - cluster_{key}_autointerp.json (per-cluster cases + metrics)"
echo "============================================================"
