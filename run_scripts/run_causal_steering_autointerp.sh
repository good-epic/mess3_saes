#!/bin/bash

# Phase 3b: Causal Steering Auto-Interpretation
#
# Classifies steered and unsteered continuations from Phase 3a into vertex
# categories using Qwen-72B-AWQ via vLLM. Uses the existing frontier-model
# synthesis (consolidated_vertex_labels) as grounding and exemplar windows
# from prepared_samples.
#
# Three test types:
#   document  — original post-trigger text from the source document
#   baseline  — unsteered (greedy) continuation
#               (measures classification ceiling per vertex)
#   steered   — continuations steered at scales [1, 5, 20] × 3 types × (k-1)
#               directions per vertex (measures causal shift rate + dose-response)
#
# GPU required. Run from project root:
#   ./run_scripts/run_causal_steering_autointerp.sh
#
# Prerequisites:
#   - Phase 3a outputs in STEERING_DIR (run_causal_steering.sh)
#   - Synthesis JSONs in SYNTHESIS_DIR (already exist for broad_2)
#   - vllm: pip install vllm

set -e

export PYTHONPATH=.

STEERING_DIR="/workspace/outputs/validation/causal_steering"
SYNTHESIS_DIR="/workspace/outputs/interpretations/sonnet_broad_2/synthesis"
PREPARED_SAMPLES_DIR="/workspace/outputs/interpretations/prepared_samples_broad_2"
OUTPUT_DIR="/workspace/outputs/validation/causal_steering_autointerp"

# All 13 priority clusters
CLUSTERS="512_17,512_22,512_67,512_181,512_229,512_261,512_471,512_504,\
768_140,768_210,768_306,768_581,768_596"

echo "============================================================"
echo "Phase 3b: CAUSAL STEERING AUTO-INTERPRETATION"
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
