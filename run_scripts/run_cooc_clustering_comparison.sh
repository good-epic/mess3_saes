#!/bin/bash

# Compare co-occurrence clustering metrics against geometry-based k-subspaces.
# Collects co-occurrence stats once (or loads from cache), then clusters with
# 4 metrics (|phi|, mutual_info, ami, jaccard) and compares.
#
# Run from project root: ./run_scripts/run_cooc_clustering_comparison.sh

set -e
export PYTHONPATH=.

OUTPUT_DIR="/workspace/outputs/cooc_comparison"
COOC_CACHE="${OUTPUT_DIR}/cooc_stats.npz"

python -u scratch/cooc_clustering_comparison.py \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN} \
    --hf_dataset "HuggingFaceFW/fineweb" \
    --hf_subset_name "sample-10BT" \
    --dataset_split "train" \
    --shuffle_buffer_size 50000 \
    --max_doc_tokens 3000 \
    --prefetch_size 1024 \
    --cooc_cache_path "${COOC_CACHE}" \
    --total_tokens 10_000_000 \
    --batch_size 32 \
    --seq_len 256 \
    --n_clusters_list 512 768 \
    --geometry_clustering_dir "/workspace/outputs/real_data_analysis_canonical" \
    --output_dir "${OUTPUT_DIR}" \
    --metrics "phi,mutual_info,ami,jaccard" \
    --seed 42

echo ""
echo "============================================================"
echo "Comparison complete!"
echo "============================================================"
echo "Results: ${OUTPUT_DIR}/comparison_n512.json"
echo "         ${OUTPUT_DIR}/comparison_n768.json"
echo "Cooc cache: ${COOC_CACHE}"
echo "============================================================"
