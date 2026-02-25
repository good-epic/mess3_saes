#!/bin/bash

# Collect random simplex samples for KL divergence analysis (2A)
#
# Uses pre-trained AANet models from selected_clusters_broad_2.
# Collects 5000 random simplex samples per cluster with token-position
# distribution matched to that cluster's near-vertex samples (decile buckets).
# Also samples 10 random control clusters per n_clusters value.
#
# Run from project root: ./run_scripts/run_collect_simplex_samples.sh

set -e

export PYTHONPATH=.

# Priority clusters from broad_2 (high + medium confidence synthesis results)
CLUSTERS_512="512:17,22,67,181,229,261,471,504"
CLUSTERS_768="768:114,140,210,306,570,581,596,672"
MANUAL_K_512="512:17=3,22=4,67=3,181=3,229=3,261=3,471=4,504=3"
MANUAL_K_768="768:114=3,140=3,210=3,306=3,570=3,581=3,596=3,672=3"

echo "============================================================"
echo "COLLECT SIMPLEX SAMPLES — broad_2 priority clusters"
echo "============================================================"

# --- 512 clusters ---
echo ""
echo "Processing 512 clusters..."
python -u real_data_tests/collect_simplex_samples.py \
    --n_clusters_list "512" \
    --source_dir "/workspace/outputs/selected_clusters_broad_2" \
    --csv_dir "/workspace/outputs/real_data_analysis_canonical" \
    --save_dir "/workspace/outputs/simplex_samples" \
    --manual_cluster_ids "${CLUSTERS_512}" \
    --manual_k "${MANUAL_K_512}" \
    --n_random_controls 10 \
    --controls_seed 99 \
    --n_simplex_samples 5000 \
    --n_position_buckets 10 \
    --skip_docs 2000000 \
    --search_batch_size 32 \
    --max_inputs_per_cluster 2000000 \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN} \
    --hf_dataset "HuggingFaceFW/fineweb" \
    --hf_subset_name "sample-10BT" \
    --dataset_split "train" \
    --activity_seq_len 256 \
    --seed 42 \
    --aanet_prefetch_size 1024 \
    --shuffle_buffer_size 50000 \
    --max_doc_tokens 3000 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05

# --- 768 clusters ---
echo ""
echo "Processing 768 clusters..."
python -u real_data_tests/collect_simplex_samples.py \
    --n_clusters_list "768" \
    --source_dir "/workspace/outputs/selected_clusters_broad_2" \
    --csv_dir "/workspace/outputs/real_data_analysis_canonical" \
    --save_dir "/workspace/outputs/simplex_samples" \
    --manual_cluster_ids "${CLUSTERS_768}" \
    --manual_k "${MANUAL_K_768}" \
    --n_random_controls 10 \
    --controls_seed 99 \
    --n_simplex_samples 5000 \
    --n_position_buckets 10 \
    --skip_docs 2500000 \
    --search_batch_size 32 \
    --max_inputs_per_cluster 2000000 \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN} \
    --hf_dataset "HuggingFaceFW/fineweb" \
    --hf_subset_name "sample-10BT" \
    --dataset_split "train" \
    --activity_seq_len 256 \
    --seed 43 \
    --aanet_prefetch_size 1024 \
    --shuffle_buffer_size 50000 \
    --max_doc_tokens 3000 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05

echo ""
echo "============================================================"
echo "Done. Simplex samples saved to: /workspace/outputs/simplex_samples/"
echo "Next: run run_kl_divergence.sh"
echo "============================================================"
