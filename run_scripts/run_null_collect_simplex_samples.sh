#!/bin/bash

# Collect within-simplex samples for null clusters (2A / 2Ba)
#
# Same script as run_collect_simplex_samples.sh but pointing at null cluster
# AANet models. Run AFTER run_null_refit_512/768.sh and after reviewing null
# vertex samples to decide which null clusters to carry forward.
#
# Fill in CLUSTERS_512, CLUSTERS_768, MANUAL_K_512, MANUAL_K_768 below
# based on null elbow plot review (same manual step as null refit).
#
# No random controls — null clusters are themselves the baseline.
#
# Run from project root: ./run_scripts/run_null_collect_simplex_samples.sh

set -e

export PYTHONPATH=.

# TODO: fill in after null refit + vertex sample review
CLUSTERS_512="512:CLUSTER_ID_1,CLUSTER_ID_2"
CLUSTERS_768="768:CLUSTER_ID_1,CLUSTER_ID_2"
MANUAL_K_512="512:CLUSTER_ID_1=K1,CLUSTER_ID_2=K2"
MANUAL_K_768="768:CLUSTER_ID_1=K1,CLUSTER_ID_2=K2"

NULL_DIR="/workspace/outputs/null_clusters"

echo "============================================================"
echo "COLLECT SIMPLEX SAMPLES — null clusters"
echo "============================================================"

# --- 512 null clusters ---
echo ""
echo "Processing null 512 clusters..."
python -u real_data_tests/collect_simplex_samples.py \
    --n_clusters_list "512" \
    --source_dir "${NULL_DIR}" \
    --csv_dir "${NULL_DIR}" \
    --save_dir "/workspace/outputs/simplex_samples" \
    --manual_cluster_ids "${CLUSTERS_512}" \
    --manual_k "${MANUAL_K_512}" \
    --n_random_controls 0 \
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
    --seed 44 \
    --aanet_prefetch_size 1024 \
    --shuffle_buffer_size 50000 \
    --max_doc_tokens 3000 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05

# --- 768 null clusters ---
echo ""
echo "Processing null 768 clusters..."
python -u real_data_tests/collect_simplex_samples.py \
    --n_clusters_list "768" \
    --source_dir "${NULL_DIR}" \
    --csv_dir "${NULL_DIR}" \
    --save_dir "/workspace/outputs/simplex_samples" \
    --manual_cluster_ids "${CLUSTERS_768}" \
    --manual_k "${MANUAL_K_768}" \
    --n_random_controls 0 \
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
    --seed 45 \
    --aanet_prefetch_size 1024 \
    --shuffle_buffer_size 50000 \
    --max_doc_tokens 3000 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05

echo ""
echo "============================================================"
echo "Done. Null simplex samples saved to: /workspace/outputs/simplex_samples/"
echo "Next: run run_kl_divergence.sh (needs both real + null simplex samples)"
echo "============================================================"
