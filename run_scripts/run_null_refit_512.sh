#!/bin/bash

# Null Cluster Baseline — Stage 3 vertex sample collection (512 clusters)
#
# Loads pre-trained null cluster AANet models from Stage 1 and collects
# near-vertex examples for selected clusters.
#
# Run from project root: ./run_scripts/run_null_refit_512.sh
#
# Fill in --manual_cluster_ids and --manual_k below after reviewing elbow
# plots from Stage 1 (same workflow as real clusters).

export PYTHONPATH=.

python -u real_data_tests/refit_selected_clusters.py \
    --skip_training \
    --stage1_models_dir "/workspace/outputs/null_clusters" \
    --vertex_skip_docs 1_000_000 \
    --n_clusters_list "512" \
    --csv_dir "/workspace/outputs/null_clusters" \
    --save_dir "/workspace/outputs/null_clusters_selected" \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN} \
    --hf_dataset "HuggingFaceFW/fineweb" \
    --hf_subset_name "sample-10BT" \
    --dataset_split "train" \
    --activity_batch_size 32 \
    --activity_seq_len 256 \
    --seed 42 \
    --aanet_prefetch_size 1024 \
    --shuffle_buffer_size 50000 \
    --max_doc_tokens 3000 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05 \
    --collect_vertex_samples \
    --samples_per_vertex 1000 \
    --vertex_distance_threshold 0.02 \
    --min_vertex_ratio 0.1 \
    --vertex_search_batch_size 32 \
    --concurrent_aanets 5 \
    --max_inputs_per_cluster 2_000_000 \
    --vertex_save_interval 5000 \
    --manual_cluster_ids "512:FILL_IN" \
    --manual_k "512:FILL_IN"

echo ""
echo "============================================================"
echo "Stage 3 complete for null 512 clusters"
echo "============================================================"
echo "Vertex samples saved to: /workspace/outputs/null_clusters_selected/"
echo ""
echo "Next: run prepare_vertex_samples.py + interpretation pipeline"
echo "============================================================"
