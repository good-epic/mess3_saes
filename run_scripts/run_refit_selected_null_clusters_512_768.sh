#!/bin/bash

# Stage 3: Load AANet models + Collect vertex samples
# Run from project root: ./run_scripts/run_refit_selected_clusters.sh
#
# Since Stage 1 now saves all trained models, the default workflow is to:
#   1. Load pre-trained models from Stage 1
#   2. Collect vertex samples only
#
# This saves ~2-3 hours vs retraining!
#
# TWO MODES:
#
# MODE 1 (Default - Load Stage 1 Models):
#   Loads pre-trained models from Stage 1, only collects vertex samples
#   ✅ Faster (saves 2-3 hours)
#   ✅ Uses models trained on more data (all k values)
#   Requires: --skip_training --stage1_models_dir --vertex_skip_docs
#
# MODE 2 (Alternative - Retrain):
#   Retrains AANets at elbow k, then collects vertex samples
#   Use when: You want different hyperparameters or fine-tuning
#   (See commented example at bottom)

export PYTHONPATH=.

# MODE 1: Load Stage 1 models + collect vertex samples (DEFAULT)
#    --vertex_skip_docs 1_000_000 \
#    --max_inputs_per_cluster 10_000_000 \
python -u real_data_tests/refit_selected_clusters.py \
    --skip_training \
    --stage1_models_dir "/workspace/outputs/null_clusters" \
    --n_clusters_list "512" \
    --vertex_skip_docs 1_000_000 \
    --csv_dir "/workspace/outputs/null_clusters" \
    --save_dir "/workspace/outputs/selected_null_clusters" \
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
    --seed 4343 \
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
    --max_inputs_per_cluster 10_000_000 \
    --vertex_save_interval 5000 \
    --manual_cluster_ids "512:75,138,203,222,268,336,345;768:113,310,548,559,646,689" \
    --manual_k "512:75=3,138=3,203=3,222=4,268=3,336=3,345=3;768:113=3,310=3,548=3,559=3,646=3,689=4"
    
echo ""
echo "============================================================"
echo "Stage 3 Complete!"
echo "============================================================"
echo "Vertex samples saved to: /workspace/outputs/selected_clusters_canonical/"
echo ""
echo "Next step: Run interpretation pipeline on vertex samples"
echo "============================================================"
