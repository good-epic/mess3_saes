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
#    --vertex_skip_docs 1_200_000 \
#    --max_inputs_per_cluster 10_000_000 \
python -u real_data_tests/refit_selected_clusters.py \
    --skip_training \
    --stage1_models_dir "/workspace/outputs/real_data_analysis_canonical" \
    --vertex_skip_docs 1_500_000 \
    --n_clusters_list "768" \
    --csv_dir "/workspace/outputs/real_data_analysis_canonical" \
    --save_dir "/workspace/outputs/selected_clusters_canonical" \
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
    --manual_cluster_ids "768:114,140,210,306,570,581,586,596,672,678,704,733,748" \
    --manual_k "768:114=3,140=3,210=5,306=3,570=3,581=3,586=3,596=3,672=3,678=3,704=4,733=5,748=4"
# Original broader search
#    --manual_cluster_ids "768:24,43,49,60,75,91,114,125,139,140,153,193,228,257,306,328,329,342,409,417,418,419,450,455,499,570,581,586,596,672,678,704,733,748" \
#    --manual_k "768:24=4,43=3,49=4,60=3,75=4,91=3,114=5,125=4,139=5,140=5,153=5,193=3,228=4,257=4,306=5,328=5,329=4,342=4,409=3,417=4,418=3,419=5,450=4,455=4,499=4,570=4,581=3,586=3,596=3,672=3,678=3,704=4,733=5,748=4"


# MODE 2: Retrain at elbow k + collect (ALTERNATIVE - uncomment to use)
# Use this if you want different hyperparameters or fine-tuning
#
# python -u real_data_tests/refit_selected_clusters.py \
#     --n_clusters_list "512,768" \
#     --csv_dir "/workspace/outputs/real_data_analysis_canonical" \
#     --save_dir "/workspace/outputs/selected_clusters_canonical" \
#     --model_name "gemma-2-9b" \
#     --sae_release "gemma-scope-9b-pt-res" \
#     --sae_id "layer_20/width_16k/average_l0_68" \
#     --device "cuda" \
#     --cache_dir "/workspace/hf_cache" \
#     --hf_token ${HF_TOKEN} \
#     --hf_dataset "HuggingFaceFW/fineweb" \
#     --hf_subset_name "sample-10BT" \
#     --dataset_split "train" \
#     --activity_batch_size 32 \
#     --activity_seq_len 256 \
#     --seed 42 \
#     --aanet_prefetch_size 1024 \
#     --shuffle_buffer_size 50000 \
#     --max_doc_tokens 3000 \
#     --aanet_streaming_steps 3000 \
#     --aanet_warmup_steps 100 \
#     --aanet_warmup_cluster_chunk_size 16 \
#     --aanet_batch_size 128 \
#     --aanet_lr 0.0025 \
#     --aanet_weight_decay 1e-5 \
#     --aanet_layer_widths 64 32 \
#     --aanet_simplex_scale 1.0 \
#     --aanet_noise 0.05 \
#     --aanet_gamma_reconstruction 1.0 \
#     --aanet_gamma_archetypal 4.0 \
#     --aanet_gamma_extrema 2.0 \
#     --aanet_grad_clip 1.0 \
#     --aanet_lr_patience 30 \
#     --aanet_lr_factor 0.5 \
#     --aanet_min_lr 1e-6 \
#     --aanet_early_stop_patience 250 \
#     --aanet_min_delta 1e-6 \
#     --aanet_loss_smoothing_window 20 \
#     --aanet_seed 43 \
#     --aanet_active_threshold 1e-6 \
#     --aanet_min_samples 32 \
#     --aanet_restarts_no_extrema 1 \
#     --extrema_enabled \
#     --extrema_knn 150 \
#     --extrema_max_points 30000 \
#     --extrema_pca 0.95 \
#     --extrema_seed 431 \
#     --extrema_warmup_samples 10000 \
#     --collect_vertex_samples \
#     --samples_per_vertex 1000 \
#     --vertex_distance_threshold 0.02 \
#     --min_vertex_ratio 0.1 \
#     --vertex_search_batch_size 32 \
#     --concurrent_aanets 5 \
#     --max_inputs_per_cluster 1000000 \
#     --vertex_save_interval 5000

echo ""
echo "============================================================"
echo "Stage 3 Complete!"
echo "============================================================"
echo "Vertex samples saved to: /workspace/outputs/selected_clusters_canonical/"
echo ""
echo "Next step: Run interpretation pipeline on vertex samples"
echo "============================================================"
