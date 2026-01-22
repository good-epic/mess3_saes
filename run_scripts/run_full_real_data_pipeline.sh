#!/bin/bash

# Stage 1: Run Real Data SAE Analysis
# This script trains AANets for all clusters across all k values,
# generates analysis plots, and saves models for later use.
#
# PCA Rank Estimation:
#   --pca_num_cycles 80 means:
#     - 80 cycles × 256 sequences/cycle × 256 tokens/sequence = 5.24M token positions sampled
#     - After SAE sparsity filtering (~30-50% hit rate) → ~2M samples per cluster
#     - Subsampled to --pca_max_samples 100K if exceeded (for computational efficiency)
#     - 100K samples is more than sufficient for PCA on high-dimensional spaces (up to ~1000-D)
#     - Note: Tokens within a sequence are correlated; diversity comes from cycling through different documents

export PYTHONPATH=.

# Only for the generic clustering pipeline master class, not used here
#    --total_samples 25000 \
python -u real_data_tests/analyze_real_saes.py \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --output_dir "/workspace/outputs/real_data_analysis_canonical" \
    --n_clusters_list 512 768 \
    --latent_activity_threshold 0 \
    --activity_batch_size 32 \
    --activity_batches 4096 \
    --activity_seq_len 256 \
    --pca_num_cycles 20 \
    --pca_max_samples 100000 \
    --pca_variance_threshold 0.95 \
    --device "cuda" \
    --seed 42 \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN} \
    --hf_dataset "HuggingFaceFW/fineweb" \
    --hf_subset_name "sample-10BT" \
    --dataset_split "train" \
    --shuffle_buffer_size 50000 \
    --max_doc_tokens 3000 \
    --aanet_prefetch_size 1024 \
    --subspace_variance_threshold 0.95 \
    --subspace_gap_threshold 2.0 \
    --aanet_batch_size 128 \
    --aanet_streaming_steps 3000 \
    --aanet_warmup_steps 100 \
    --aanet_warmup_cluster_chunk_size 16 \
    --aanet_sequential_k \
    --aanet_lr 0.0025 \
    --aanet_weight_decay 1e-5 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05 \
    --aanet_gamma_reconstruction 1.0 \
    --aanet_gamma_archetypal 4.0 \
    --aanet_gamma_extrema 2.0 \
    --aanet_lr_patience 30 \
    --aanet_lr_factor 0.5 \
    --aanet_min_lr 1e-6 \
    --aanet_early_stop_patience 250 \
    --aanet_min_delta 1e-6 \
    --aanet_loss_smoothing_window 20 \
    --aanet_grad_clip 1.0 \
    --aanet_seed 43 \
    --aanet_active_threshold 1e-6 \
    --aanet_min_samples 32 \
    --aanet_restarts_no_extrema 1 \
    --extrema_enabled \
    --extrema_knn 150 \
    --extrema_max_points 30000 \
    --extrema_pca 0.95 \
    --extrema_seed 431

echo ""
echo "============================================================"
echo "Stage 1 Complete!"
echo "============================================================"
echo "Outputs saved to: /workspace/outputs/real_data_analysis_canonical/"
echo ""
echo "Next steps:"
echo "1. Review auto-generated plots in: .../clusters_N/plots/"
echo "2. (Optional) Run scratch/analyze_aanet_results.py for manual selection"
echo "3. Run: ./run_scripts/run_refit_selected_clusters.sh"
echo "============================================================"


# MODE 1: Load Stage 1 models + collect vertex samples (DEFAULT)
python -u real_data_tests/refit_selected_clusters.py \
    --skip_training \
    --stage1_models_dir "/workspace/outputs/real_data_analysis_canonical" \
    --vertex_skip_docs 300000 \
    --n_clusters_list "512,768" \
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
    --max_inputs_per_cluster 10_000_000 \
    --vertex_save_interval 5000

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
