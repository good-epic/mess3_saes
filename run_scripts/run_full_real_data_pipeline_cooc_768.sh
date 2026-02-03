#!/bin/bash

# Full Real Data Pipeline with Co-occurrence Clustering (768 clusters)
#
# This script runs the full pipeline for BOTH phi and mutual_info similarity metrics:
#   1. Co-occurrence statistics collection (shared between metrics)
#   2. Spectral clustering with co-occurrence affinity
#   3. AANet fitting for all clusters across k values
#   4. Near-vertex example collection
#
# PCA rank estimation is SKIPPED to save time.
#
# Run this on one GPU while run_full_real_data_pipeline_cooc_512.sh runs on another.

set -e  # Exit on error

export PYTHONPATH=.

OUTPUT_BASE="/workspace/outputs/real_data_cooc"
COOC_CACHE="${OUTPUT_BASE}/cooc_stats_768.npz"

echo "============================================================"
echo "CO-OCCURRENCE CLUSTERING PIPELINE (768 clusters)"
echo "============================================================"
echo "Metrics: phi, mutual_info"
echo "Output base: ${OUTPUT_BASE}"
echo "Co-occurrence cache: ${COOC_CACHE}"
echo "============================================================"

# ============================================================
# STAGE 1A: PHI COEFFICIENT CLUSTERING
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 1A: Running pipeline with PHI similarity (768 clusters)"
echo "============================================================"

python -u real_data_tests/analyze_real_saes.py \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --output_dir "${OUTPUT_BASE}/phi_768" \
    --n_clusters_list 768 \
    --latent_activity_threshold 0 \
    --activity_batch_size 32 \
    --activity_batches 4096 \
    --activity_seq_len 256 \
    --skip_pca \
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
    --sim_metric "phi" \
    --cooc_n_batches 1000 \
    --cooc_batch_size 32 \
    --cooc_seq_len 256 \
    --cooc_activation_threshold 1e-6 \
    --cooc_cache_path "${COOC_CACHE}" \
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
echo "PHI clustering + AANet fitting complete for 768 clusters"
echo ""

# ============================================================
# STAGE 1B: MUTUAL INFORMATION CLUSTERING
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 1B: Running pipeline with MUTUAL INFO similarity (768 clusters)"
echo "============================================================"
echo "Note: Co-occurrence stats will be loaded from cache"

python -u real_data_tests/analyze_real_saes.py \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --output_dir "${OUTPUT_BASE}/mutual_info_768" \
    --n_clusters_list 768 \
    --latent_activity_threshold 0 \
    --activity_batch_size 32 \
    --activity_batches 4096 \
    --activity_seq_len 256 \
    --skip_pca \
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
    --sim_metric "mutual_info" \
    --cooc_n_batches 1000 \
    --cooc_batch_size 32 \
    --cooc_seq_len 256 \
    --cooc_activation_threshold 1e-6 \
    --cooc_cache_path "${COOC_CACHE}" \
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
echo "MUTUAL INFO clustering + AANet fitting complete for 768 clusters"
echo ""

# ============================================================
# STAGE 2A: VERTEX SAMPLE COLLECTION FOR PHI CLUSTERS
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 2A: Collecting vertex samples for PHI clusters (768)"
echo "============================================================"

python -u real_data_tests/refit_selected_clusters.py \
    --skip_training \
    --stage1_models_dir "${OUTPUT_BASE}/phi_768" \
    --vertex_skip_docs 300000 \
    --n_clusters_list "768" \
    --csv_dir "${OUTPUT_BASE}/phi_768" \
    --save_dir "${OUTPUT_BASE}/phi_768_selected" \
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
    --max_inputs_per_cluster 10000000 \
    --vertex_save_interval 5000

echo ""
echo "PHI vertex sample collection complete"
echo ""

# ============================================================
# STAGE 2B: VERTEX SAMPLE COLLECTION FOR MUTUAL INFO CLUSTERS
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 2B: Collecting vertex samples for MUTUAL INFO clusters (768)"
echo "============================================================"

python -u real_data_tests/refit_selected_clusters.py \
    --skip_training \
    --stage1_models_dir "${OUTPUT_BASE}/mutual_info_768" \
    --vertex_skip_docs 300000 \
    --n_clusters_list "768" \
    --csv_dir "${OUTPUT_BASE}/mutual_info_768" \
    --save_dir "${OUTPUT_BASE}/mutual_info_768_selected" \
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
    --max_inputs_per_cluster 10000000 \
    --vertex_save_interval 5000

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE (768 clusters)"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  PHI clustering:        ${OUTPUT_BASE}/phi_768/"
echo "  PHI vertex samples:    ${OUTPUT_BASE}/phi_768_selected/"
echo "  MI clustering:         ${OUTPUT_BASE}/mutual_info_768/"
echo "  MI vertex samples:     ${OUTPUT_BASE}/mutual_info_768_selected/"
echo "  Co-occurrence cache:   ${COOC_CACHE}"
echo ""
echo "============================================================"
