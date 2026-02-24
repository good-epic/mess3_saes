#!/bin/bash

# Null Cluster Baseline — Stage 1 (768 clusters)
#
# Generates a random partition of SAE latents matching the real 768-cluster
# size distribution, then runs the full AANet training + elbow analysis
# pipeline on those null clusters.
#
# Run from project root: ./run_scripts/run_null_stage1_768.sh
#
# After this completes:
#   1. Download outputs/null_clusters/clusters_768/consolidated_metrics_n768.csv
#   2. Review elbow plots: PYTHONPATH=. python scripts/plot_elbow_browser.py
#        --csv_dir outputs/null_clusters --n_clusters 768
#   3. Select clusters and fill in run_null_refit_768.sh, then run it

set -e

export PYTHONPATH=.

NULL_DIR="/workspace/outputs/null_clusters"
LABELS_FILE="${NULL_DIR}/null_labels_n768.npy"
SOURCE_CSV="/workspace/outputs/real_data_analysis_canonical/clusters_768/consolidated_metrics_n768.csv"

echo "============================================================"
echo "NULL CLUSTER BASELINE — STAGE 1 (768 clusters)"
echo "============================================================"
echo "Source CSV: ${SOURCE_CSV}"
echo "Labels file: ${LABELS_FILE}"
echo "Output dir: ${NULL_DIR}"
echo "============================================================"

# ============================================================
# STEP 1: Generate null cluster labels
# ============================================================
echo ""
echo "Step 1: Generating null cluster labels..."

mkdir -p "${NULL_DIR}"

python -u real_data_tests/generate_null_clusters.py \
    --source_csv "${SOURCE_CSV}" \
    --output_file "${LABELS_FILE}" \
    --seed 42

echo "Null cluster labels generated."

# ============================================================
# STEP 2: AANet training + elbow analysis on null clusters
# ============================================================
echo ""
echo "Step 2: Running AANet training on null clusters (768)..."

python -u real_data_tests/analyze_real_saes.py \
    --cluster_labels_file "${LABELS_FILE}" \
    --n_clusters_list 768 \
    --output_dir "${NULL_DIR}" \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res-canonical" \
    --sae_id "layer_20/width_16k/canonical" \
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
echo "Stage 1 complete for null 768 clusters"
echo "============================================================"
echo ""
echo "Outputs: ${NULL_DIR}/clusters_768/"
echo "  consolidated_metrics_n768.csv"
echo "  aanet_cluster_*_k*.pt"
echo ""
echo "Next: review elbow plots, then run run_null_refit_768.sh"
echo "============================================================"
