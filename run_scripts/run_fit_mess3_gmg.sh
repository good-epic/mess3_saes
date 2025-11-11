#!/usr/bin/bash
set -euo pipefail

START_TIME=$(date +%s)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ARGS=(
  --sae_folder "/workspace/outputs/saes/multipartite_004"
  --metrics_summary "/workspace/outputs/saes/multipartite_004/metrics_summary.json"
  --output_dir "/workspace/outputs/reports/multipartite_004"
  --model_ckpt "/workspace/outputs/checkpoints/multipartite_004/checkpoint_step_30001_best.pt"
  --device "cuda"
  --seed 43
  --process_config "${ROOT_DIR}/process_configs.json"
  --process_config_name "5xmess3_001"
  --d_model 128
  --n_heads 4
  --n_layers 3
  --n_ctx 32
  --d_head 32
  --act_fn "relu"
  --sim_metric "euclidean"
  --clustering_method "k_subspaces"
  --subspace_n_clusters 6
  --min_clusters 6
  --max_clusters 12
  --cosine_dedup_threshold 0.99
  --sae_type top_k
  --force_k 12
  --sample_sequences 1024
  --max_activations 50000
  --cluster_activation_threshold 1e-6
  --center_decoder_rows
  --latent_activity_threshold 0.001
  --latent_activation_eps 1e-6
  --activation_batches 8
  --refine_with_geometries
  --geo_include_circle
  --geo_filter_metrics gw_full
  --geo_sinkhorn_max_iter 5000
  --geo_sinkhorn_epsilon 0.2
  --geo_threshold_mode raw
  --geo_per_point_threshold 0.0001
  --sites layer_0 layer_1 layer_2
)

"${PYTHON_BIN}" -u "${ROOT_DIR}/fit_mess3_gmg.py" "${ARGS[@]}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds."



# START_TIME=$(date +%s)
# python -u fit_mess3_gmg.py \
#     --sae_folder "outputs/saes/multipartite_004" \
#     --metrics_summary "outputs/saes/multipartite_004/metrics_summary.json" \
#     --output_dir "outputs/reports/multipartite_004" \
#     --model_ckpt "outputs/checkpoints/multipartite_004/checkpoint_step_30001_best.pt" \
#     --device "cuda" \
#     --seed 43 \
#     --process_config "process_configs.json" \
#     --process_config_name "5xmess3_001" \
#     --d_model 128 \
#     --n_heads 4 \
#     --n_layers 3 \
#     --n_ctx 32 \
#     --d_head 32 \
#     --act_fn "relu" \
#     --sim_metric "cosine" \
#     --clustering_method "k_subspaces" \
#     --subspace_variance_threshold 0.85 \
#     --subspace_gap_threshold 2.0 \
#     --min_clusters 6 \
#     --max_clusters 12 \
#     --plot_eigengap \
#     --cosine_dedup_threshold 0.99 \
#     --sample_sequences 1024 \
#     --max_activations 50000 \
#     --cluster_activation_threshold 1e-6 \
#     --center_decoder_rows \
#     --latent_activity_threshold 0.01 \
#     --latent_activation_eps 1e-6 \
#     --activation_batches 8 \
#     --refine_with_geometries \
#     --geo_include_circle \
#     --geo_filter_metrics gw_full \
#     --geo_sinkhorn_max_iter 5000 \
#     --geo_sinkhorn_epsilon 0.2 \
#     --geo_threshold_mode raw \
#     --geo_per_point_threshold 0.0005
# END_TIME=$(date +%s)
# DURATION=$((END_TIME - START_TIME))
# echo "Runtime: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds."

#    --ensc_lambda1 0.001 \
#    --ensc_lambda2 0.0005 \
#    --sae_type "vanilla" \
#    --force_lambda 0.05

#    --subspace_n_clusters 6 \

#    --build_cluster_epdfs \
#    --epdf_sites layer_0 layer_1 layer_2 \
#    --epdf_plot_mode "both" \
#    --epdf_bandwidth "scott" \
#    --epdf_grid_size 200 \

#    --seed_with_beliefs \
#    --seed_lasso_cv 8 \
#    --seed_coef_threshold 1e-4 \
#    --seed_max_latents 40 \
#    --belief_ridge_alpha 1e-3 \
#    --skip_pca_plots \
#    --min_cluster_samples 512 \
#    --plot_max_points 2500 \
#    --pca_components 15
