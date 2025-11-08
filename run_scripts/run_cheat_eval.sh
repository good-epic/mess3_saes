#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

START_TIME=$(date +%s)

ARGS=(
    --sae_folder "${ROOT_DIR}/outputs/saes/multipartite_003e"
    --model_ckpt "${ROOT_DIR}/outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt"
    --output_path "${ROOT_DIR}/outputs/reports/multipartite_003e/cheat_eval"
    --metrics_summary "${ROOT_DIR}/outputs/saes/multipartite_003e/metrics_summary.json"
    --process_config "${ROOT_DIR}/process_configs.json"
    --process_config_name 3xmess3_2xtquant_003
    --sites layer_1 layer_2
    --top_k_vals 12
    --eval_top_k 12
    --r2_cutoff 0.01
    --ridge_alpha 1e-3
    --device cuda
    --save_assignments_csv
    --geo_simplex_k_min 1
    --geo_simplex_k_max 7
    --geo_gw_epsilon 0.1
    --geo_sinkhorn_epsilon 0.1
    --top_r2_summary_dir "${ROOT_DIR}/outputs/reports/multipartite_003e"
    --top_r2_summary_pattern "top_r2_run_layer_{layer}_cluster_summary.json"
    --top_r2_compare_k 12
)
#    --compute_geometry

printf 'Running latent_geometry_eval_clusters with args:\n'
first_arg=true
for arg in "${ARGS[@]}"; do
    if [[ $arg == -* && $first_arg == false ]]; then
        printf '\n'
    fi
    printf '%q ' "$arg"
    first_arg=false
done
printf '\n'
"${PYTHON_BIN}" -u "${ROOT_DIR}/latent_geometry_eval_clusters.py" "${ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
printf 'Finished in %02dh:%02dm:%02ds\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
