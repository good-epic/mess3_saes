#!/usr/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

START_TIME=$(date +%s)

ARGS=(
    --sae_folder "${ROOT_DIR}/outputs/saes/multipartite_003e"
    --metrics_summary "${ROOT_DIR}/outputs/saes/multipartite_003e/metrics_summary.json"
    --model_ckpt "${ROOT_DIR}/outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt"
    --output_path "${ROOT_DIR}/outputs/reports/multipartite_003e/ground_truth_diagnostics/"
    --device cuda
    --process_config "${ROOT_DIR}/process_configs.json"
    --process_config_name 3xmess3_2xtquant_003
    --d_model 128
    --n_heads 4
    --n_layers 3
    --n_ctx 16
    --d_head 32
    --act_fn relu
    --top_k_vals 12 14
    --sites layer_0 layer_1 layer_2
    --transformer_batch_size 1024
    --transformer_total_sample_size 25000
    --sample_seq_len 16
    --activation_eps 1e-6
    --seed 43
    --act_rate_threshold 0.00025
    --lasso_lambdas 6e-3
    --lasso_max_iter 10000
    --elasticnet_l1_ratio 0.9
    --elasticnet_stability_fraction 0.5
    --elasticnet_stability_runs 25
    --elasticnet_selection_threshold 0.5
    --r2_threshold 0.005
    --format csv
)

printf 'Running latent_geometry_create_metrics with args:\n'
first_arg=true
for arg in "${ARGS[@]}"; do
    if [[ $arg == -* && $first_arg == false ]]; then
        printf '\n'
    fi
    printf '%q ' "$arg"
    first_arg=false
done
printf '\n'
"${PYTHON_BIN}" -u "${ROOT_DIR}/latent_geometry_create_metrics.py" "${ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
printf 'Finished in %02dh:%02dm:%02ds\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
