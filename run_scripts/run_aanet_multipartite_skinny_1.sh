#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="${ROOT_DIR}/outputs/reports/multipartite_003e/AAnet"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --process-config "${ROOT_DIR}/process_configs.json"
    --process-config-name "3xmess3_2xtquant_003"
    --model-ckpt "/workspace/outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt"
    --sae-root "/workspace/outputs/saes/multipartite_003e"
    --cluster-summary-dir "/workspace/outputs/reports/multipartite_003e"
    --cluster-summary-pattern "top_r2_run_layer_{layer}_cluster_summary.json"
    --output-dir "/workspace/outputs/reports/multipartite_003e/AAnet_filtered_skinny_1"
    --d-model 128
    --n-heads 4
    --n-layers 3
    --n-ctx 16
    --d-head 32
    --act-fn "relu"
    --device "cuda"
    --layers 1 2
    --topk 12
    --batch-size 256
    --seq-len 16
    --num-batches 512
    --activation-threshold 0.01
    --max-samples-per-cluster 1000000
    --min-cluster-samples 100000
    --sampling-seed 123
    --token-indices 4 9 14
    --k-values 2 3 4 5 6 7 8
    --aanet-epochs 100
    --aanet-batch-size 256
    --aanet-lr 0.001
    --aanet-weight-decay 0.0
    --aanet-layer-widths 64 32
    --aanet-simplex-scale 1.0
    --aanet-noise 0.05
    --aanet-noise-relative
    --aanet-gamma-reconstruction 1.0
    --aanet-gamma-archetypal 4.0
    --aanet-gamma-extrema 2.0
    --aanet-min-samples 35000
    --aanet-num-workers 0
    --aanet-seed 43
    --aanet-val-fraction 0.1
    --aanet-val-min-size 1024
    --aanet-early-stop-patience 15
    --aanet-early-stop-delta 1e-6
    --aanet-lr-patience 10
    --aanet-lr-factor 0.5
    --aanet-grad-clip 1.0
    --aanet-restarts-no-extrema 3
    --extrema-enabled
    --extrema-knn 200
    --extrema-max-points 50000
    --extrema-seed 431
    --save-models
    --overwrite
)

printf 'Running AAnet experiment with args:\n'
first_arg=true
for arg in "${ARGS[@]}"; do
    if [[ $arg == --* && $first_arg == false ]]; then
        printf '\n'
    fi
    printf '%q ' "$arg"
    first_arg=false
done
printf '\n\n'

"${PYTHON_BIN}" -u "${ROOT_DIR}/run_aanet_belief_geometry.py" "${ARGS[@]}"
