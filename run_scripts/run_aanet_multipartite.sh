#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="/workspace/outputs/reports/multipartite_004/AAnet"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --process-config "/root/mess3_saes/process_configs.json"
    --process-config-name "5xmess3_001"
    --model-ckpt "/workspace/outputs/checkpoints/multipartite_004/checkpoint_step_30001_best.pt"
    --sae-root "/workspace/outputs/saes/multipartite_004"
    --cluster-summary-dir "/workspace/outputs/reports/multipartite_004"
    --cluster-summary-pattern "layer_{layer}_cluster_summary.json"
    --output-dir "/workspace/outputs/reports/multipartite_004/AAnet"
    --d-model 128
    --n-heads 4
    --n-layers 3
    --n-ctx 32
    --d-head 32
    --act-fn "relu"
    --device "cuda"
    --layers 0 1 2
    --topk 12
    --batch-size 256
    --seq-len 32
    --num-batches 512
    --activation-threshold 1e-6
    --max-samples-per-cluster 1000000
    --min-cluster-samples 100000
    --sampling-seed 123
    --token-indices 14 19 24 29
    --k-values 2 3 4 5 6 7 8
    --aanet-epochs 100
    --aanet-batch-size 256
    --aanet-lr 0.001
    --aanet-weight-decay 0.0
    --aanet-layer-widths 256 128
    --aanet-simplex-scale 1.0
    --aanet-noise 0.05
    --aanet-noise-relative
    --aanet-gamma-reconstruction 1.0
    --aanet-gamma-archetypal 3.0
    --aanet-gamma-extrema 1.0
    --aanet-min-samples 50000
    --aanet-num-workers 0
    --aanet-seed 43
    --aanet-val-fraction 0.1
    --aanet-val-min-size 1024
    --aanet-early-stop-patience 20
    --aanet-early-stop-delta 1e-6
    --aanet-lr-patience 10
    --aanet-lr-factor 0.5
    --aanet-grad-clip 1.0
    --aanet-restarts-no-extrema 3
    --extrema-enabled
    --extrema-knn 100
    --extrema-max-points 20000
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
