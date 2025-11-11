#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="${ROOT_DIR}/outputs/reports/mess3/AAnet"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --process-config "/root/mess3_saes/process_configs.json"
    --process-config-name "single_mess3"
    --model-ckpt "/workspace/outputs/checkpoints/mess3/mess3_transformer.pt"
    --sae-root "/workspace/outputs/saes/mess3"
    --cluster-summary-dir "/workspace/outputs/reports/mess3/aanet_cluster_summaries"
    --cluster-summary-pattern "mess3_layer_{layer}_cluster_summary.json"
    --output-dir "/workspace/outputs/reports/mess3/AAnet"
    --d-model 64
    --n-heads 4
    --n-layers 3
    --n-ctx 10
    --d-head 32
    --act-fn "relu"
    --device "cuda"
    --layers 1 2
    --topk 3
    --batch-size 256
    --seq-len 10
    --num-batches 256
    --activation-threshold 0.01
    --max-samples-per-cluster 200000
    --min-cluster-samples 10000
    --sampling-seed 123
    --token-indices 4 8
    --k-values 2 3 4 5 6 7 8
    --aanet-epochs 100
    --aanet-batch-size 256
    --aanet-lr 0.001
    --aanet-weight-decay 0.0
    --aanet-layer-widths 32 16
    --aanet-simplex-scale 1.0
    --aanet-noise 0.05
    --aanet-noise-relative
    --aanet-gamma-reconstruction 1.0
    --aanet-gamma-archetypal 3.0
    --aanet-gamma-extrema 1.0
    --aanet-min-samples 10000
    --aanet-num-workers 0
    --aanet-seed 43
    --aanet-val-fraction 0.1
    --aanet-val-min-size 512
    --aanet-early-stop-patience 15
    --aanet-early-stop-delta 1e-5
    --aanet-lr-patience 8
    --aanet-lr-factor 0.5
    --aanet-grad-clip 1.0
    --aanet-restarts-no-extrema 2
    --extrema-enabled
    --extrema-knn 100
    --extrema-max-points 40000
    --extrema-seed 777
    --save-models
    --overwrite
)

printf 'Running mess3 AAnet experiment with args:\n'
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
