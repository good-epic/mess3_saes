#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_NAME="3xmess3_2xtquant_003"
CHECKPOINT_DIR="${ROOT_DIR}/outputs/checkpoints/multipartite_003"
SAE_OUTPUT_DIR="${ROOT_DIR}/outputs/saes/multipartite_003_bsae"

D_MODEL=128
N_HEADS=4
N_LAYERS=3
N_CTX=16
D_HEAD=32
DICT_MUL=4

AR_LAMBDA_SPARSE=(0.005 0.01 0.015)
AR_LAMBDA_AR=(0.005 0.01 0.015)
AR_P=3
AR_BETA_SLOPE=2.0
AR_DELTA=0.25
AR_EPSILON=1e-5
AR_SPARSITY_MODE="l0"

echo "=== Training bSAEs for checkpoint ${CHECKPOINT_DIR} ==="

FINAL_CHECKPOINT=$(
"${PYTHON_BIN}" - <<'PY' "${CHECKPOINT_DIR}"
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
best = sorted(ckpt_dir.glob("checkpoint_step_*_best.pt"))
if best:
    print(best[-1])
else:
    finals = sorted(ckpt_dir.glob("checkpoint_step_*_final.pt"))
    if not finals:
        raise SystemExit(f"No checkpoint found in {ckpt_dir}")
    print(finals[-1])
PY
) || exit 1

echo "Using checkpoint: ${FINAL_CHECKPOINT}"
mkdir -p "${SAE_OUTPUT_DIR}"

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_saes.py" \
    --d_model "${D_MODEL}" \
    --n_heads "${N_HEADS}" \
    --n_layers "${N_LAYERS}" \
    --n_ctx "${N_CTX}" \
    --d_head "${D_HEAD}" \
    --dict_mul "${DICT_MUL}" \
    --device cuda \
    --act_fn relu \
    --input_unit_norm \
    --n_batches_to_dead 5 \
    --aux_penalty 0.03125 \
    --bandwidth 0.001 \
    --sae_batch_size 256 \
    --sae_steps 60000 \
    --sae_early_stopping_min_steps 15000 \
    --sae_early_stopping_patience 40 \
    --sae_early_stopping_delta 1e-7 \
    --sae_early_stopping_beta 0.995 \
    --sae_log_interval 500 \
    --sae_scheduler_final_ratio 0.1 \
    --sae_output_dir "${SAE_OUTPUT_DIR}" \
    --load_model "${FINAL_CHECKPOINT}" \
    --process_config "${ROOT_DIR}/process_configs.json" \
    --process_config_name "${CONFIG_NAME}" \
    --ar_lambda_sparse "${AR_LAMBDA_SPARSE[@]}" \
    --ar_lambda_ar "${AR_LAMBDA_AR[@]}" \
    --ar_cartesian_lambdas \
    --ar_p "${AR_P}" \
    --ar_beta_slope "${AR_BETA_SLOPE}" \
    --ar_delta "${AR_DELTA}" \
    --ar_epsilon "${AR_EPSILON}" \
    --ar_sparsity_mode "${AR_SPARSITY_MODE}"

echo "bSAE training complete. Results saved to ${SAE_OUTPUT_DIR}"
