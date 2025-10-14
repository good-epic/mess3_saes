#!/usr/bin/bash

set -euo pipefail

#export CUDA_VISIBLE_DEVICES=0
#export JAX_PLATFORMS=cpu
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.90

CHECKPOINT_DIR="/workspace/outputs/checkpoints/multipartite_003a"

# python -u train_simplexity_3xmess3_2xtquant.py \
#     --d_model 128 \
#     --n_heads 4 \
#     --n_layers 3 \
#     --n_ctx 16 \
#     --d_head 32 \
#     --checkpoint_path "${CHECKPOINT_DIR}" \
#     --fig_out_dir /workspace/outputs/reports/multipartite_003a \
#     --num_steps 100000 \
#     --act_fn relu \
#     --device cuda \
#     --batch_size 4096 \
#     --pct_var_explained 0.99 \
#     --process_config "process_configs.json" \
#     --process_config_name "3xmess3_2xtquant_003" \
#     --early_stopping_patience 30 \
#     --early_stopping_delta 5e-5 \
#     --early_stopping_min_steps 6000


FINAL_CHECKPOINT=$(
python - <<'PY' "${CHECKPOINT_DIR}"
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
best_matches = sorted(ckpt_dir.glob("checkpoint_step_*_best.pt"))
if best_matches:
    print(best_matches[-1])
else:
    matches = sorted(ckpt_dir.glob("checkpoint_step_*_final.pt"))
    if not matches:
        raise SystemExit(f"No *_final.pt checkpoint found in {ckpt_dir}")
    print(matches[-1])
PY
) || exit 1

echo "Using final checkpoint: ${FINAL_CHECKPOINT}"

python -u train_mess3_and_saes.py \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 3 \
    --n_ctx 16 \
    --d_head 32 \
    --act_fn relu \
    --device cuda \
    --dict_mul 4 \
    --l1_coeff_seq 0.02 0.03 0.04 0.06 0.07 0.08 0.09 0.125 \
    --input_unit_norm True \
    --no_input_unit_norm False \
    --n_batches_to_dead 5 \
    --aux_penalty 0.03125 \
    --bandwidth 0.001 \
    --sae_batch_size 64 \
    --sae_output_dir /workspace/outputs/saes/multipartite_003e \
    --load_model "${FINAL_CHECKPOINT}" \
    --process_config "process_configs.json" \
    --process_config_name "3xmess3_2xtquant_003" \
    --sae_steps 50000 \
    --sae_early_stopping_min_steps 15000 \
    --sae_early_stopping_patience 30 \
    --sae_early_stopping_delta 1e-7 \
    --sae_early_stopping_beta 0.995 \
    --sae_log_interval 250 \
    --sae_scheduler_final_ratio 0.15

#    --l1_coeff_seq 0.001 0.005 0.01 0.05 0.1 0.15 \
#    --k 3 4 5 6 7 8 10 12 14 16 19 22 25 \
