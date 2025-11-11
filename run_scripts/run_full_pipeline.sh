#!/usr/bin/bash

set -euo pipefail

#export CUDA_VISIBLE_DEVICES=0
#export JAX_PLATFORMS=cpu
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.90

CHECKPOINT_DIR="/workspace/outputs/checkpoints/mess3_x_0.05_a_0.05"
D_MODEL=64
N_HEADS=4
N_LAYERS=3
N_CTX=16
D_HEAD=32
DICT_MUL=4
SAE_PRIMARY_TOPK=3

python -u train_transformer.py \
    --d_model "${D_MODEL}" \
    --n_heads "${N_HEADS}" \
    --n_layers "${N_LAYERS}" \
    --n_ctx "${N_CTX}" \
    --d_head "${D_HEAD}" \
    --checkpoint_path "${CHECKPOINT_DIR}" \
    --fig_out_dir /workspace/outputs/reports/mess3_x_0.05_a_0.05 \
    --num_steps 100000 \
    --act_fn relu \
    --device cuda \
    --batch_size 4096 \
    --pct_var_explained 0.99 \
    --process_config "process_configs.json" \
    --process_config_name "mess3_x_0.05_a_0.05" \
    --early_stopping_patience 30 \
    --early_stopping_delta 5e-5 \
    --early_stopping_min_steps 50000


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

python -u train_saes.py \
    --d_model "${D_MODEL}" \
    --n_heads "${N_HEADS}" \
    --n_layers "${N_LAYERS}" \
    --n_ctx "${N_CTX}" \
    --d_head "${D_HEAD}" \
    --act_fn relu \
    --device cuda \
    --dict_mul "${DICT_MUL}" \
    --k 3 4 \
    --input_unit_norm True \
    --no_input_unit_norm False \
    --n_batches_to_dead 5 \
    --aux_penalty 0.03125 \
    --bandwidth 0.001 \
    --sae_batch_size 64 \
    --sae_output_dir /workspace/outputs/saes/mess3_x_0.05_a_0.05 \
    --load_model "${FINAL_CHECKPOINT}" \
    --process_config "process_configs.json" \
    --process_config_name "mess3_x_0.05_a_0.05" \
    --sae_steps 50000 \
    --sae_early_stopping_min_steps 15000 \
    --sae_early_stopping_patience 30 \
    --sae_early_stopping_delta 1e-7 \
    --sae_early_stopping_beta 0.995 \
    --sae_log_interval 250 \
    --sae_scheduler_final_ratio 0.15

#    --l1_coeff_seq 0.001 0.005 0.01 0.05 0.1 0.15 \
#    --k 3 4 5 6 7 8 10 12 14 16 19 22 25 \

CLUSTER_OUTPUT_DIR="/workspace/outputs/reports/mess3_x_0.05_a_0.05/aanet_cluster_summaries"
mkdir -p "${CLUSTER_OUTPUT_DIR}"
N_LATENTS=$((D_MODEL * DICT_MUL))
for layer in 0 1 2; do
    python - <<'PY' "${CLUSTER_OUTPUT_DIR}" "${layer}" "${N_LATENTS}" "${SAE_PRIMARY_TOPK}"
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
layer = int(sys.argv[2])
n_latents = int(sys.argv[3])
sae_top_k = int(sys.argv[4])
component_name = "mess3"

cluster_summary = {
    "layer": layer,
    "sae_top_k": sae_top_k,
    "clusters": {
        "0": {
            "latent_indices": list(range(n_latents)),
            "activation_fraction": 1.0,
            "is_noise": False,
            "metadata": {
                "component_name": component_name,
            },
        }
    },
    "component_assignment_hard": {
        "assignments": {component_name: 0},
        "noise_clusters": [],
        "component_order": [component_name],
    },
    "latent_cluster_assignments": {str(i): 0 for i in range(n_latents)},
}

out_path = out_dir / f"mess3_layer_{layer}_cluster_summary.json"
out_path.write_text(json.dumps(cluster_summary, indent=2))
print(f"Wrote {out_path}")
PY
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="${ROOT_DIR}/outputs/reports/mess3_x_0.05_a_0.05/AAnet"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --process-config "/root/mess3_saes/process_configs.json"
    --process-config-name "mess3_x_0.05_a_0.05"
    --model-ckpt ${FINAL_CHECKPOINT}
    --sae-root "/workspace/outputs/saes/mess3_x_0.05_a_0.05"
    --cluster-summary-dir "/workspace/outputs/reports/mess3_x_0.05_a_0.05/aanet_cluster_summaries"
    --cluster-summary-pattern "mess3_layer_{layer}_cluster_summary.json"
    --output-dir "/workspace/outputs/reports/mess3_x_0.05_a_0.05/AAnet"
    --d-model 64
    --n-heads 4
    --n-layers 3
    --n-ctx 10
    --d-head 32
    --act-fn "relu"
    --device "cuda"
    --layers 0 1 2
    --topk 3
    --batch-size 256
    --seq-len 10
    --num-batches 256
    --activation-threshold 0.01
    --max-samples-per-cluster 200000
    --min-cluster-samples 10000
    --sampling-seed 123
    --token-indices 4 8
    --k-values 2 3 4 5 6 7
    --aanet-epochs 100
    --aanet-batch-size 256
    --aanet-lr 0.001
    --aanet-weight-decay 0.0
    --aanet-layer-widths 32 16 12
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
