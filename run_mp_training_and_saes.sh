#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90

python -u train_simplexity_3xmess3_2xtquant.py \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 3 \
    --n_ctx 16 \
    --d_head 32 \
    --n_mess3 3 \
    --n_tom_quantum 2 \
    --checkpoint_path /workspace/outputs/checkpoints/multipartite_002 \
    --fig_out_dir /workspace/outputs/reports/multipartite_002 \
    --num_steps 400000 \
    --act_fn relu \
    --device cuda \
    --batch_size 4096 \
    --pct_var_explained 0.99

python -u train_mess3_and_saes.py \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 3 \
    --n_ctx 16 \
    --d_head 32 \
    --act_fn relu \
    --device cuda \
    --dict_mul 4 \
    --k 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --input_unit_norm True \
    --no_input_unit_norm False \
    --n_batches_to_dead 5 \
    --aux_penalty 0.03125 \
    --bandwidth 0.001 \
    --sae_steps 10000 \
    --sae_batch_size 2048 \
    --sae_output_dir /workspace/outputs/saes/multipartite_002 \
    --load_model /workspace/outputs/checkpoints/multipartite_001/checkpoint_step_600000_final.pt \
    --process_preset 3xmess3_2xtquant
