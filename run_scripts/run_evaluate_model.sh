#!/usr/bin/bash

python -u evaluate_model_accuracy.py \
    --model_ckpt "outputs/checkpoints/multipartite_002/checkpoint_step_125000.pt" \
    --process_config "process_configs.json" \
    --process_config_name "3xmess3_2xtquant_002" \
    --output_dir "outputs/model_eval/multipartite_002" \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 3 \
    --n_ctx 16 \
    --d_head 32 \
    --act_fn "relu" \
    --batch_size 256 \
    --n_batches 100 \
    --seq_len 16 \
    --seed 43 \
    --device "cuda"
