#!/usr/bin/bash

python -u mp_exploratory_analysis.py \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 3 \
    --n_ctx 16 \
    --d_head 32 \
    --act_fn relu \
    --device cuda \
    --dict_mul 4 \
    --l1_coeff_seq 0.001 0.005 0.01 0.05 0.1 0.15 \
    --k 3 4 5 6 7 8 10 12 14 16 19 22 25 \
    --input_unit_norm \
    --n_batches_to_dead 5 \
    --aux_penalty 0.03125 \
    --bandwidth 0.001 \
    --analysis_batch_size 8192 \
    --linear_prediction_layer "layer_2" \
    --load_model "outputs/checkpoints/multipartite_002/checkpoint_step_125000.pt" \
    --fig_out_dir "outputs/reports/multipartite_002" \
    --process_config "process_configs.json" \
    --process_config_name "3xmess3_2xtquant_002" \
    --build_epdfs \
    --epdf_cache_dir outputs/reports/multipartite_002/epdfs/cache \
    --epdf_use_cache \
    --epdf_max_points 2000 \
    --sae_folder "outputs/saes/multipartite_002"
