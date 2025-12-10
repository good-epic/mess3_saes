#!/bin/bash

# Run a quick test of the refactored script
export PYTHONPATH=$PYTHONPATH:.
python real_data_tests/analyze_real_saes.py \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_32k/average_l0_57" \
    --model_name "gemma-2-9b" \
    --output_dir "outputs/test_refactor" \
    --n_clusters_list 128 \
    --total_samples 1000 \
    --aanet_streaming_steps 10 \
    --aanet_warmup_steps 5 \
    --aanet_k_min 2 \
    --aanet_k_max 2 \
    --aanet_batch_size 32 \
    --device "cuda"
