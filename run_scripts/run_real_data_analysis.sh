#!/bin/bash

# Run Real Data SAE Analysis with explicit default arguments

# Ensure we are in the project root (one level up from run_scripts if running from there, or current dir)
# Ideally run this from the project root: ./run_scripts/run_real_data_analysis.sh

export PYTHONPATH=.

python real_data_tests/analyze_real_saes.py \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_32k/average_l0_57" \
    --output_dir "/workspace/outputs/real_data_analysis" \
    --n_clusters_list 128 256 512 645 \
    --total_samples 25000 \
    --latent_activity_threshold 1e-5 \
    --device "cuda" \
    --seed 42 \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN}
