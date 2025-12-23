#!/bin/bash

# Run Real Data SAE Analysis with explicit default arguments

# Ensure we are in the project root (one level up from run_scripts if running from there, or current dir)
# Ideally run this from the project root: ./run_scripts/run_real_data_analysis.sh

export PYTHONPATH=.

#    --sae_release "gemma-scope-9b-pt-res-canonical" \
#    --sae_id "layer_20/width_32k/average_l0_57" \
python -u real_data_tests/analyze_real_saes.py \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --output_dir "/workspace/outputs/real_data_analysis" \
    --n_clusters_list 128 256 512 768 \
    --total_samples 25000 \
    --latent_activity_threshold 0 \
    --activity_batch_size 32 \
    --activity_batches 4096 \
    --activity_seq_len 128 \
    --device "cuda" \
    --seed 42 \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN} \
    --dataset_name "monology/pile-uncopyrighted" \
    --dataset_config "default" \
    --dataset_split "train" \
    --subspace_variance_threshold 0.95 \
    --subspace_gap_threshold 2.0 \
    --aanet_batch_size 128 \
    --aanet_streaming_steps 2000 \
    --aanet_warmup_steps 100 \
    --aanet_warmup_cluster_chunk_size 16 \
    --aanet_prefetch_size 1024 \
    --aanet_sequential_k \
    --aanet_lr 0.0025 \
    --aanet_weight_decay 1e-5 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05 \
    --aanet_noise_relative \
    --aanet_gamma_reconstruction 1.0 \
    --aanet_gamma_archetypal 4.0 \
    --aanet_gamma_extrema 2.0 \
    --aanet_num_workers 0 \
    --aanet_seed 43 \
    --aanet_val_fraction 0.1 \
    --aanet_val_min_size 1024 \
    --aanet_early_stop_patience 20 \
    --aanet_early_stop_delta 1e-6 \
    --aanet_lr_patience 5 \
    --aanet_lr_factor 0.5 \
    --aanet_grad_clip 1.0 \
    --aanet_restarts_no_extrema 1 \
    --extrema_enabled \
    --extrema_knn 150 \
    --extrema_max_points 30000 \
    --extrema_pca 0.95 \
    --extrema_seed 431
