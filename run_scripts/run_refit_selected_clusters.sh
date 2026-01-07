#!/bin/bash

# Refit AANet models for selected clusters
# Run from project root: ./run_scripts/run_refit_selected_clusters.sh

export PYTHONPATH=.

python -u real_data_tests/refit_selected_clusters.py \
    --n_clusters_list "128,256,512,768" \
    --corrected_csv_dir "outputs/real_data_analysis_canonical" \
    --save_dir "/workspace/outputs/selected_clusters_canonical" \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res-canonical" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN} \
    --dataset_name "monology/pile-uncopyrighted" \
    --dataset_config "default" \
    --dataset_split "train" \
    --activity_batch_size 32 \
    --activity_seq_len 128 \
    --seed 42 \
    --aanet_streaming_steps 3000 \
    --aanet_batch_size 128 \
    --aanet_lr 0.0025 \
    --aanet_weight_decay 1e-5 \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05 \
    --aanet_gamma_reconstruction 1.0 \
    --aanet_gamma_archetypal 4.0 \
    --aanet_gamma_extrema 2.0 \
    --aanet_grad_clip 1.0 \
    --active_threshold 0.0 \
    --extrema_enabled \
    --extrema_knn 150 \
    --extrema_warmup_samples 10000 \
    --collect_vertex_samples \
    --samples_per_vertex 1000 \
    --vertex_distance_threshold 0.1 \
    --min_vertex_ratio 0.1 \
    --vertex_search_batch_size 32 \
    --concurrent_aanets 5 \
    --max_inputs_per_cluster 1000000 \
    --vertex_save_interval 10000
