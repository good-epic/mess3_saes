#!/bin/bash

# Null Baseline Validation for AANet Simplex Fits
# Run from project root: ./run_scripts/run_null_baselines.sh
#
# TWO MODES:
#
# MODE 1: Build activation buffer (requires GPU + model loaded)
#   Streams data through Gemma-2-9B + SAE, saves activations to disk
#
# MODE 2: Run baselines (requires buffer on disk, GPU for AANet training)
#   Trains AANets on random latent subsets, compares to real clusters

export PYTHONPATH=.

# ============================================================
# MODE 1: Build buffer only
# ============================================================
# Uncomment to run just the buffer building step:
#
# python -u validation/null_baselines.py --build_buffer \
#     --model_name "gemma-2-9b" \
#     --sae_release "gemma-scope-9b-pt-res-canonical" \
#     --sae_id "layer_20/width_16k/average_l0_68" \
#     --buffer_size 100000 \
#     --batch_size 32 \
#     --seq_len 256 \
#     --output_dir "outputs/validation/null_baselines" \
#     --device "cuda" \
#     --cache_dir "/workspace/hf_cache" \
#     --hf_token ${HF_TOKEN} \
#     --hf_dataset "HuggingFaceFW/fineweb" \
#     --hf_subset_name "sample-10BT" \
#     --dataset_split "train" \
#     --prefetch_size 1024 \
#     --shuffle_buffer_size 50000 \
#     --max_doc_tokens 3000 \
#     --seed 42

# ============================================================
# MODE 2: Run baselines only (requires buffer already built)
# ============================================================
# Uncomment to run just the baseline comparisons:
#
# python -u validation/null_baselines.py --run_baselines \
#     --buffer_path "outputs/validation/null_baselines/activation_buffer.pt" \
#     --csv_dir "outputs/real_data_analysis_canonical" \
#     --n_baselines 20 \
#     --max_steps 3000 \
#     --train_batch_size 128 \
#     --output_dir "outputs/validation/null_baselines" \
#     --device "cuda" \
#     --seed 42

# ============================================================
# BOTH: Build buffer + run baselines (default)
# ============================================================
python -u validation/null_baselines.py \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res-canonical" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --buffer_size 100000 \
    --batch_size 32 \
    --seq_len 256 \
    --csv_dir "outputs/real_data_analysis_canonical" \
    --n_baselines 20 \
    --max_steps 3000 \
    --train_batch_size 128 \
    --output_dir "outputs/validation/null_baselines" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN} \
    --hf_dataset "HuggingFaceFW/fineweb" \
    --hf_subset_name "sample-10BT" \
    --dataset_split "train" \
    --prefetch_size 1024 \
    --shuffle_buffer_size 50000 \
    --max_doc_tokens 3000 \
    --seed 42

echo ""
echo "============================================================"
echo "Null Baseline Validation Complete!"
echo "============================================================"
echo "Results saved to: outputs/validation/null_baselines/"
echo "============================================================"
