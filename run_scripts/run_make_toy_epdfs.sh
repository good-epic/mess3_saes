#!/usr/bin/env bash
# Generate per-cluster EPDF figures for multipartite_003e (layer 1, TopK k=12).
# Uses imshow+pixel-grid KDE (triangle for Mess3, disk for Tom Quantum).
# Run from project root: bash run_scripts/run_make_toy_epdfs.sh

set -euo pipefail
export PYTHONPATH=.
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu

python -u make_toy_epdfs.py \
    --cluster_summary outputs/reports/multipartite_003e/top_r2_run_layer_1_cluster_summary.json \
    --model_ckpt     outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt \
    --sae_path       outputs/saes/multipartite_003e/layer_1_top_k_k12.pt \
    --output_dir     outputs/reports/multipartite_003e/cluster_epdfs \
    --n_sequences    10000 \
    --batch_size     512 \
    --grid_size      80 \
    --base_opacity   0.55 \
    --dpi            150 \
    --ext            png

echo ""
echo "Done. Figures in outputs/reports/multipartite_003e/cluster_epdfs/"
echo "  cluster_*_all.png        — all-latents overlay per cluster"
echo "  cluster_*/latent_*.png   — solo per-latent figures"
