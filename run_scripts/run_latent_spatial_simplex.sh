#!/bin/bash

# 2Ba: Latent spatial structure in the simplex
#
# CPU-only analysis. Reads simplex samples produced by collect_simplex_samples.py
# (which must have been run with the updated code that saves latent_acts).
# Computes activation-weighted centroid and variance for each cluster latent,
# and renders simplex heatmaps + centroid scatter plots for K=3 clusters.
#
# Run from project root: ./run_scripts/run_latent_spatial_simplex.sh
#
# Prerequisite: run_collect_simplex_samples.sh must have completed.

set -e

export PYTHONPATH=.

SIMPLEX_SAMPLES_DIR="outputs/simplex_samples"
OUTPUT_DIR="outputs/validation/latent_spatial"

# All priority clusters (high + medium confidence, broad_2)
CLUSTERS="512_17,512_22,512_67,512_181,512_229,512_261,512_471,512_504,\
768_140,768_210,768_306,768_581,768_596"

echo "============================================================"
echo "2Ba: LATENT SPATIAL SIMPLEX ANALYSIS"
echo "============================================================"
echo "Simplex samples dir: ${SIMPLEX_SAMPLES_DIR}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "Clusters:            ${CLUSTERS}"
echo ""

python -u validation/latent_spatial_simplex.py \
    --simplex_samples_dir "${SIMPLEX_SAMPLES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${CLUSTERS}" \
    --heatmap_grid_size 100

echo ""
echo "============================================================"
echo "Done. Results in: ${OUTPUT_DIR}/"
echo "  - all_spatial_stats.json (combined stats)"
echo "  - cluster_*/  (per-cluster PNGs and JSON)"
echo "============================================================"
