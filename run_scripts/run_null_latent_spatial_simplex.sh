#!/bin/bash

# 2Ba (null control): Latent spatial structure in the simplex
#
# Same analysis as run_latent_spatial_simplex.sh but on null AANet clusters
# (random latent groupings with fitted AANets).
# Used to verify that null clusters do NOT show the vertex-localised latent
# activation patterns seen in real clusters.
#
# Priority null clusters: 512_138, 512_345, 768_310
#
# CPU-only. Run from project root:
#   ./run_scripts/run_null_latent_spatial_simplex.sh
#
# Prerequisites:
#   - null_simplex_samples/ must exist (from run_null_collect_simplex_samples.sh)
#   - selected_null_clusters/ must exist (for vertex_acts_dir)

set -e

export PYTHONPATH=.

SIMPLEX_SAMPLES_DIR="/workspace/outputs/null_simplex_samples"
VERTEX_ACTS_DIR="/workspace/outputs/selected_null_clusters"
OUTPUT_DIR="/workspace/outputs/validation/latent_spatial_null"

# Priority null clusters
CLUSTERS="512_138,512_345,768_310"

echo "============================================================"
echo "2Ba NULL CONTROL: LATENT SPATIAL SIMPLEX ANALYSIS"
echo "============================================================"
echo "Simplex samples dir: ${SIMPLEX_SAMPLES_DIR}"
echo "Vertex acts dir:     ${VERTEX_ACTS_DIR}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "Clusters:            ${CLUSTERS}"
echo ""

python -u validation/latent_spatial_simplex.py \
    --simplex_samples_dir "${SIMPLEX_SAMPLES_DIR}" \
    --vertex_acts_dir "${VERTEX_ACTS_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --clusters "${CLUSTERS}" \
    --heatmap_grid_size 100

echo ""
echo "============================================================"
echo "Done. Results in: ${OUTPUT_DIR}/"
echo "  - all_spatial_stats.json (combined stats)"
echo "  - cluster_*/  (per-cluster PNGs and JSON)"
echo "============================================================"
