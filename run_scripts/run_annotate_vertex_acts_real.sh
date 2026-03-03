#!/bin/bash

# Annotate real priority cluster vertex_samples.jsonl with latent_acts + barycentric_coords
#
# Post-processing pass: re-runs each saved 256-token chunk through
# model + SAE + AANet at each trigger position. Adds two fields to
# every record:
#   latent_acts      — cluster activation vector (one per trigger)
#   barycentric_coords — full k-dim simplex coords (one per trigger)
#
# Writes vertex_samples_with_acts.jsonl alongside existing files.
#
# GPU required. Run from project root:
#   ./run_scripts/run_annotate_vertex_acts_real.sh
#
# Prerequisites:
#   - vertex_samples.jsonl files in REAL_SOURCE_DIR
#   - AANet .pt models in CSV_DIR (real_data_analysis_canonical)

set -e

export PYTHONPATH=.

REAL_SOURCE_DIR="/workspace/outputs/selected_clusters_broad_2"
CSV_DIR="/workspace/outputs/real_data_analysis_canonical"

# All 13 priority clusters (4 HIGH + 9 MEDIUM confidence)
REAL_CLUSTERS="512_17,512_22,512_67,512_181,512_229,512_261,512_471,512_504,\
768_140,768_210,768_306,768_581,768_596"

echo "============================================================"
echo "ANNOTATE REAL VERTEX SAMPLES WITH LATENT ACTS"
echo "============================================================"

python -u real_data_tests/annotate_vertex_acts.py \
    --source_dir "${REAL_SOURCE_DIR}" \
    --csv_dir "${CSV_DIR}" \
    --output_dir "${REAL_SOURCE_DIR}" \
    --clusters "${REAL_CLUSTERS}" \
    --max_samples_per_vertex 500 \
    --batch_size 16 \
    --model_name "gemma-2-9b" \
    --sae_release "gemma-scope-9b-pt-res" \
    --sae_id "layer_20/width_16k/average_l0_68" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token "${HF_TOKEN}" \
    --aanet_layer_widths 64 32 \
    --aanet_simplex_scale 1.0 \
    --aanet_noise 0.05

echo ""
echo "============================================================"
echo "Done. Output: *_vertex_samples_with_acts.jsonl"
echo "  Real clusters: ${REAL_SOURCE_DIR}/n{N}/"
echo "============================================================"
