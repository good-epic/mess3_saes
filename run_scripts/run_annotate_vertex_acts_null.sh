#!/bin/bash

# Annotate null cluster vertex_samples.jsonl with latent_acts + barycentric_coords
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
#   ./run_scripts/run_annotate_vertex_acts_null.sh
#
# Prerequisites:
#   - vertex_samples.jsonl files in NULL_SOURCE_DIR
#   - AANet .pt models in NULL_CSV_DIR (null_clusters)

set -e

export PYTHONPATH=.

NULL_SOURCE_DIR="outputs/selected_null_clusters"
NULL_CSV_DIR="outputs/null_clusters"

# Priority null clusters
NULL_CLUSTERS="512_138,512_345,768_310"

echo "============================================================"
echo "ANNOTATE NULL VERTEX SAMPLES WITH LATENT ACTS"
echo "============================================================"

python -u real_data_tests/annotate_vertex_acts.py \
    --source_dir "${NULL_SOURCE_DIR}" \
    --csv_dir "${NULL_CSV_DIR}" \
    --output_dir "${NULL_SOURCE_DIR}" \
    --clusters "${NULL_CLUSTERS}" \
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
echo "  Null clusters: ${NULL_SOURCE_DIR}/n{N}/"
echo "============================================================"
