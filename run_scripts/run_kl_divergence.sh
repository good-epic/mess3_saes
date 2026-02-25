#!/bin/bash

# KL divergence distributions across the simplex (Validation 2A)
#
# Computes three distributions of pairwise divergences per cluster:
#   1. Same-vertex pairs
#   2. Cross-vertex pairs
#   3. Within-simplex pairs (using collected simplex samples)
#
# Reports sym_kl and JS divergence, plus head/tail diagnostic.
# Requires GPU (model forward passes).
#
# Run AFTER run_collect_simplex_samples.sh
# Run from project root: ./run_scripts/run_kl_divergence.sh

set -e

export PYTHONPATH=.

# All broad_2 priority clusters
PRIORITY_CLUSTERS="512_17,512_22,512_67,512_181,512_229,512_261,512_471,512_504,768_114,768_140,768_210,768_306,768_570,768_581,768_596,768_672"

SIMPLEX_DIR="/workspace/outputs/simplex_samples"

# Auto-discover control clusters from stats files (is_control=true)
CONTROL_CLUSTERS=$(python -c "
import json, glob, sys
stats_files = glob.glob('${SIMPLEX_DIR}/n*/*_simplex_stats.json')
keys = []
for f in sorted(stats_files):
    d = json.load(open(f))
    if d.get('is_control'):
        keys.append(f'{d[\"n_clusters\"]}_{d[\"cluster_id\"]}')
print(','.join(keys))
")

echo "============================================================"
echo "KL DIVERGENCE ANALYSIS — broad_2 clusters (2A)"
echo "============================================================"
echo "Priority clusters: ${PRIORITY_CLUSTERS}"
echo "Control clusters:  ${CONTROL_CLUSTERS}"

python -u validation/kl_divergence_simplex.py \
    --clusters "${PRIORITY_CLUSTERS}" \
    --control_clusters "${CONTROL_CLUSTERS}" \
    --vertex_samples_dir "outputs/interpretations/prepared_samples_broad_2_no_whitespace" \
    --simplex_samples_dir "${SIMPLEX_DIR}" \
    --output_dir "outputs/validation/kl_divergence_broad_2" \
    --batch_size 256 \
    --n_pairs_per_sample 20 \
    --max_samples_per_vertex 200 \
    --max_simplex_samples 5000 \
    --n_diagnostic_pairs 1000 \
    --seed 42 \
    --model_name "gemma-2-9b" \
    --device "cuda" \
    --cache_dir "/workspace/hf_cache" \
    --hf_token ${HF_TOKEN}

echo ""
echo "============================================================"
echo "Done. Results in outputs/validation/kl_divergence_broad_2/"
echo "============================================================"
