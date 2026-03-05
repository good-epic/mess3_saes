#!/usr/bin/env bash
# NC backbone + Leiden CPM clustering of SAE latent co-occurrence network.
#
# Runs a small parameter sweep over z_threshold (backbone strictness) and
# gamma (CPM resolution / cluster granularity).  Each configuration writes
# to its own subdirectory under outputs/cooc_leiden_clustering/.
#
# Parameters:
#   --z_threshold  NC backbone significance threshold.  Higher = fewer edges
#                  retained; 1.96 = p<0.05 two-tailed (standard starting point).
#
#   --min_weight   Pre-filter raw co-occurrence counts below this value.
#                  Removes very rare pairs before backbone computation.
#                  Reduces computation and avoids noisy low-count edges.
#
#   --gamma        CPM resolution parameter.  Roughly equals the expected
#                  internal edge density of communities:
#                    0.005  -> coarser clusters (analogous to ~512-768 spectral)
#                    0.01   -> medium clusters
#                    0.05   -> finer clusters
#                    0.1    -> tight small clusters
#                  Tune based on the size distribution in diagnostics.json.
#
#   --weight_type  Edge weights passed to Leiden: 'raw' (co-occurrence counts)
#                  or 'zscore' (NC z-scores).  'raw' is recommended; 'zscore'
#                  treats all retained edges as equally significant.
#
# After running, inspect each diagnostics.json to compare:
#   - backbone_retention_pct (% of edges kept)
#   - n_valid_clusters, size_min/max/mean/median
#   - cpm_quality
# Then pick a configuration to run AANet analysis on the resulting clusters.

set -euo pipefail

COOC_PATH="outputs/cooc_comparison/cooc_stats.npz"
OUTPUT_DIR="outputs/cooc_leiden_clustering"
MIN_CLUSTER_SIZE=3
SEED=42

echo "============================================================"
echo "NC Backbone + Leiden CPM — parameter sweep"
echo "============================================================"

# ---------------------------------------------------------------------------
# Sweep: z_threshold x gamma  (backbone strictness x cluster granularity)
# ---------------------------------------------------------------------------

for Z_THRESH in 1.96 3.0; do
  for GAMMA in 0.005 0.01 0.05; do

    echo ""
    echo "------------------------------------------------------------"
    echo "z_threshold=${Z_THRESH}  gamma=${GAMMA}"
    echo "------------------------------------------------------------"

    python -u clustering/cooc_leiden.py \
      --cooc_path "${COOC_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --z_threshold "${Z_THRESH}" \
      --min_weight 5 \
      --gamma "${GAMMA}" \
      --weight_type raw \
      --min_cluster_size "${MIN_CLUSTER_SIZE}" \
      --seed "${SEED}" \
      --n_iterations -1

  done
done

echo ""
echo "============================================================"
echo "Sweep complete.  Summary of results:"
echo "============================================================"
echo ""

# Print a quick comparison table from the diagnostics files
python -u - <<'PYEOF'
import json, os, glob

root = "outputs/cooc_leiden_clustering"
rows = []
for diag_path in sorted(glob.glob(f"{root}/*/diagnostics.json")):
    with open(diag_path) as f:
        d = json.load(f)
    bb = d['backbone_stats']
    cs = d['cluster_stats']
    rows.append({
        'run': os.path.basename(os.path.dirname(diag_path)),
        'z': bb['z_threshold'],
        'gamma': cs['gamma'],
        'edges_bb': bb['n_edges_backbone'],
        'bb_pct': bb['backbone_retention_pct'],
        'n_clusters': cs['n_valid_clusters'],
        'singletons': cs['leiden_n_singletons'],
        'sz_min': cs.get('size_min', '-'),
        'sz_max': cs.get('size_max', '-'),
        'sz_mean': cs.get('size_mean', '-'),
        'quality': round(cs['cpm_quality'], 4),
    })

if not rows:
    print("No diagnostics files found.")
else:
    hdr = f"{'run':<42} {'z':>5} {'gamma':>6} {'edges_bb':>10} {'bb%':>6} {'n_clust':>8} {'singletons':>11} {'sz_min':>7} {'sz_max':>7} {'sz_mean':>8} {'quality':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['run']:<42} {r['z']:>5} {r['gamma']:>6} {r['edges_bb']:>10,} "
            f"{r['bb_pct']:>5.1f}% {r['n_clusters']:>8,} {r['singletons']:>11,} "
            f"{str(r['sz_min']):>7} {str(r['sz_max']):>7} {str(r['sz_mean']):>8} "
            f"{r['quality']:>9}"
        )
PYEOF
