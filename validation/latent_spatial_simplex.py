#!/usr/bin/env python3
"""
2Ba: Latent spatial structure analysis in the simplex.

Loads simplex samples produced by collect_simplex_samples.py (which must be run
with the latent_acts field enabled) and computes, for each cluster latent:
  - Activation-weighted centroid in barycentric coordinate space
  - Weighted variance around that centroid
  - Herfindahl index (1 = all mass at one vertex, 1/K = uniform)
  - Distance of centroid from the simplex centroid (0 = interior, max = corner)

For K=3 clusters, also generates:
  - Per-latent KDE heatmaps over the simplex triangle (via epdf_utils)
  - Combined "all latents" heatmap
  - Centroid scatter plot: all latents as points in the 2D simplex, sized by
    total activation mass, colored by total variance

CPU-only. Run after collect_simplex_samples.py has completed.

Usage:
    python validation/latent_spatial_simplex.py \\
        --simplex_samples_dir /workspace/outputs/simplex_samples \\
        --output_dir outputs/validation/latent_spatial \\
        --clusters 512_181,768_140,512_17,768_596,768_210,768_306,768_581,\\
                   512_22,512_67,512_229,512_261,512_471,512_504
"""

import os
import sys
import json
import argparse
import glob as _glob
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from epdf_utils import LatentEPDF, plot_epdfs_to_directory
    from training_and_analysis_utils import project_simplex3_to_2d
    HAS_EPDF = True
except ImportError as _epdf_exc:
    print(f"WARNING: epdf_utils not importable ({_epdf_exc}). Stats will be computed but heatmaps skipped.")
    HAS_EPDF = False


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="2Ba: Latent spatial structure in simplex")
    parser.add_argument("--simplex_samples_dir", type=str, required=True,
                        help="Dir produced by collect_simplex_samples.py. "
                             "Structure: simplex_samples_dir/n{N}/cluster_{id}_k{k}_simplex_samples.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/validation/latent_spatial",
                        help="Directory to write results")
    parser.add_argument("--clusters", type=str, default=None,
                        help="Comma-separated cluster keys to process (e.g. '512_181,768_140'). "
                             "Default: auto-discover all clusters with stats files.")
    parser.add_argument("--skip_heatmaps", action="store_true",
                        help="Skip KDE heatmap generation (faster, stats only)")
    parser.add_argument("--heatmap_grid_size", type=int, default=100,
                        help="Grid resolution for KDE heatmaps")
    parser.add_argument("--vertex_acts_dir", type=str, default=None,
                        help="Dir containing vertex_samples_with_acts.jsonl files "
                             "(from run_annotate_vertex_acts.sh). When provided, adds "
                             "per-vertex latent activation pie charts for k=3 clusters.")
    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def iter_simplex_samples(samples_path):
    """Yield records from a simplex samples JSONL file."""
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_clusters(simplex_samples_dir):
    """Auto-discover all cluster stats files under simplex_samples_dir."""
    import glob
    stats_files = sorted(glob.glob(
        str(Path(simplex_samples_dir) / "n*" / "*_simplex_stats.json")
    ))
    keys = []
    for sf in stats_files:
        with open(sf) as f:
            meta = json.load(f)
        keys.append(meta["cluster_key"])
    return keys


def load_cluster_meta(simplex_samples_dir, cluster_key):
    """Load stats JSON for a cluster and return (meta_dict, samples_path)."""
    n_clusters_str, cluster_id_str = cluster_key.split("_")
    n_clusters = int(n_clusters_str)
    cluster_id = int(cluster_id_str)

    stats_dir = Path(simplex_samples_dir) / f"n{n_clusters}"
    stats_files = sorted(stats_dir.glob(f"cluster_{cluster_id}_k*_simplex_stats.json"))
    if not stats_files:
        return None, None

    with open(stats_files[0]) as f:
        meta = json.load(f)

    samples_path = Path(meta["samples_path"])
    if not samples_path.exists():
        # Try relative to simplex_samples_dir
        samples_path = stats_dir / samples_path.name
    if not samples_path.exists():
        return meta, None

    return meta, samples_path


# =============================================================================
# Core stats computation
# =============================================================================

def compute_latent_stats(samples_path, n_latents, k):
    """Compute activation-weighted centroid and variance for each cluster latent.

    Uses running accumulators — no per-sample storage required.

    Returns dict with keys:
        centroids         (n_latents, k) float64
        variances         (n_latents, k) float64  per-dimension weighted variance
        total_variance    (n_latents,)   sum of per-dim variances = 1 - herfindahl
        herfindahl        (n_latents,)   sum of squared centroid components
        weight_sums       (n_latents,)   total activation mass
        n_active          (n_latents,)   int, samples where latent > 0
        n_total           int, total samples
        has_latent_acts   bool
    """
    weight_sums = np.zeros(n_latents, dtype=np.float64)
    weighted_coord_sums = np.zeros((n_latents, k), dtype=np.float64)
    weighted_sq_sums = np.zeros((n_latents, k), dtype=np.float64)
    n_active = np.zeros(n_latents, dtype=np.int64)
    n_total = 0
    has_latent_acts = True

    for record in iter_simplex_samples(samples_path):
        if "latent_acts" not in record:
            has_latent_acts = False
            break

        bary = np.array(record["barycentric_coords"], dtype=np.float64)  # (k,)
        acts = np.array(record["latent_acts"], dtype=np.float64)          # (n_latents,)

        pos_mask = acts > 0
        if pos_mask.any():
            a = acts[pos_mask]                              # (n_pos,)
            weight_sums[pos_mask] += a
            weighted_coord_sums[pos_mask] += a[:, None] * bary[None, :]
            weighted_sq_sums[pos_mask] += a[:, None] * (bary[None, :] ** 2)
            n_active[pos_mask] += 1

        n_total += 1

    if not has_latent_acts:
        return {"has_latent_acts": False, "n_total": n_total}

    # Compute centroid and variance
    centroids = np.zeros((n_latents, k), dtype=np.float64)
    variances = np.zeros((n_latents, k), dtype=np.float64)

    active_l = weight_sums > 0
    centroids[active_l] = weighted_coord_sums[active_l] / weight_sums[active_l, None]
    variances[active_l] = (
        weighted_sq_sums[active_l] / weight_sums[active_l, None]
        - centroids[active_l] ** 2
    )
    variances = np.clip(variances, 0.0, None)  # numerical guard

    herfindahl = np.sum(centroids ** 2, axis=1)   # = 1 - total_variance
    total_variance = 1.0 - herfindahl             # 0 at corner, 1-1/K at center

    # Distance of centroid from simplex centroid (1/K, ..., 1/K)
    simplex_center = np.full(k, 1.0 / k)
    dist_from_center = np.linalg.norm(centroids - simplex_center[None, :], axis=1)

    return {
        "has_latent_acts": True,
        "centroids": centroids,
        "variances": variances,
        "total_variance": total_variance,
        "herfindahl": herfindahl,
        "weight_sums": weight_sums,
        "n_active": n_active,
        "n_total": n_total,
        "dist_from_center": dist_from_center,
    }


# =============================================================================
# LatentEPDF construction for heatmap (K=3 only)
# =============================================================================

_TRIANGLE_VERTICES = [[0.0, 0.0], [1.0, 0.0], [0.5, float(np.sqrt(3) / 2)]]


def build_epdf_objects(samples_path, n_latents, latent_indices):
    """Build LatentEPDF objects for K=3 heatmap visualization.

    Each EPDF uses a single component named "simplex" with 2D projected
    barycentric coordinates as positions and latent activations as weights.
    """
    coords_per_latent = [[] for _ in range(n_latents)]
    weights_per_latent = [[] for _ in range(n_latents)]

    for record in iter_simplex_samples(samples_path):
        if "latent_acts" not in record:
            return {}
        bary = np.array(record["barycentric_coords"], dtype=np.float32)  # (3,)
        acts = np.array(record["latent_acts"], dtype=np.float32)          # (n_latents,)

        x, y = project_simplex3_to_2d(bary[None, :])

        for l in range(n_latents):
            if acts[l] > 0:
                coords_per_latent[l].append([float(x[0]), float(y[0])])
                weights_per_latent[l].append(float(acts[l]))

    epdfs = {}
    comp_name = "simplex"
    comp_info = {comp_name: {"geometry": "triangle", "triangle_vertices": _TRIANGLE_VERTICES}}

    for l in range(n_latents):
        if not coords_per_latent[l]:
            continue
        coords = np.array(coords_per_latent[l])    # (n_active, 2)
        weights = np.array(weights_per_latent[l])  # (n_active,)
        act_frac = float(len(weights)) / max(1, sum(len(w) for w in weights_per_latent))

        epdf = LatentEPDF(
            site_name="cluster",
            sae_id=(0, 0),
            latent_idx=latent_indices[l],
            component_coords={comp_name: coords},
            component_weights={comp_name: weights},
            component_info=comp_info,
            activation_fraction=act_frac,
        )
        epdf.fit_component_kdes()
        epdfs[latent_indices[l]] = epdf

    return epdfs


# =============================================================================
# Centroid scatter plot
# =============================================================================

def plot_centroid_scatter(centroids, weight_sums, total_variance, latent_indices,
                          cluster_key, output_dir):
    """Plot each latent as a point at its centroid in the 2D simplex.

    Point size ∝ total activation mass.
    Color = total variance (0 = corner-like, high = interior/diffuse).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))

    # Draw triangle
    tri = np.array(_TRIANGLE_VERTICES)
    line_x = np.append(tri[:, 0], tri[0, 0])
    line_y = np.append(tri[:, 1], tri[0, 1])
    ax.plot(line_x, line_y, 'k-', linewidth=1.5, zorder=5)

    # Vertex labels
    vertex_labels = ["V0", "V1", "V2"]
    offsets = [(-0.07, -0.04), (1.04, -0.04), (0.5, float(np.sqrt(3) / 2) + 0.03)]
    for label, (ox, oy) in zip(vertex_labels, offsets):
        ax.text(ox, oy, label, fontsize=9, ha='center', va='center', color='#444444')

    # Filter to latents with non-zero weight
    active = weight_sums > 0
    if not active.any():
        ax.set_title(f"Cluster {cluster_key}: no active latents")
        out_path = Path(output_dir) / f"cluster_{cluster_key}_centroid_scatter.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path

    cent_active = centroids[active]
    w_active = weight_sums[active]
    var_active = total_variance[active]
    idx_active = [latent_indices[i] for i, a in enumerate(active) if a]

    x, y = project_simplex3_to_2d(cent_active)

    w_norm = w_active / (w_active.max() + 1e-12)
    sizes = w_norm * 350 + 20

    k = centroids.shape[1]
    max_var = 1.0 - 1.0 / k
    sc = ax.scatter(x, y, s=sizes, c=var_active, cmap='RdYlBu_r',
                    vmin=0, vmax=max_var, zorder=10,
                    edgecolors='black', linewidths=0.4, alpha=0.85)

    # Latent index labels
    for xi, yi, li in zip(x, y, idx_active):
        ax.annotate(str(li), (xi, yi), fontsize=5, ha='center', va='center',
                    zorder=15, color='white',
                    fontweight='bold')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f'Total variance (0=corner, {max_var:.2f}=center)', fontsize=8)

    # Size legend
    for ref_frac, label in [(0.25, '25%'), (0.75, '75%'), (1.0, '100%')]:
        ax.scatter([], [], s=ref_frac * 350 + 20, c='gray', alpha=0.6,
                   label=f'Weight {label}', edgecolors='black', linewidths=0.4)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, float(np.sqrt(3) / 2) + 0.1)
    ax.set_title(f"Cluster {cluster_key}: latent centroids in simplex\n"
                 f"({sum(active)}/{len(active)} latents with nonzero activation)")
    fig.tight_layout()

    out_path = Path(output_dir) / f"cluster_{cluster_key}_centroid_scatter.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Centroid scatter → {out_path}")
    return out_path


# =============================================================================
# Vertex latent activation pie charts (requires vertex_samples_with_acts.jsonl)
# =============================================================================

def find_vertex_acts_path(vertex_acts_dir, n_clusters, cluster_id, k):
    """Find vertex_samples_with_acts.jsonl for a cluster."""
    pattern = str(Path(vertex_acts_dir) / f"n{n_clusters}"
                  / f"cluster_{cluster_id}_k{k}_*_vertex_samples_with_acts.jsonl")
    matches = _glob.glob(pattern)
    return Path(matches[0]) if matches else None


def load_vertex_latent_means(vertex_acts_path, n_latents, k):
    """Compute mean activation per latent per vertex.

    latent_acts in each record is a list-of-lists (one per trigger).
    We average over all triggers within a record, then average over records.

    Returns:
        mean_acts  (k, n_latents) float64 — mean activation at each vertex
        counts     (k,)           int64   — number of records per vertex
    """
    sum_acts = np.zeros((k, n_latents), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    with open(vertex_acts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            v = record.get("vertex_id")
            if v is None or v >= k:
                continue
            acts_list = record.get("latent_acts")
            if not acts_list:
                continue
            acts = np.mean(np.array(acts_list, dtype=np.float64), axis=0)
            if acts.shape[0] != n_latents:
                continue
            sum_acts[v] += acts
            counts[v] += 1
    mean_acts = np.where(
        counts[:, None] > 0,
        sum_acts / np.maximum(counts[:, None], 1),
        0.0,
    )
    return mean_acts, counts


def plot_vertex_latent_pies(mean_acts, vertex_counts, latent_indices, k,
                             cluster_key, output_dir):
    """Plot a simplex triangle with mean-activation pie charts at each vertex (k=3 only).

    Each pie chart shows the mean magnitude of each cluster latent's activation
    across all near-vertex samples at that vertex.  Only latents contributing
    more than 1% of total activation at a vertex are shown individually;
    the remainder is grouped as 'other'.
    """
    sqrt3_2 = float(np.sqrt(3) / 2)

    # Padded data limits so pie charts at corners have room
    xlim = (-0.5, 1.5)
    ylim = (-0.5, sqrt3_2 + 0.5)
    ax_w = xlim[1] - xlim[0]   # 2.0
    ax_h = ylim[1] - ylim[0]   # sqrt3_2 + 1.0

    def to_axes(dx, dy):
        """Convert data coords to axes fraction [0,1]."""
        return (dx - xlim[0]) / ax_w, (dy - ylim[0]) / ax_h

    # Assign a consistent color per latent position in latent_indices list
    n_latents = len(latent_indices)
    cmap = plt.get_cmap('tab20')
    latent_colors = [cmap(l % 20) for l in range(n_latents)]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw triangle
    tri = np.array(_TRIANGLE_VERTICES)
    line_x = np.append(tri[:, 0], tri[0, 0])
    line_y = np.append(tri[:, 1], tri[0, 1])
    ax.plot(line_x, line_y, 'k-', linewidth=1.5, zorder=5)

    pie_diam = 0.28   # diameter of each inset pie as fraction of axes

    active_latents_seen = set()

    for v in range(3):
        vax, vay = to_axes(*tri[v])

        acts = mean_acts[v]   # (n_latents,)
        total = acts.sum()
        n_samples = int(vertex_counts[v])

        if total <= 0 or n_samples == 0:
            # Label-only, no pie
            ax.text(tri[v][0], tri[v][1] - 0.15, f"V{v}\n(n=0)",
                    fontsize=8, ha='center', va='top', color='#666666')
            continue

        # Filter: keep latents above 1% threshold, top 15 max
        threshold = total * 0.01
        active_idxs = np.where(acts >= threshold)[0]
        if len(active_idxs) > 15:
            active_idxs = active_idxs[np.argsort(-acts[active_idxs])[:15]]
        active_idxs = sorted(active_idxs.tolist())

        pie_vals = acts[active_idxs]
        other_val = total - pie_vals.sum()

        colors = [latent_colors[l] for l in active_idxs]
        pie_vals_full = list(pie_vals)
        colors_full = list(colors)
        if other_val > 1e-12:
            pie_vals_full.append(other_val)
            colors_full.append('#dddddd')

        active_latents_seen.update(active_idxs)

        inset_ax = ax.inset_axes(
            [vax - pie_diam / 2, vay - pie_diam / 2, pie_diam, pie_diam]
        )
        inset_ax.pie(
            pie_vals_full,
            colors=colors_full,
            startangle=90,
            wedgeprops={'linewidth': 0.4, 'edgecolor': 'white'},
        )
        inset_ax.set_title(f"V{v}  (n={n_samples})", fontsize=8, pad=3)

    # Legend: one entry per latent that appeared in any vertex pie
    from matplotlib.patches import Patch
    handles = [
        Patch(color=latent_colors[l], label=f"L{latent_indices[l]}")
        for l in sorted(active_latents_seen)
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc='lower center',
            fontsize=7,
            ncol=min(len(handles), 6),
            bbox_to_anchor=(0.5, 0.0),
            title="Latent index",
            title_fontsize=8,
            framealpha=0.8,
        )

    ax.set_title(
        f"Cluster {cluster_key}: mean latent activation at each vertex\n"
        f"(slice ∝ mean magnitude; latents < 1% of total grouped as 'other')",
        fontsize=10,
    )

    out_path = Path(output_dir) / f"cluster_{cluster_key}_vertex_latent_pies.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Vertex latent pies → {out_path}")
    return out_path


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine clusters
    if args.clusters:
        cluster_keys = [c.strip() for c in args.clusters.split(',') if c.strip()]
    else:
        cluster_keys = discover_clusters(args.simplex_samples_dir)
        print(f"Auto-discovered {len(cluster_keys)} clusters: {cluster_keys}")

    if not cluster_keys:
        print("No clusters to process.")
        return

    all_results = {}

    for cluster_key in cluster_keys:
        print(f"\n{'=' * 60}")
        print(f"Cluster {cluster_key}")

        meta, samples_path = load_cluster_meta(args.simplex_samples_dir, cluster_key)
        if meta is None:
            print(f"  No stats file found, skipping")
            continue
        if samples_path is None:
            print(f"  Samples file not found at {meta.get('samples_path')}, skipping")
            continue

        k = meta["k"]
        latent_indices = meta.get("latent_indices")
        if not latent_indices:
            print(f"  No latent_indices in stats file. "
                  f"Re-run collect_simplex_samples.py to include them.")
            continue

        n_latents = len(latent_indices)
        print(f"  k={k}, n_latents={n_latents}, samples={meta.get('n_collected', '?')}")

        # Compute stats
        stats = compute_latent_stats(samples_path, n_latents, k)

        if not stats["has_latent_acts"]:
            print(f"  ERROR: JSONL records missing 'latent_acts' field. "
                  f"Re-run collect_simplex_samples.py.")
            continue

        n_total = stats["n_total"]
        print(f"  Processed {n_total} samples")

        # Print per-latent summary (top 10 by total variance = most interior)
        print(f"\n  Latent centroid summary (top 10 by total_variance = most interior):")
        print(f"  {'Latent':>8}  {'Centroid':^{k * 7}}  {'TotVar':>7}  {'WeightSum':>10}  {'N_active':>8}")
        header_bar = "  " + "-" * (8 + 2 + k * 7 + 2 + 7 + 2 + 10 + 2 + 8)
        print(header_bar)

        sorted_l = sorted(range(n_latents), key=lambda l: -stats["total_variance"][l])
        for l in sorted_l[:10]:
            centroid_str = " ".join(f"{c:.3f}" for c in stats["centroids"][l])
            print(f"  {latent_indices[l]:>8}  [{centroid_str}]  "
                  f"{stats['total_variance'][l]:>7.3f}  "
                  f"{stats['weight_sums'][l]:>10.1f}  "
                  f"{stats['n_active'][l]:>8}")

        # Build result dict
        result = {
            "cluster_key": cluster_key,
            "k": k,
            "n_latents": n_latents,
            "latent_indices": latent_indices,
            "n_total_samples": n_total,
            "is_control": meta.get("is_control", False),
            "per_latent": [
                {
                    "latent_idx": latent_indices[l],
                    "centroid": stats["centroids"][l].tolist(),
                    "variance_per_dim": stats["variances"][l].tolist(),
                    "total_variance": float(stats["total_variance"][l]),
                    "herfindahl": float(stats["herfindahl"][l]),
                    "dist_from_center": float(stats["dist_from_center"][l]),
                    "weight_sum": float(stats["weight_sums"][l]),
                    "n_active": int(stats["n_active"][l]),
                }
                for l in range(n_latents)
            ],
        }

        # Save per-cluster JSON
        stats_out = output_dir / f"cluster_{cluster_key}_spatial_stats.json"
        with open(stats_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Stats → {stats_out}")

        # Plots
        cluster_plot_dir = output_dir / f"cluster_{cluster_key}"
        cluster_plot_dir.mkdir(exist_ok=True)

        if k == 3:
            # Centroid scatter plot (always)
            plot_centroid_scatter(
                stats["centroids"],
                stats["weight_sums"],
                stats["total_variance"],
                latent_indices,
                cluster_key,
                cluster_plot_dir,
            )

            # Vertex latent activation pie charts (requires vertex_acts_dir)
            if args.vertex_acts_dir:
                n_clusters_str, cluster_id_str = cluster_key.split("_")
                vacts_path = find_vertex_acts_path(
                    args.vertex_acts_dir,
                    int(n_clusters_str), int(cluster_id_str), k,
                )
                if vacts_path is None:
                    print(f"  Vertex acts file not found in {args.vertex_acts_dir}, skipping pies")
                else:
                    mean_acts, vcounts = load_vertex_latent_means(vacts_path, n_latents, k)
                    plot_vertex_latent_pies(
                        mean_acts, vcounts, latent_indices, k,
                        cluster_key, cluster_plot_dir,
                    )

            # KDE heatmaps (unless skipped or epdf unavailable)
            if not args.skip_heatmaps and HAS_EPDF:
                print(f"  Building KDE heatmaps (grid_size={args.heatmap_grid_size})...")
                epdfs = build_epdf_objects(samples_path, n_latents, latent_indices)
                if epdfs:
                    plot_epdfs_to_directory(
                        epdfs,
                        output_dir=str(cluster_plot_dir),
                        component_order=["simplex"],
                        title_prefix=f"Cluster {cluster_key}: ",
                        grid_size=args.heatmap_grid_size,
                        legend_outside=True,
                    )
                    print(f"  Heatmaps → {cluster_plot_dir}/")
                else:
                    print(f"  No active latents for heatmap")
            elif args.skip_heatmaps:
                print(f"  Heatmaps skipped (--skip_heatmaps)")
            else:
                print(f"  Heatmaps skipped (epdf_utils not available)")

        elif k == 4:
            print(f"  k=4 (3-simplex): heatmap projection not supported; stats only")

        all_results[cluster_key] = result

    # Save combined
    combined_out = output_dir / "all_spatial_stats.json"
    with open(combined_out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results → {combined_out}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Cluster':>12}  {'k':>3}  {'n_lat':>5}  "
          f"{'Frac_interior':>13}  {'MeanTotVar':>10}  {'MaxTotVar':>10}")
    print("  " + "-" * 70)
    for ck, res in all_results.items():
        k_ = res["k"]
        n_lat = res["n_latents"]
        tvars = [pl["total_variance"] for pl in res["per_latent"]]
        # "Interior" = total_variance > 0.25 (heuristic: not clearly at a corner)
        n_interior = sum(1 for tv in tvars if tv > (1 - 1.0 / k_) * 0.4)
        frac_interior = n_interior / n_lat if n_lat > 0 else 0
        mean_tv = float(np.mean(tvars)) if tvars else 0
        max_tv = float(np.max(tvars)) if tvars else 0
        print(f"  {ck:>12}  {k_:>3}  {n_lat:>5}  "
              f"{frac_interior:>13.1%}  {mean_tv:>10.3f}  {max_tv:>10.3f}")


if __name__ == "__main__":
    main()
