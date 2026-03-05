#!/usr/bin/env python3
"""NC backbone + Leiden CPM clustering of SAE latent co-occurrence networks.

Pipeline:
  1. Load co-occurrence statistics (N11, N1, n_samples) from cooc_stats.npz
  2. Apply Noise Corrected (NC) backbone (Coscia & Neffke 2017) to prune
     statistically non-significant edges
  3. Run Leiden community detection with CPM objective on the backbone graph
  4. Output cluster assignments in a format compatible with consolidated_metrics CSV

References:
  - Coscia & Neffke (2017) "Network Backboning with Noisy Data", ICDE
  - Traag, Waltman & van Eck (2019) "From Louvain to Leiden", Sci Reports

NC backbone null model:
  For each undirected edge (i,j) with observed weight w_ij, compute:
    E[w_ij]   = s_i * s_j / W          (null expected weight)
    Var[w_ij] = W * p * (1-p)          (binomial variance)
              where p = s_i * s_j / W^2
    z_ij      = (w_ij - E[w_ij]) / sqrt(Var)
  Keep edge if z_ij > z_threshold.
  s_i = sum_j w_ij (node strength), W = sum_i s_i (total directed weight = 2 * edge sum)
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# NC backbone
# ---------------------------------------------------------------------------

def compute_nc_backbone(N11, z_threshold=1.96, min_weight=1, verbose=True):
    """Compute the Noise Corrected backbone of a co-occurrence network.

    Args:
        N11: (n, n) co-occurrence count matrix, symmetric, diagonal may be non-zero.
        z_threshold: Keep edges with z-score above this value. Default 1.96 (p<0.05).
        min_weight: Pre-filter: ignore edges with N11[i,j] < min_weight. Default 1.
        verbose: Print progress.

    Returns:
        dict with keys:
            rows, cols:         backbone edge indices (upper triangle only)
            weights:            original co-occurrence counts for backbone edges
            z_scores:           z-scores for backbone edges
            node_strengths:     per-node strength s[i] = sum_j N11[i,j] (diagonal excluded)
            n_edges_raw:        non-zero upper-triangle edges before backbone filter
            n_edges_backbone:   edges surviving backbone filter
    """
    n = N11.shape[0]
    if verbose:
        print(f"  Matrix: {n} x {n}")

    # --- node strengths (exclude diagonal) ---
    # N11 is symmetric; s[i] = sum_j N11[i,j] for j != i.
    # We compute row sums then subtract diagonal.
    if verbose:
        print("  Computing node strengths...")
    diag_vals = np.diag(N11).astype(np.float64)
    row_sums = N11.sum(axis=1).astype(np.float64)
    s = row_sums - diag_vals   # shape (n,)

    # Total directed weight: W = sum_i s_i = 2 * (sum of upper triangle)
    W = s.sum()
    if verbose:
        print(f"  Total directed weight W: {W:,.0f}")
        print(f"  Active nodes (s > 0): {(s > 0).sum():,}")

    # --- convert upper triangle to COO sparse (exclude diagonal) ---
    if verbose:
        print("  Converting to sparse upper triangle...")
    # Keep int64 through sparsification to avoid a full 2GB float64 copy;
    # convert to float64 only for the (much smaller) non-zero values.
    N11_sp = sparse.csr_matrix(N11)           # int64, only non-zeros stored
    upper = sparse.triu(N11_sp, k=1).tocoo()  # k=1 excludes diagonal

    rows = upper.row.astype(np.int32)
    cols = upper.col.astype(np.int32)
    weights = upper.data.astype(np.float64)   # convert only nnz values

    n_edges_raw = len(weights)
    nonzero_mask = weights > 0
    rows, cols, weights = rows[nonzero_mask], cols[nonzero_mask], weights[nonzero_mask]

    if verbose:
        print(f"  Non-zero edges (upper triangle): {len(weights):,}")

    # Pre-filter by minimum raw weight
    if min_weight > 1:
        mw_mask = weights >= min_weight
        rows, cols, weights = rows[mw_mask], cols[mw_mask], weights[mw_mask]
        if verbose:
            print(f"  After min_weight={min_weight} filter: {len(weights):,} edges")

    if len(weights) == 0:
        print("  WARNING: No edges remain after min_weight filter.")
        return {
            'rows': np.array([], dtype=np.int32),
            'cols': np.array([], dtype=np.int32),
            'weights': np.array([], dtype=np.float64),
            'z_scores': np.array([], dtype=np.float64),
            'node_strengths': s,
            'n_edges_raw': n_edges_raw,
            'n_edges_backbone': 0,
        }

    # --- NC z-scores ---
    if verbose:
        print("  Computing NC z-scores (vectorized)...")

    s_i = s[rows]
    s_j = s[cols]

    # Null expected weight: E[w_ij] = s_i * s_j / W
    expected = s_i * s_j / W

    # Binomial variance: n=W trials, p=s_i*s_j/W^2
    # Var = W * p * (1-p) = E[w] * (1 - s_i*s_j/W^2)
    p_ij = s_i * s_j / (W * W)
    variance = W * p_ij * (1.0 - p_ij)
    std = np.sqrt(np.maximum(variance, 1e-10))

    z_scores = (weights - expected) / std

    # --- threshold ---
    keep = z_scores > z_threshold
    n_keep = int(keep.sum())

    if verbose:
        print(f"  Z-threshold: {z_threshold}")
        if len(z_scores) > 0:
            print(f"  Z-score range (all edges): [{z_scores.min():.2f}, {z_scores.max():.2f}]")
        print(f"  Backbone edges: {n_keep:,} / {len(weights):,}  "
              f"({100.0 * n_keep / max(len(weights), 1):.1f}%)")
        if n_keep > 0:
            print(f"  Backbone z-score range: [{z_scores[keep].min():.2f}, {z_scores[keep].max():.2f}]")

    return {
        'rows': rows[keep],
        'cols': cols[keep],
        'weights': weights[keep],
        'z_scores': z_scores[keep],
        'node_strengths': s,
        'n_edges_raw': n_edges_raw,
        'n_edges_backbone': n_keep,
    }


# ---------------------------------------------------------------------------
# Leiden CPM
# ---------------------------------------------------------------------------

def run_leiden_cpm(n_nodes, edges_src, edges_dst, edge_weights,
                   gamma, seed=42, n_iterations=-1, verbose=True):
    """Run Leiden algorithm with CPM (Constant Potts Model) objective.

    CPM avoids the resolution limit of modularity. The resolution parameter
    gamma sets the expected internal edge density of communities:
      - lower gamma  -> fewer, larger communities
      - higher gamma -> more, smaller communities

    Args:
        n_nodes: Total number of nodes in the graph.
        edges_src, edges_dst: Arrays of edge endpoints (undirected).
        edge_weights: Array of edge weights.
        gamma: CPM resolution parameter.
        seed: Random seed.
        n_iterations: Leiden iterations. -1 = run until convergence.
        verbose: Print progress.

    Returns:
        dict with:
            labels:        np.array(n_nodes,) of community IDs (Leiden's own numbering)
            n_communities: total number of distinct communities
            n_singletons:  communities of size 1
            quality:       CPM quality value
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError(
            "leidenalg and igraph are required.\n"
            "Install with: pip install leidenalg igraph"
        )

    if len(edges_src) == 0:
        print("  WARNING: No backbone edges — every node will be its own community.")
        labels = np.arange(n_nodes, dtype=np.int32)
        return {
            'labels': labels,
            'n_communities': n_nodes,
            'n_singletons': n_nodes,
            'quality': 0.0,
        }

    if verbose:
        print(f"  Building igraph ({n_nodes:,} nodes, {len(edges_src):,} edges)...")

    edges = list(zip(edges_src.tolist(), edges_dst.tolist()))
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    g.es['weight'] = edge_weights.tolist()

    if verbose:
        components = g.clusters()
        print(f"  Connected components: {len(components)} "
              f"(largest: {max(len(c) for c in components):,} nodes)")
        print(f"  Running Leiden CPM (gamma={gamma}, seed={seed})...")

    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights='weight',
        resolution_parameter=gamma,
        n_iterations=n_iterations,
        seed=seed,
    )

    labels = np.array(partition.membership, dtype=np.int32)
    quality = float(partition.quality())

    unique_labels, counts = np.unique(labels, return_counts=True)
    n_communities = len(unique_labels)
    n_singletons = int((counts == 1).sum())

    if verbose:
        print(f"  Communities: {n_communities:,}  (singletons: {n_singletons:,},  "
              f"non-trivial: {n_communities - n_singletons:,})")
        print(f"  CPM quality: {quality:.6f}")

    return {
        'labels': labels,
        'n_communities': n_communities,
        'n_singletons': n_singletons,
        'quality': quality,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NC backbone + Leiden CPM clustering of SAE latent co-occurrence network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--cooc_path', default='outputs/cooc_comparison/cooc_stats.npz',
        help='Path to cooc_stats.npz (fields: N11, N1, n_samples)',
    )
    parser.add_argument(
        '--output_dir', default='outputs/cooc_leiden_clustering',
        help='Root output directory. A subdirectory named by parameters is created inside.',
    )
    parser.add_argument(
        '--run_name', default=None,
        help='Override subdirectory name (default: auto-generated from parameters)',
    )
    # NC backbone parameters
    parser.add_argument(
        '--z_threshold', type=float, default=1.96,
        help='NC backbone z-score threshold. Higher = more conservative pruning.',
    )
    parser.add_argument(
        '--min_weight', type=int, default=5,
        help='Pre-filter: skip edges with raw co-occurrence count below this value.',
    )
    # Leiden parameters
    parser.add_argument(
        '--gamma', type=float, default=0.01,
        help='CPM resolution parameter. Higher = more, smaller communities.',
    )
    parser.add_argument(
        '--weight_type', choices=['raw', 'zscore'], default='raw',
        help='Edge weight passed to Leiden: raw co-occurrence counts or NC z-scores.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for Leiden.',
    )
    parser.add_argument(
        '--n_iterations', type=int, default=-1,
        help='Leiden iterations. -1 = run until convergence.',
    )
    # Output filtering
    parser.add_argument(
        '--min_cluster_size', type=int, default=3,
        help='Minimum cluster size to include in the output CSV (singletons/pairs excluded).',
    )
    args = parser.parse_args()

    # --- output directory ---
    if args.run_name:
        run_dir = os.path.join(args.output_dir, args.run_name)
    else:
        run_dir = os.path.join(
            args.output_dir,
            f"z{args.z_threshold}_mw{args.min_weight}_g{args.gamma}_{args.weight_type}"
        )
    os.makedirs(run_dir, exist_ok=True)

    t_start = time.time()
    timestamp = datetime.now().isoformat()

    run_info = {
        'timestamp': timestamp,
        'cooc_path': args.cooc_path,
        'z_threshold': args.z_threshold,
        'min_weight': args.min_weight,
        'gamma': args.gamma,
        'weight_type': args.weight_type,
        'seed': args.seed,
        'n_iterations': args.n_iterations,
        'min_cluster_size': args.min_cluster_size,
        'run_dir': run_dir,
    }

    print("=" * 70)
    print("NC Backbone + Leiden CPM Clustering")
    print("=" * 70)
    print(f"  cooc_path:      {args.cooc_path}")
    print(f"  z_threshold:    {args.z_threshold}")
    print(f"  min_weight:     {args.min_weight}")
    print(f"  gamma:          {args.gamma}")
    print(f"  weight_type:    {args.weight_type}")
    print(f"  min_cluster_sz: {args.min_cluster_size}")
    print(f"  output:         {run_dir}")

    # -----------------------------------------------------------------------
    # Step 1: Load co-occurrence data
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1: Loading co-occurrence data")
    print("=" * 70)
    t1 = time.time()

    data = np.load(args.cooc_path)
    N11 = data['N11']          # (n_latents, n_latents), int64
    N1 = data['N1']            # (n_latents,), int64
    n_samples = int(data['n_samples'].item() if data['n_samples'].ndim > 0
                    else data['n_samples'])
    n_latents = N11.shape[0]
    n_active = int((N1 > 0).sum())

    print(f"  N11 shape: {N11.shape},  dtype: {N11.dtype}")
    print(f"  n_samples: {n_samples:,}")
    print(f"  n_latents: {n_latents:,}  (active N1>0: {n_active:,})")
    print(f"  Load time: {time.time() - t1:.1f}s")

    # -----------------------------------------------------------------------
    # Step 2: NC backbone
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2: NC backbone")
    print("=" * 70)
    t2 = time.time()

    backbone = compute_nc_backbone(
        N11,
        z_threshold=args.z_threshold,
        min_weight=args.min_weight,
        verbose=True,
    )

    print(f"  Time: {time.time() - t2:.1f}s")

    # Free N11 — no longer needed
    del N11
    gc.collect()

    n_edges_raw = backbone['n_edges_raw']
    n_edges_bb = backbone['n_edges_backbone']
    backbone_pct = 100.0 * n_edges_bb / max(n_edges_raw, 1)

    # -----------------------------------------------------------------------
    # Step 3: Leiden CPM
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3: Leiden CPM community detection")
    print("=" * 70)
    t3 = time.time()

    edge_weights_for_leiden = (
        backbone['z_scores'] if args.weight_type == 'zscore'
        else backbone['weights']
    )

    leiden_result = run_leiden_cpm(
        n_nodes=n_latents,
        edges_src=backbone['rows'],
        edges_dst=backbone['cols'],
        edge_weights=edge_weights_for_leiden,
        gamma=args.gamma,
        seed=args.seed,
        n_iterations=args.n_iterations,
        verbose=True,
    )

    print(f"  Time: {time.time() - t3:.1f}s")

    # -----------------------------------------------------------------------
    # Step 4: Build filtered cluster list
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: Building cluster list")
    print("=" * 70)

    raw_labels = leiden_result['labels']   # (n_latents,) Leiden community IDs

    community_to_latents: dict[int, list[int]] = defaultdict(list)
    for latent_idx, cid in enumerate(raw_labels):
        community_to_latents[int(cid)].append(latent_idx)

    # Separate by size
    singleton_count = 0
    small_count = 0
    valid_clusters = []   # (latent_list,) tuples in size-descending order

    for cid, latent_list in community_to_latents.items():
        sz = len(latent_list)
        if sz == 1:
            singleton_count += 1
        elif sz < args.min_cluster_size:
            small_count += 1
        else:
            valid_clusters.append(latent_list)

    valid_clusters.sort(key=lambda lst: -len(lst))   # largest first
    n_valid = len(valid_clusters)
    sizes = [len(lst) for lst in valid_clusters]

    print(f"  Total Leiden communities:  {leiden_result['n_communities']:,}")
    print(f"  Singletons (size=1):       {singleton_count:,}")
    print(f"  Small (size 2–{args.min_cluster_size - 1}):       {small_count:,}")
    print(f"  Valid (size >= {args.min_cluster_size}):       {n_valid:,}")

    if n_valid == 0:
        print(
            "\nWARNING: No clusters found at or above min_cluster_size. "
            "Try lowering --gamma or --z_threshold."
        )
        # Still write diagnostics so the sweep can be evaluated
    else:
        sizes_arr = np.array(sizes)
        print(f"  Size range:   [{sizes_arr.min()}, {sizes_arr.max()}]")
        print(f"  Mean / Median: {sizes_arr.mean():.1f} / {np.median(sizes_arr):.1f}")
        print(f"  P25 / P75:     {np.percentile(sizes_arr, 25):.0f} / {np.percentile(sizes_arr, 75):.0f}")

    # -----------------------------------------------------------------------
    # Step 5: Write outputs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 5: Writing outputs")
    print("=" * 70)

    # --- 5a: Cluster CSV (matches consolidated_metrics column layout for core fields) ---
    csv_rows = []
    for new_id, latent_list in enumerate(valid_clusters):
        csv_rows.append({
            # Core columns matching consolidated_metrics format
            'n_clusters_total': n_valid,
            'cluster_id': new_id,
            'n_latents': len(latent_list),
            'latent_indices': str(sorted(latent_list)),
            # Leiden-specific
            'gamma': args.gamma,
            'z_threshold': args.z_threshold,
            'min_weight': args.min_weight,
            'weight_type': args.weight_type,
            'cpm_quality': leiden_result['quality'],
        })

    csv_path = os.path.join(run_dir, 'clusters_leiden.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Cluster CSV:    {csv_path}  ({n_valid} clusters)")

    # --- 5b: Full per-latent label array (-1 for excluded/singleton) ---
    labels_out = np.full(n_latents, -1, dtype=np.int32)
    for new_id, latent_list in enumerate(valid_clusters):
        for li in latent_list:
            labels_out[li] = new_id

    labels_path = os.path.join(run_dir, 'labels_leiden.npy')
    np.save(labels_path, labels_out)
    print(f"  Labels array:   {labels_path}")

    # --- 5c: Cluster size histogram ---
    size_bins = [3, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 999999]
    size_hist = {}
    if sizes:
        sizes_arr = np.array(sizes)
        for lo, hi in zip(size_bins[:-1], size_bins[1:]):
            count = int(((sizes_arr >= lo) & (sizes_arr < hi)).sum())
            label = f"{lo}–{hi - 1}" if hi < 999999 else f"{lo}+"
            if count:
                size_hist[label] = count

    # --- 5d: Full diagnostics JSON ---
    backbone_stats = {
        'z_threshold': args.z_threshold,
        'min_weight': args.min_weight,
        'n_edges_raw': n_edges_raw,
        'n_edges_backbone': n_edges_bb,
        'backbone_retention_pct': round(backbone_pct, 2),
        'backbone_density': round(float(n_edges_bb) / max(n_latents * (n_latents - 1) / 2, 1), 8),
    }

    cluster_stats: dict = {
        'gamma': args.gamma,
        'weight_type': args.weight_type,
        'seed': args.seed,
        'leiden_n_communities_total': leiden_result['n_communities'],
        'leiden_n_singletons': singleton_count,
        'leiden_n_small': small_count,
        'n_valid_clusters': n_valid,
        'min_cluster_size': args.min_cluster_size,
        'cpm_quality': leiden_result['quality'],
        'cluster_size_histogram': size_hist,
    }
    if sizes:
        sizes_arr = np.array(sizes)
        cluster_stats.update({
            'size_min': int(sizes_arr.min()),
            'size_max': int(sizes_arr.max()),
            'size_mean': round(float(sizes_arr.mean()), 1),
            'size_median': float(np.median(sizes_arr)),
            'size_p25': float(np.percentile(sizes_arr, 25)),
            'size_p75': float(np.percentile(sizes_arr, 75)),
        })

    data_stats = {
        'cooc_path': args.cooc_path,
        'n_latents': n_latents,
        'n_active_latents': n_active,
        'n_samples': n_samples,
    }

    diagnostics = {
        'run_info': run_info,
        'data_stats': data_stats,
        'backbone_stats': backbone_stats,
        'cluster_stats': cluster_stats,
        'total_time_s': round(time.time() - t_start, 1),
    }

    diag_path = os.path.join(run_dir, 'diagnostics.json')
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  Diagnostics:    {diag_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  z_threshold={args.z_threshold}  min_weight={args.min_weight}  "
          f"gamma={args.gamma}  weight={args.weight_type}")
    print(f"  Backbone: {n_edges_bb:,} / {n_edges_raw:,} edges retained ({backbone_pct:.1f}%)")
    if n_valid:
        sizes_arr = np.array(sizes)
        print(f"  Clusters: {n_valid} valid  "
              f"(sizes {sizes_arr.min()}–{sizes_arr.max()}, mean {sizes_arr.mean():.1f})")
    else:
        print("  Clusters: 0 valid (adjust parameters)")
    print(f"  Singletons excluded: {singleton_count:,}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Output dir: {run_dir}")


if __name__ == '__main__':
    main()
