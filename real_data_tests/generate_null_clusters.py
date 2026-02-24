#!/usr/bin/env python3
"""
Generate null cluster labels by randomly shuffling SAE latents into clusters
that mirror the size distribution of a real co-occurrence clustering.

Produces a .npy file with shape (sae_width,) where each entry is the cluster
id (0..n_clusters-1) for that latent index, or -1 if unassigned.  The result
can be passed to analyze_real_saes.py via --cluster_labels_file to bypass
co-occurrence collection and spectral clustering.

Usage:
    python real_data_tests/generate_null_clusters.py \
        --source_csv outputs/real_data_analysis_canonical/clusters_512/consolidated_metrics_n512.csv \
        --output_file outputs/null_clusters/null_labels_n512.npy \
        --seed 42
"""

import argparse
import ast
import os
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate null cluster labels matching a real clustering's size distribution")
    parser.add_argument("--source_csv", type=str, required=True,
                        help="Path to consolidated_metrics_nN.csv from a real clustering run")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the null cluster labels (.npy)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading real cluster CSV: {args.source_csv}")
    df = pd.read_csv(args.source_csv)

    # One row per (cluster_id, k) — take first row per cluster to get sizes and latent indices
    per_cluster = df.groupby("cluster_id").first().reset_index()
    per_cluster = per_cluster.sort_values("cluster_id")

    n_clusters = len(per_cluster)
    cluster_ids = per_cluster["cluster_id"].tolist()
    cluster_sizes = per_cluster["n_latents"].tolist()

    print(f"  n_clusters: {n_clusters}")
    print(f"  Cluster sizes — min: {min(cluster_sizes)}, max: {max(cluster_sizes)}, "
          f"mean: {sum(cluster_sizes)/len(cluster_sizes):.1f}, total: {sum(cluster_sizes)}")

    # Collect all assigned latent indices from the real clustering
    all_assigned = []
    for latent_str in per_cluster["latent_indices"]:
        all_assigned.extend(ast.literal_eval(latent_str))
    all_assigned = np.array(sorted(set(all_assigned)), dtype=int)

    print(f"  Total assigned latents: {len(all_assigned)}")
    assert len(all_assigned) == sum(cluster_sizes), (
        f"Mismatch: {len(all_assigned)} unique latents vs {sum(cluster_sizes)} total cluster sizes"
    )

    # Determine SAE width from max latent index
    sae_width = int(all_assigned.max()) + 1
    print(f"  Inferred SAE width: {sae_width}")

    # Randomly shuffle and reassign latents to clusters with the same size distribution
    rng = np.random.default_rng(args.seed)
    shuffled = all_assigned.copy()
    rng.shuffle(shuffled)

    labels = np.full(sae_width, -1, dtype=int)
    pos = 0
    for cid, size in zip(cluster_ids, cluster_sizes):
        labels[shuffled[pos:pos + size]] = cid
        pos += size

    assert pos == len(shuffled), "Not all latents were assigned"
    assert (labels != -1).sum() == len(all_assigned), "Assignment count mismatch"

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    np.save(args.output_file, labels)
    print(f"\nSaved null cluster labels to: {args.output_file}")
    print(f"  Array shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"  Assigned: {(labels != -1).sum()}, Unassigned (-1): {(labels == -1).sum()}")


if __name__ == "__main__":
    main()
