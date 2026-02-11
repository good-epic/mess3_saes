#!/usr/bin/env python3
"""Compare cluster assignments across different clustering methods.

Loads clustering_result.pkl files from multiple runs and computes:
- Adjusted Rand Index (ARI): chance-corrected similarity, 1.0 = identical
- Normalized Mutual Information (NMI): information-theoretic overlap
- Contingency table summary: how clusters map across methods

Usage:
    python scratch/compare_clusterings.py \
        --clustering_dirs \
            "k_subspaces:outputs/real_data_analysis_canonical/clusters_512" \
            "phi:outputs/real_data_cooc/phi_512/clusters_512" \
            "mutual_info:outputs/real_data_cooc/mutual_info_512/clusters_512" \
        --output_dir outputs/clustering_comparison_512
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_cluster_labels(pkl_path: str) -> np.ndarray:
    """Load cluster labels from a clustering_result.pkl file."""
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    return result.cluster_labels


def compare_pair(labels_a: np.ndarray, labels_b: np.ndarray, name_a: str, name_b: str):
    """Compare two clusterings and print metrics."""
    # Only compare latents that are active in both (label != -1)
    valid_mask = (labels_a >= 0) & (labels_b >= 0)
    la = labels_a[valid_mask]
    lb = labels_b[valid_mask]

    ari = adjusted_rand_score(la, lb)
    nmi = normalized_mutual_info_score(la, lb, average_method='arithmetic')

    print(f"\n  {name_a} vs {name_b}:")
    print(f"    Shared active latents: {valid_mask.sum():,} / {len(labels_a):,}")
    print(f"    ARI:  {ari:.4f}  (1.0 = identical, 0.0 = random)")
    print(f"    NMI:  {nmi:.4f}  (1.0 = identical, 0.0 = independent)")

    # Count how many latents change clusters
    same_cluster = (la == lb).sum()
    print(f"    Same label: {same_cluster:,} / {len(la):,} ({same_cluster/len(la):.1%})")

    return {"ari": ari, "nmi": nmi, "n_shared": int(valid_mask.sum()),
            "same_label_frac": float(same_cluster / len(la)) if len(la) > 0 else 0}


def main():
    parser = argparse.ArgumentParser(description="Compare cluster assignments across methods")
    parser.add_argument("--clustering_dirs", type=str, nargs="+", required=True,
                        help="'name:path' pairs, e.g. 'k_subspaces:outputs/.../clusters_512'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional directory to save comparison results")
    args = parser.parse_args()

    # Parse name:path pairs
    clusterings = {}
    for spec in args.clustering_dirs:
        if ":" not in spec:
            print(f"ERROR: Expected 'name:path' format, got '{spec}'")
            sys.exit(1)
        name, path = spec.split(":", 1)
        pkl_path = os.path.join(path, "clustering_result.pkl")
        if not os.path.exists(pkl_path):
            print(f"WARNING: {pkl_path} not found, skipping '{name}'")
            continue
        labels = load_cluster_labels(pkl_path)
        n_clusters = len(set(labels[labels >= 0]))
        clusterings[name] = labels
        print(f"Loaded '{name}': {n_clusters} clusters, {(labels >= 0).sum():,} active latents")

    if len(clusterings) < 2:
        print("ERROR: Need at least 2 clusterings to compare")
        sys.exit(1)

    # Pairwise comparisons
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISONS")
    print("=" * 60)

    results = {}
    names = list(clusterings.keys())
    for name_a, name_b in combinations(names, 2):
        pair_key = f"{name_a}_vs_{name_b}"
        results[pair_key] = compare_pair(
            clusterings[name_a], clusterings[name_b], name_a, name_b
        )

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Pair':<35} {'ARI':>8} {'NMI':>8}")
    print("-" * 55)
    for pair_key, metrics in results.items():
        print(f"{pair_key:<35} {metrics['ari']:>8.4f} {metrics['nmi']:>8.4f}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        import json
        out_path = os.path.join(args.output_dir, "clustering_comparison.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
