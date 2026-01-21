#!/usr/bin/env python3
"""
Patch existing CSVs with PCA rank data from pickle files.

This script fixes CSVs where activation_pca_rank is all NaN because
the CSV generation code was looking for "explained_variance_ratio"
but the pickle stores "activation_pca_rank" directly.

Usage:
    python scratch/patch_csv_pca_ranks.py
    python scratch/patch_csv_pca_ranks.py --n_clusters 512 768
    python scratch/patch_csv_pca_ranks.py --dry_run
"""

import argparse
import pickle
import pandas as pd
from pathlib import Path


def patch_csv_with_pca_ranks(base_dir: Path, n_clusters: int, dry_run: bool = False) -> dict:
    """
    Patch a single CSV with PCA ranks from its corresponding pickle file.

    Returns dict with statistics about the patching.
    """
    cluster_dir = base_dir / f"clusters_{n_clusters}"
    pkl_path = cluster_dir / "clustering_result.pkl"
    csv_path = cluster_dir / f"consolidated_metrics_n{n_clusters}.csv"

    stats = {
        "n_clusters": n_clusters,
        "pkl_exists": pkl_path.exists(),
        "csv_exists": csv_path.exists(),
        "clusters_with_pca": 0,
        "rows_updated": 0,
        "rows_total": 0,
    }

    if not pkl_path.exists():
        print(f"  n={n_clusters}: Pickle file not found at {pkl_path}")
        return stats

    if not csv_path.exists():
        print(f"  n={n_clusters}: CSV file not found at {csv_path}")
        return stats

    # Load pickle
    print(f"  n={n_clusters}: Loading pickle...")
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    if not hasattr(result, 'cluster_stats') or result.cluster_stats is None:
        print(f"  n={n_clusters}: No cluster_stats in pickle")
        return stats

    # Extract PCA ranks from pickle
    pca_ranks = {}
    for cid, cstats in result.cluster_stats.items():
        if "activation_pca_rank" in cstats:
            pca_ranks[cid] = cstats["activation_pca_rank"]

    stats["clusters_with_pca"] = len(pca_ranks)
    print(f"  n={n_clusters}: Found PCA ranks for {len(pca_ranks)} clusters in pickle")

    if len(pca_ranks) == 0:
        print(f"  n={n_clusters}: No PCA data to patch")
        return stats

    # Load CSV
    df = pd.read_csv(csv_path)
    stats["rows_total"] = len(df)

    # Check current state
    non_null_before = df['activation_pca_rank'].notna().sum()
    print(f"  n={n_clusters}: CSV has {non_null_before}/{len(df)} non-null PCA ranks before patching")

    # Patch the activation_pca_rank column
    rows_updated = 0
    for idx, row in df.iterrows():
        cid = row['cluster_id']
        if cid in pca_ranks:
            if pd.isna(row['activation_pca_rank']) or row['activation_pca_rank'] != pca_ranks[cid]:
                df.at[idx, 'activation_pca_rank'] = pca_ranks[cid]
                rows_updated += 1

    stats["rows_updated"] = rows_updated

    non_null_after = df['activation_pca_rank'].notna().sum()
    print(f"  n={n_clusters}: Updated {rows_updated} rows, now {non_null_after}/{len(df)} non-null")

    # Save CSV
    if not dry_run:
        df.to_csv(csv_path, index=False)
        print(f"  n={n_clusters}: Saved updated CSV to {csv_path}")
    else:
        print(f"  n={n_clusters}: DRY RUN - would save to {csv_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Patch CSVs with PCA ranks from pickles")
    parser.add_argument('--base_dir', type=str,
                        default="outputs/real_data_analysis_canonical",
                        help='Base directory for outputs')
    parser.add_argument('--n_clusters', type=int, nargs='+',
                        default=[128, 256, 512, 768],
                        help='Which n_clusters values to patch')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without saving')

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    print("=" * 60)
    print("PATCHING CSVs WITH PCA RANKS FROM PICKLES")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"N clusters to patch: {args.n_clusters}")
    if args.dry_run:
        print("DRY RUN MODE - no files will be modified")
    print()

    all_stats = []
    for n in args.n_clusters:
        print(f"\nProcessing n_clusters={n}...")
        stats = patch_csv_with_pca_ranks(base_dir, n, args.dry_run)
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_updated = sum(s["rows_updated"] for s in all_stats)
    total_rows = sum(s["rows_total"] for s in all_stats)
    print(f"Total rows updated: {total_updated}/{total_rows}")

    for s in all_stats:
        status = "OK" if s["rows_updated"] > 0 else "SKIP"
        print(f"  n={s['n_clusters']}: {s['rows_updated']} rows updated [{status}]")


if __name__ == "__main__":
    main()
