#!/usr/bin/env python3
"""
Create a manifest.json from existing vertex sample files.
This allows prepare_vertex_samples.py to work without a full rerun.
"""

import json
import re
import ast
from pathlib import Path
import pandas as pd

# Configuration
OUTPUT_DIR = Path("outputs/selected_clusters_canonical")
CSV_DIR = Path("outputs/real_data_analysis_canonical")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

def parse_filename(filename):
    """Extract cluster info from filename like cluster_456_k3_categoryA_vertex_stats.json"""
    match = re.match(r'cluster_(\d+)_k(\d+)_category([A-Z])_', filename)
    if match:
        return {
            'cluster_id': int(match.group(1)),
            'k': int(match.group(2)),
            'category': match.group(3)
        }
    return None

def main():
    # Find all stats files
    stats_files = list(OUTPUT_DIR.glob("n*/*_vertex_stats.json"))
    print(f"Found {len(stats_files)} vertex stats files")

    # Group by n_clusters
    clusters_by_n = {}
    for stats_path in stats_files:
        n_clusters = int(stats_path.parent.name[1:])  # n768 -> 768
        if n_clusters not in clusters_by_n:
            clusters_by_n[n_clusters] = []
        clusters_by_n[n_clusters].append(stats_path)

    # Load CSVs for latent_indices
    csv_data = {}
    for n_clusters in clusters_by_n.keys():
        csv_path = CSV_DIR / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Get unique cluster info (first row per cluster)
            for cluster_id, group in df.groupby('cluster_id'):
                row = group.iloc[0]
                csv_data[(n_clusters, cluster_id)] = {
                    'latent_indices': ast.literal_eval(row['latent_indices']),
                    'n_latents': int(row['n_latents'])
                }
            print(f"Loaded CSV for n={n_clusters}: {len(df.groupby('cluster_id'))} clusters")
        else:
            print(f"WARNING: CSV not found at {csv_path}")

    # Build cluster metadata
    all_clusters = []
    vertex_collection_results = []

    for n_clusters, stats_paths in sorted(clusters_by_n.items()):
        for stats_path in stats_paths:
            # Load stats
            with open(stats_path) as f:
                stats = json.load(f)

            cluster_id = stats['cluster_id']
            k = stats['k']
            category = stats['category']

            # Get latent info from CSV
            csv_info = csv_data.get((n_clusters, cluster_id), {})
            latent_indices = csv_info.get('latent_indices', [])
            n_latents = csv_info.get('n_latents', len(latent_indices))

            # Build cluster metadata
            cluster_meta = {
                'n_clusters': n_clusters,
                'cluster_id': cluster_id,
                'arch_elbow_k': k,
                'category': category,
                'latent_indices': latent_indices,
                'n_latents': n_latents,
            }
            all_clusters.append(cluster_meta)

            # Add to vertex collection results
            vertex_collection_results.append(stats)

            print(f"  n={n_clusters} cluster={cluster_id} k={k} cat={category} latents={n_latents}")

    # Build manifest
    manifest = {
        'total_clusters': len(all_clusters),
        'n_clusters_list': sorted(clusters_by_n.keys()),
        'model_name': 'gemma-2-9b',
        'tokenizer': 'GemmaTokenizerFast',
        'sae_release': 'gemma-scope-9b-pt-res',
        'sae_id': 'layer_20/width_16k/average_l0_68',
        'clusters': all_clusters,
        'vertex_collection': {
            'enabled': True,
            'samples_per_vertex_target': 1000,
            'distance_threshold': 0.02,
            'results': vertex_collection_results
        }
    }

    # Save
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved manifest to {MANIFEST_PATH}")
    print(f"Total clusters: {len(all_clusters)}")
    for n in sorted(clusters_by_n.keys()):
        print(f"  n={n}: {len(clusters_by_n[n])} clusters")

if __name__ == '__main__':
    main()
