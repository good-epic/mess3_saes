#!/usr/bin/env python3
"""
Prepare vertex samples for cluster interpretation.

Loads vertex samples from completed AANet runs and organizes them
for downstream interpretation tasks.

Usage:
    python prepare_vertex_samples.py \
        --manifest /workspace/outputs/selected_clusters_canonical/manifest.json \
        --output_dir outputs/interpretations/prepared_samples \
        --max_samples_per_vertex 1000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_manifest(manifest_path):
    """Load the manifest from the AANet training run."""
    print(f"Loading manifest from {manifest_path}")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"  Found {manifest['total_clusters']} clusters")
    print(f"  Model: {manifest['model_name']}")
    print(f"  SAE: {manifest['sae_release']}/{manifest['sae_id']}")

    return manifest


def load_vertex_samples(samples_path, max_samples_per_vertex=None):
    """Load vertex samples from JSONL file."""
    if not os.path.exists(samples_path):
        print(f"  WARNING: Samples file not found: {samples_path}")
        return None

    samples_by_vertex = defaultdict(list)

    with open(samples_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            vertex_id = sample['vertex_id']
            samples_by_vertex[vertex_id].append(sample)

    # Limit samples if requested
    if max_samples_per_vertex:
        for vertex_id in samples_by_vertex:
            samples_by_vertex[vertex_id] = samples_by_vertex[vertex_id][:max_samples_per_vertex]

    return dict(samples_by_vertex)


def prepare_cluster_samples(cluster_metadata, max_samples_per_vertex):
    """Prepare samples for one cluster."""
    cluster_id = cluster_metadata['cluster_id']
    n_clusters = cluster_metadata['n_clusters']
    k = cluster_metadata['arch_elbow_k']

    print(f"\nProcessing cluster {cluster_id} (n={n_clusters}, k={k})")

    # Load vertex samples
    if 'vertex_samples_path' in cluster_metadata:
        samples_path = cluster_metadata['vertex_samples_path']
    else:
        # Samples weren't collected for this cluster
        print(f"  No vertex samples collected for this cluster")
        return None

    samples_by_vertex = load_vertex_samples(samples_path, max_samples_per_vertex)

    if samples_by_vertex is None:
        return None

    # Count samples
    total_samples = sum(len(samples) for samples in samples_by_vertex.values())
    print(f"  Loaded {total_samples} samples across {len(samples_by_vertex)} vertices")
    for vertex_id, samples in sorted(samples_by_vertex.items()):
        print(f"    Vertex {vertex_id}: {len(samples)} samples")

    # Create prepared sample structure
    prepared = {
        'cluster_id': cluster_id,
        'n_clusters': n_clusters,
        'k': k,
        'n_latents': cluster_metadata['n_latents'],
        'latent_indices': cluster_metadata['latent_indices'],
        'category': cluster_metadata.get('category', 'unknown'),
        'vertices': samples_by_vertex,
        'total_samples_available': total_samples,
    }

    return prepared


def main():
    parser = argparse.ArgumentParser(description="Prepare vertex samples for interpretation")
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to manifest.json from AANet training run')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save prepared samples')
    parser.add_argument('--max_samples_per_vertex', type=int, default=None,
                        help='Maximum samples to keep per vertex (default: all)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    manifest = load_manifest(args.manifest)

    # Process each cluster
    prepared_count = 0
    skipped_count = 0

    for cluster_metadata in manifest['clusters']:
        prepared = prepare_cluster_samples(cluster_metadata, args.max_samples_per_vertex)

        if prepared is None:
            skipped_count += 1
            continue

        # Save prepared samples
        cluster_id = prepared['cluster_id']
        n_clusters = prepared['n_clusters']
        output_path = os.path.join(args.output_dir, f"cluster_{n_clusters}_{cluster_id}.json")

        with open(output_path, 'w') as f:
            json.dump(prepared, f, indent=2)

        print(f"  Saved to {output_path}")
        prepared_count += 1

    # Summary
    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print("="*80)
    print(f"Prepared: {prepared_count} clusters")
    print(f"Skipped: {skipped_count} clusters (no vertex samples)")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
