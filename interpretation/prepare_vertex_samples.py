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
    """Load vertex samples from JSONL file.

    Returns:
        Tuple of (samples_by_vertex dict or None, trigger_stats dict, total_count)
        trigger_stats contains counts of problematic trigger words (empty, newline, tab)
    """
    if not os.path.exists(samples_path):
        print(f"  WARNING: Samples file not found: {samples_path}")
        return None, {'empty': 0, 'newline': 0, 'tab': 0}, 0

    samples_by_vertex = defaultdict(list)
    trigger_stats = {'empty': 0, 'newline': 0, 'tab': 0}
    total_count = 0

    with open(samples_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            vertex_id = sample['vertex_id']
            samples_by_vertex[vertex_id].append(sample)
            total_count += 1

            # Check for problematic trigger_words
            trigger_words = sample.get('trigger_words', [])
            for tw in trigger_words:
                if tw == '':
                    trigger_stats['empty'] += 1
                elif tw == '\n':
                    trigger_stats['newline'] += 1
                elif tw == '\t':
                    trigger_stats['tab'] += 1

    # Limit samples if requested
    if max_samples_per_vertex:
        for vertex_id in samples_by_vertex:
            samples_by_vertex[vertex_id] = samples_by_vertex[vertex_id][:max_samples_per_vertex]

    return dict(samples_by_vertex), trigger_stats, total_count


def prepare_cluster_samples(cluster_metadata, vertex_collection_results, max_samples_per_vertex, min_vertices_with_samples):
    """Prepare samples for one cluster.

    Returns:
        Tuple of (prepared_dict or None, trigger_stats dict, total_count, skip_reason or None)
    """
    cluster_id = cluster_metadata['cluster_id']
    n_clusters = cluster_metadata['n_clusters']
    k = cluster_metadata.get('arch_elbow_k', cluster_metadata.get('k'))

    print(f"\nProcessing cluster {cluster_id} (n={n_clusters}, k={k})")

    # Find vertex collection results for this cluster
    collection_result = None
    if vertex_collection_results:
        for result in vertex_collection_results:
            if result['cluster_id'] == cluster_id and result['n_clusters'] == n_clusters:
                collection_result = result
                # Update k and category from collection result if available
                k = collection_result.get('k', k)
                break

    # Determine samples path
    if 'vertex_samples_path' in cluster_metadata:
        # Old format: path directly in cluster metadata
        samples_path = cluster_metadata['vertex_samples_path']
    elif collection_result and 'all_coords_path' in collection_result:
        # New format: construct from all_coords_path
        import os
        coords_path = collection_result['all_coords_path']
        # Replace _all_barycentric_coords.jsonl with _vertex_samples.jsonl
        samples_path = coords_path.replace('_all_barycentric_coords.jsonl', '_vertex_samples.jsonl')

        # Handle local vs RunPod paths
        if samples_path.startswith('/workspace/'):
            # Convert RunPod path to local path
            samples_path = samples_path.replace('/workspace/outputs/', 'outputs/')
    else:
        # Samples weren't collected for this cluster
        print(f"  No vertex samples collected for this cluster")
        return None, {'empty': 0, 'newline': 0, 'tab': 0}, 0, "no_samples_file"

    samples_by_vertex, trigger_stats, total_count = load_vertex_samples(samples_path, max_samples_per_vertex)

    if samples_by_vertex is None:
        return None, {'empty': 0, 'newline': 0, 'tab': 0}, 0, "samples_file_not_found"

    # Count vertices with samples
    vertices_with_samples = sum(1 for v in samples_by_vertex.values() if len(v) > 0)

    # Count samples
    total_samples = sum(len(samples) for samples in samples_by_vertex.values())
    print(f"  Loaded {total_samples} samples across {vertices_with_samples} vertices (k={k})")
    for vertex_id, samples in sorted(samples_by_vertex.items()):
        print(f"    Vertex {vertex_id}: {len(samples)} samples")

    # Report problematic trigger words if any
    problematic = trigger_stats['empty'] + trigger_stats['newline'] + trigger_stats['tab']
    if problematic > 0:
        print(f"  Trigger word issues: {trigger_stats['empty']} empty, {trigger_stats['newline']} newline, {trigger_stats['tab']} tab")

    # Check minimum vertices requirement
    if vertices_with_samples < min_vertices_with_samples:
        print(f"  Skipping: only {vertices_with_samples} vertices have samples (need at least {min_vertices_with_samples})")
        return None, trigger_stats, total_count, "insufficient_vertices"

    # Get category from collection result or cluster metadata
    category = cluster_metadata.get('category', 'unknown')
    if collection_result and 'category' in collection_result:
        category = collection_result['category']

    # Create prepared sample structure
    prepared = {
        'cluster_id': cluster_id,
        'n_clusters': n_clusters,
        'k': k,
        'n_latents': cluster_metadata['n_latents'],
        'latent_indices': cluster_metadata['latent_indices'],
        'category': category,
        'vertices': samples_by_vertex,
        'total_samples_available': total_samples,
    }

    return prepared, trigger_stats, total_count, None


def main():
    parser = argparse.ArgumentParser(description="Prepare vertex samples for interpretation")
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to manifest.json from AANet training run')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save prepared samples')
    parser.add_argument('--max_samples_per_vertex', type=int, default=None,
                        help='Maximum samples to keep per vertex (default: all)')
    parser.add_argument('--min_vertices_with_samples', type=int, default=2,
                        help='Minimum number of vertices that must have samples (default: 2)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    manifest = load_manifest(args.manifest)

    # Get vertex collection results if available
    vertex_collection_results = None
    if 'vertex_collection' in manifest and manifest['vertex_collection'].get('enabled'):
        vertex_collection_results = manifest['vertex_collection'].get('results', [])
        print(f"\nFound vertex collection results for {len(vertex_collection_results)} clusters")

    # Process each cluster
    prepared_count = 0
    skip_reasons = defaultdict(int)

    # Track global trigger word statistics
    global_trigger_stats = {'empty': 0, 'newline': 0, 'tab': 0}
    global_total_count = 0

    for cluster_metadata in manifest['clusters']:
        prepared, trigger_stats, total_count, skip_reason = prepare_cluster_samples(
            cluster_metadata, vertex_collection_results,
            args.max_samples_per_vertex, args.min_vertices_with_samples
        )

        # Accumulate trigger stats
        for key in global_trigger_stats:
            global_trigger_stats[key] += trigger_stats[key]
        global_total_count += total_count

        if prepared is None:
            skip_reasons[skip_reason] += 1
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

    total_skipped = sum(skip_reasons.values())
    if total_skipped > 0:
        print(f"Skipped: {total_skipped} clusters")
        for reason, count in sorted(skip_reasons.items()):
            print(f"  - {reason}: {count}")

    print(f"Output directory: {args.output_dir}")

    # Report global trigger word statistics
    print("\n" + "-"*80)
    print("TRIGGER WORD STATISTICS")
    print("-"*80)
    if global_total_count > 0:
        print(f"Total samples processed: {global_total_count}")

        # Calculate total trigger words (samples can have multiple trigger words)
        total_trigger_words = sum(global_trigger_stats.values())

        for category, count in global_trigger_stats.items():
            if count > 0:
                print(f"  {category}: {count}")

        if global_trigger_stats['empty'] > 0:
            print("\nWARNING: Non-zero empty trigger_words count suggests special tokens may not have been filtered correctly")
        if global_trigger_stats['newline'] > 0 or global_trigger_stats['tab'] > 0:
            print("NOTE: Newline/tab trigger words may be legitimate but are worth reviewing")
    else:
        print("No samples processed")


if __name__ == '__main__':
    main()
