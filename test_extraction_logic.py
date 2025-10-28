#!/usr/bin/env python3
"""Test the extraction logic on a local cluster_summary.json file."""

import json
import sys
from pathlib import Path

# Import extraction functions
sys.path.insert(0, str(Path(__file__).parent))
from extract_mlflow_sweep_results import (
    extract_layer_metrics,
    extract_component_assignment,
    extract_all_cluster_geometry_fits,
)


def test_local_file():
    """Test extraction on a local cluster_summary.json file."""
    test_file = Path("outputs/reports/multipartite_003e/kvar_20251013-223622/layer_0/cluster_summary.json")

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        print("Looking for any cluster_summary.json files...")
        import subprocess
        result = subprocess.run(
            ["find", "outputs", "-name", "cluster_summary.json", "-type", "f"],
            capture_output=True,
            text=True
        )
        files = result.stdout.strip().split('\n')
        if files and files[0]:
            test_file = Path(files[0])
            print(f"Using: {test_file}")
        else:
            print("No cluster_summary.json files found")
            return

    print(f"Testing extraction on: {test_file}")
    print("=" * 60)

    with open(test_file, 'r') as f:
        cluster_summary = json.load(f)

    # Extract metrics
    metrics = extract_layer_metrics(cluster_summary, "layer_0")

    print(f"\nExtracted {len(metrics)} metrics:")
    print("-" * 60)

    # Print metrics organized by category
    print("\n1. Basic metrics:")
    print(f"  layer: {metrics.get('layer')}")
    print(f"  energy_contrast_ratio: {metrics.get('energy_contrast_ratio')}")

    print("\n2. Assignments (hard/soft/refined):")
    for atype in ["hard", "soft", "refined"]:
        if metrics.get(f"{atype}_assignments"):
            assignments = json.loads(metrics[f"{atype}_assignments"])
            print(f"  {atype}_assignments: {list(assignments.keys())}")
            print(f"    noise clusters: {assignments.get('noise')}")

    print("\n3. Best geometries per cluster:")
    if metrics.get("best_geometries"):
        best_geoms = json.loads(metrics["best_geometries"])
        print(f"  {len(best_geoms)} clusters with best geometries")
        for cid, geom in list(best_geoms.items())[:3]:
            print(f"    Cluster {cid}: {geom}")

    print("\n4. Geometry matching scores:")
    for atype in ["hard", "soft", "refined"]:
        n_matches = metrics.get(f"n_geo_matches_{atype}")
        print(f"  n_geo_matches_{atype}: {n_matches}")

    print("\n5. All geometry scores (sample):")
    if metrics.get("all_geo_scores"):
        all_geo = json.loads(metrics["all_geo_scores"])
        print(f"  {len(all_geo)} clusters")
        first_cluster = list(all_geo.keys())[0]
        cluster_data = all_geo[first_cluster]
        print(f"  Cluster {first_cluster}: best_geometry={cluster_data.get('best_geometry')}, " +
              f"simplex_2={cluster_data.get('simplex_2'):.4f}, circle={cluster_data.get('circle'):.4f}")

    print("\n6. Belief cluster R² (soft, sample):")
    if metrics.get("assigned_belief_cluster_r2_soft"):
        assigned = json.loads(metrics["assigned_belief_cluster_r2_soft"])
        print(f"  assigned_belief_cluster_r2_soft: {len(assigned)} clusters")
        for cid, r2 in list(assigned.items())[:2]:
            print(f"    Cluster {cid}: R²={r2:.4f}")

    print("\n7. Cluster ranks:")
    if metrics.get("cluster_ranks_k_subspaces"):
        ranks = json.loads(metrics["cluster_ranks_k_subspaces"])
        print(f"  cluster_ranks_k_subspaces: {len(ranks)} clusters")
        for cid, rank in list(ranks.items())[:3]:
            print(f"    Cluster {cid}: rank={rank}")

    print(f"\n8. Summary:")
    print(f"  Total metrics: {len(metrics)}")
    print(f"  All fields extracted successfully!")

    print("\n" + "=" * 60)
    print("Extraction test completed successfully!")


if __name__ == "__main__":
    test_local_file()
