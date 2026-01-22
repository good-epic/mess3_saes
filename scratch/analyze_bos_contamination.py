#!/usr/bin/env python3
"""
Analyze what percentage of vertex samples are triggered by BOS/special tokens.
"""

import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("outputs/selected_clusters_canonical")

def analyze_samples():
    results = []

    for samples_file in sorted(OUTPUT_DIR.glob("n*/*_vertex_samples.jsonl")):
        # Parse filename
        n_clusters = int(samples_file.parent.name[1:])  # n512 -> 512
        filename = samples_file.name

        # Extract cluster info from filename
        parts = filename.replace("_vertex_samples.jsonl", "").split("_")
        cluster_id = int(parts[1])
        k = int(parts[2][1:])  # k3 -> 3
        category = parts[3].replace("category", "")

        # Count samples per vertex, and BOS samples per vertex
        vertex_total = defaultdict(int)
        vertex_bos = defaultdict(int)
        vertex_has_any_bos = defaultdict(int)  # samples where ANY trigger is BOS

        with open(samples_file) as f:
            for line in f:
                sample = json.loads(line)
                vertex_id = sample["vertex_id"]
                trigger_indices = sample["trigger_token_indices"]

                vertex_total[vertex_id] += 1

                # Check if any trigger is at position 0 (BOS)
                has_bos = 0 in trigger_indices
                if has_bos:
                    vertex_has_any_bos[vertex_id] += 1

                # Check if ALL triggers are at position 0
                all_bos = all(idx == 0 for idx in trigger_indices)
                if all_bos:
                    vertex_bos[vertex_id] += 1

        # Store results
        cluster_result = {
            "n_clusters": n_clusters,
            "cluster_id": cluster_id,
            "k": k,
            "category": category,
            "vertices": {}
        }

        for v in range(k):
            total = vertex_total[v]
            all_bos = vertex_bos[v]
            any_bos = vertex_has_any_bos[v]

            cluster_result["vertices"][v] = {
                "total": total,
                "all_triggers_bos": all_bos,
                "any_trigger_bos": any_bos,
                "pct_all_bos": (all_bos / total * 100) if total > 0 else 0,
                "pct_any_bos": (any_bos / total * 100) if total > 0 else 0,
            }

        results.append(cluster_result)

    return results

def print_results(results):
    print("=" * 100)
    print("BOS/SPECIAL TOKEN CONTAMINATION ANALYSIS")
    print("=" * 100)
    print()

    # Summary table
    print(f"{'Cluster':<20} {'Cat':<4} {'Vertex':<8} {'Total':<10} {'All BOS':<10} {'%All BOS':<10} {'Any BOS':<10} {'%Any BOS':<10}")
    print("-" * 100)

    problem_clusters = []

    for r in sorted(results, key=lambda x: (x["n_clusters"], x["cluster_id"])):
        cluster_label = f"n{r['n_clusters']}_c{r['cluster_id']}_k{r['k']}"

        has_problem = False
        for v, stats in sorted(r["vertices"].items()):
            pct_all_bos = stats["pct_all_bos"]
            pct_any_bos = stats["pct_any_bos"]

            # Flag if >50% of samples are all-BOS
            flag = " ⚠️" if pct_all_bos > 50 else ""
            if pct_all_bos > 50:
                has_problem = True

            print(f"{cluster_label:<20} {r['category']:<4} {v:<8} {stats['total']:<10} {stats['all_triggers_bos']:<10} {pct_all_bos:<10.1f} {stats['any_trigger_bos']:<10} {pct_any_bos:<10.1f}{flag}")

        if has_problem:
            problem_clusters.append(r)

        print()  # Blank line between clusters

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total clusters analyzed: {len(results)}")
    print(f"Clusters with at least one vertex >50% BOS-only: {len(problem_clusters)}")
    print()

    if problem_clusters:
        print("Problem clusters (vertices with >50% BOS-only samples):")
        for r in problem_clusters:
            cluster_label = f"n{r['n_clusters']}_c{r['cluster_id']}_k{r['k']} (cat {r['category']})"
            problem_vertices = [v for v, s in r["vertices"].items() if s["pct_all_bos"] > 50]
            print(f"  {cluster_label}: vertices {problem_vertices}")

if __name__ == "__main__":
    results = analyze_samples()
    print_results(results)
