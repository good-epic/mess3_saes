#!/usr/bin/env python3
"""
Standalone co-occurrence clustering comparison.

Collects co-occurrence statistics, clusters using multiple metrics,
and compares against existing geometry-based k-subspaces clustering.

Usage:
    python scratch/cooc_clustering_comparison.py \
        --model_name "gemma-2-9b" \
        --sae_release "gemma-scope-9b-pt-res" \
        --sae_id "layer_20/width_16k/average_l0_68" \
        --n_clusters_list 512 768 \
        --geometry_clustering_dir outputs/real_data_analysis_canonical \
        --cooc_cache_path outputs/cooc_comparison/cooc_stats.npz \
        --output_dir outputs/cooc_comparison \
        --total_tokens 10_000_000
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from clustering.affinity_metrics import (
    CooccurrenceStats,
    collect_cooccurrence_stats,
    cooccurrence_affinity_from_stats,
)
from subspace_clustering_utils import k_subspaces_clustering
from mess3_gmg_analysis_utils import sae_encode_features
from real_data_utils import RealDataSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Co-occurrence clustering comparison")

    # Model & SAE
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/average_l0_68")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)

    # Data sampling
    parser.add_argument("--hf_dataset", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--hf_subset_name", type=str, default="sample-10BT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--shuffle_buffer_size", type=int, default=50000)
    parser.add_argument("--max_doc_tokens", type=int, default=3000)
    parser.add_argument("--prefetch_size", type=int, default=1024)

    # Co-occurrence collection
    parser.add_argument("--cooc_cache_path", type=str, default=None,
                        help="Path to save/load CooccurrenceStats .npz. Collect once, reuse for all metrics.")
    parser.add_argument("--total_tokens", type=int, default=10_000_000,
                        help="Target number of token positions for co-occurrence collection")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--activation_threshold", type=float, default=1e-6)

    # Clustering
    parser.add_argument("--n_clusters_list", type=int, nargs="+", required=True,
                        help="List of cluster counts to produce (e.g. 512 768)")
    parser.add_argument("--geometry_clustering_dir", type=str, required=True,
                        help="Path to existing k-subspaces results (e.g. outputs/real_data_analysis_canonical)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_eigenvectors", type=int, default=None,
                        help="Number of Laplacian eigenvectors. Default: match n_clusters.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", type=str, default="phi,mutual_info,ami,jaccard",
                        help="Comma-separated co-occurrence metrics to compare")

    return parser.parse_args()


def load_model_and_sae(args):
    """Load HookedTransformer and SAE."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    print(f"Loading model: {args.model_name}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        center_unembed=False,
        center_writing_weights=False,
        dtype="bfloat16",
        cache_dir=args.cache_dir,
    )

    print(f"Loading SAE: {args.sae_release}/{args.sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    sae.eval()

    # Derive hook_name from sae_id
    # e.g. "layer_20/width_16k/average_l0_68" -> "blocks.20.hook_resid_post"
    parts = args.sae_id.split("/")
    layer_num = parts[0].replace("layer_", "")
    hook_name = f"blocks.{layer_num}.hook_resid_post"
    print(f"Hook: {hook_name}")

    return model, sae, hook_name


def collect_or_load_cooc_stats(args, model, sae, hook_name):
    """Collect co-occurrence stats or load from cache."""
    if args.cooc_cache_path and Path(args.cooc_cache_path).exists():
        print(f"\nLoading cached co-occurrence stats from {args.cooc_cache_path}")
        stats = CooccurrenceStats.load(args.cooc_cache_path)
        print(stats.summary())
        return stats

    print(f"\nCollecting co-occurrence statistics...")

    # Create sampler
    hf_subset = None if args.hf_subset_name.lower() == "none" else args.hf_subset_name
    sampler = RealDataSampler(
        model,
        hf_dataset=args.hf_dataset,
        hf_subset_name=hf_subset,
        split=args.dataset_split,
        seed=args.seed,
        prefetch_size=args.prefetch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_doc_tokens=args.max_doc_tokens,
    )

    # Compute n_batches from total_tokens
    tokens_per_batch = args.batch_size * (args.seq_len - 1)  # -1 for BOS skip
    n_batches = args.total_tokens // tokens_per_batch
    actual_tokens = n_batches * tokens_per_batch
    print(f"  Target: {args.total_tokens:,} tokens")
    print(f"  Batch: {args.batch_size} seqs x {args.seq_len} tokens ({tokens_per_batch:,} usable/batch)")
    print(f"  Batches: {n_batches:,} ({actual_tokens:,} actual tokens)")

    stats = collect_cooccurrence_stats(
        model,
        sae,
        sampler,
        hook_name,
        sae_encode_fn=sae_encode_features,
        n_batches=n_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        activation_threshold=args.activation_threshold,
        device=args.device,
        show_progress=True,
        skip_special_tokens=True,
    )

    print(stats.summary())

    if args.cooc_cache_path:
        stats.save(args.cooc_cache_path)

    return stats


def compute_laplacian_eigenvectors(affinity, n_vectors):
    """Compute bottom eigenvectors of the normalized Laplacian."""
    n = affinity.shape[0]
    d = np.array(affinity.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    # Compute D^{-1/2} A D^{-1/2} efficiently
    affinity_norm = affinity * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    L_norm = np.eye(n) - affinity_norm

    # Request n_vectors + 1 to skip the trivial eigenvector (eigenvalue ≈ 0)
    k = min(n_vectors + 1, n - 1)
    print(f"  Computing {k} smallest eigenvectors of {n}x{n} Laplacian...")
    eigenvalues, eigenvectors = eigsh(csr_matrix(L_norm), k=k, which='SM')

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip trivial eigenvector (eigenvalue ≈ 0) if present
    if eigenvalues[0] < 1e-8:
        eigenvectors = eigenvectors[:, 1:n_vectors + 1]
        eigenvalues = eigenvalues[1:n_vectors + 1]
    else:
        eigenvectors = eigenvectors[:, :n_vectors]
        eigenvalues = eigenvalues[:n_vectors]

    # L2-normalize rows
    row_norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / np.maximum(row_norms, 1e-12)

    print(f"  Eigenvalue range: [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")
    return eigenvectors


def cluster_from_eigenvectors(eigenvectors, n_clusters, seed):
    """Run k-subspaces clustering on spectral embedding."""
    print(f"  Running k-subspaces on {eigenvectors.shape} embedding, n_clusters={n_clusters}...")
    result = k_subspaces_clustering(
        eigenvectors,
        n_clusters=n_clusters,
        random_state=seed,
        variance_threshold=0.95,
        gap_threshold=2.0,
    )
    labels = result.cluster_labels
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    return labels


def load_geometry_clustering(geometry_dir, n_clusters):
    """Load existing k-subspaces clustering result."""
    pkl_path = os.path.join(geometry_dir, f"clusters_{n_clusters}", "clustering_result.pkl")
    if not os.path.exists(pkl_path):
        print(f"  WARNING: {pkl_path} not found")
        return None
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    print(f"  Loaded geometry clustering: {result.n_clusters} clusters, "
          f"{(result.cluster_labels >= 0).sum():,} active latents")
    return result.cluster_labels


def compute_within_cluster_cosine(labels, decoder_np, n_clusters):
    """Compute within-cluster cosine similarity statistics."""
    # Normalize decoder directions
    norms = np.linalg.norm(decoder_np, axis=1, keepdims=True)
    decoder_normed = decoder_np / np.maximum(norms, 1e-12)

    cluster_cos_means = []
    cluster_sizes = []

    for cid in range(n_clusters):
        mask = labels == cid
        size = mask.sum()
        cluster_sizes.append(int(size))

        if size < 2:
            continue

        cluster_dirs = decoder_normed[mask]
        # For large clusters, subsample to avoid O(n^2) blowup
        if size > 200:
            rng = np.random.RandomState(42)
            idx = rng.choice(size, 200, replace=False)
            cluster_dirs = cluster_dirs[idx]

        cos_sim = cosine_similarity(cluster_dirs)
        triu_idx = np.triu_indices(len(cos_sim), k=1)
        mean_cos = float(cos_sim[triu_idx].mean())
        cluster_cos_means.append(mean_cos)

    cluster_sizes = np.array(cluster_sizes)

    return {
        "cos_sim_mean": float(np.mean(cluster_cos_means)) if cluster_cos_means else 0.0,
        "cos_sim_median": float(np.median(cluster_cos_means)) if cluster_cos_means else 0.0,
        "cos_sim_std": float(np.std(cluster_cos_means)) if cluster_cos_means else 0.0,
        "n_clusters_gt10": int((cluster_sizes > 10).sum()),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "mean_cluster_size": float(np.mean(cluster_sizes)),
        "n_singletons": int((cluster_sizes <= 1).sum()),
        "n_empty": int((cluster_sizes == 0).sum()),
        "max_cluster_size": int(cluster_sizes.max()) if len(cluster_sizes) > 0 else 0,
    }


def compare_clusterings(all_labels, decoder_np, n_clusters):
    """Compare all clustering methods pairwise and individually."""
    names = list(all_labels.keys())

    # Pairwise comparisons
    pairwise = {}
    for name_a, name_b in combinations(names, 2):
        la = all_labels[name_a]
        lb = all_labels[name_b]
        # Only compare latents active in both
        valid = (la >= 0) & (lb >= 0)
        la_v = la[valid]
        lb_v = lb[valid]
        ari = adjusted_rand_score(la_v, lb_v)
        nmi = normalized_mutual_info_score(la_v, lb_v, average_method='arithmetic')
        pair_key = f"{name_a} vs {name_b}"
        pairwise[pair_key] = {"ari": ari, "nmi": nmi, "n_shared": int(valid.sum())}

    # Per-clustering stats
    per_method = {}
    for name, labels in all_labels.items():
        active_labels = labels[labels >= 0]
        actual_n = len(set(active_labels))
        stats = compute_within_cluster_cosine(labels, decoder_np, max(labels) + 1 if len(labels) > 0 else 0)
        stats["actual_n_clusters"] = actual_n
        per_method[name] = stats

    return pairwise, per_method


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics = [m.strip() for m in args.metrics.split(",")]
    print(f"Metrics to compare: {metrics}")

    # Step 1: Load model + SAE
    print("\n" + "=" * 60)
    print("LOADING MODEL AND SAE")
    print("=" * 60)
    model, sae, hook_name = load_model_and_sae(args)
    decoder_np = sae.W_dec.detach().cpu().float().numpy()
    n_features = decoder_np.shape[0]
    print(f"SAE features: {n_features}, d_model: {decoder_np.shape[1]}")

    # Step 2: Collect or load co-occurrence stats
    print("\n" + "=" * 60)
    print("CO-OCCURRENCE STATISTICS")
    print("=" * 60)
    stats = collect_or_load_cooc_stats(args, model, sae, hook_name)

    # Free model from GPU memory — no longer needed
    del model
    torch.cuda.empty_cache()

    # Step 3 & 4: For each n_clusters value
    for n_clusters in args.n_clusters_list:
        print(f"\n{'=' * 60}")
        print(f"CLUSTERING WITH n_clusters={n_clusters}")
        print("=" * 60)

        n_eig = args.n_eigenvectors if args.n_eigenvectors else n_clusters
        all_labels = {}

        # Load geometry-based clustering
        print(f"\nLoading geometry-based clustering...")
        geo_labels = load_geometry_clustering(args.geometry_clustering_dir, n_clusters)
        if geo_labels is not None:
            all_labels["geometry"] = geo_labels

        # For each co-occurrence metric
        for metric in metrics:
            print(f"\n--- Metric: {metric} ---")

            # Check for cached clustering result
            labels_path = os.path.join(args.output_dir, f"labels_{metric}_n{n_clusters}.npy")
            if os.path.exists(labels_path):
                print(f"  Loading cached labels from {labels_path}")
                labels = np.load(labels_path)
                all_labels[metric] = labels
                continue

            # Compute affinity matrix
            print(f"  Computing {metric} affinity matrix...")
            affinity = cooccurrence_affinity_from_stats(stats, metric)

            # Compute Laplacian eigenvectors
            eigvecs = compute_laplacian_eigenvectors(affinity, n_eig)
            del affinity  # Free memory

            # Cluster
            labels = cluster_from_eigenvectors(eigvecs, n_clusters, args.seed)
            del eigvecs

            # Pad to full feature space (mark all as active — cooc uses all features)
            all_labels[metric] = labels

            # Save labels
            np.save(labels_path, labels)
            print(f"  Saved labels to {labels_path}")

        # Step 5: Compare
        print(f"\n{'=' * 60}")
        print(f"COMPARISON (n_clusters={n_clusters})")
        print("=" * 60)

        pairwise, per_method = compare_clusterings(all_labels, decoder_np, n_clusters)

        # Print pairwise table
        print(f"\n{'Pair':<35} {'ARI':>8} {'NMI':>8}")
        print("-" * 55)
        for pair, metrics_dict in sorted(pairwise.items()):
            print(f"{pair:<35} {metrics_dict['ari']:>8.4f} {metrics_dict['nmi']:>8.4f}")

        # Print per-method table
        print(f"\n{'Method':<15} {'Clusters':>8} {'CosSim':>8} {'CosSim_med':>10} "
              f"{'Size_med':>8} {'Single':>6} {'Empty':>5}")
        print("-" * 75)
        for name, s in per_method.items():
            print(f"{name:<15} {s['actual_n_clusters']:>8} {s['cos_sim_mean']:>8.4f} "
                  f"{s['cos_sim_median']:>10.4f} {s['median_cluster_size']:>8.1f} "
                  f"{s['n_singletons']:>6} {s['n_empty']:>5}")

        # Save results
        results = {
            "n_clusters": n_clusters,
            "n_eigenvectors": n_eig,
            "pairwise": pairwise,
            "per_method": per_method,
        }
        results_path = os.path.join(args.output_dir, f"comparison_n{n_clusters}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
