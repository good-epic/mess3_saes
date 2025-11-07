#!/usr/bin/env python3

"""
Compute projection errors for SAE decoder directions against cluster subspaces,
and plot histograms of the sorted errors (1st smallest, 2nd smallest, ...).

Usage example:
python scratch/cluster_projection_error_histograms.py \
  --layer 1 \
  --sae-root outputs/saes/multipartite_003e \
  --cluster-summary outputs/reports/multipartite_003e/top_r2_run_layer_1_cluster_summary.json \
  --output-dir scratch/projection_histograms_layer1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_sae_decoder(sae_path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(sae_path, map_location=device)
    state_dict = payload["state_dict"]
    decoder = state_dict["W_dec"]  # shape: (dict_size, act_size)
    return decoder.detach().cpu()


def orthonormalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] == 0:
        return np.zeros((vectors.shape[1], 0))
    u, s, vh = np.linalg.svd(vectors, full_matrices=False)
    rank = np.sum(s > 1e-8)
    basis = vh[:rank].T  # shape (dim, rank)
    return basis


def compute_projection_error(vec: np.ndarray, basis: np.ndarray) -> float:
    if basis.shape[0] == 0:
        return float(np.linalg.norm(vec) ** 2)
    projection = basis.T @ vec
    residual = vec - basis @ projection
    return float(residual @ residual)


def compute_errors(decoder: torch.Tensor, clusters: Dict[int, List[int]]) -> np.ndarray:
    decoder_np = decoder.numpy()
    n_latents = decoder_np.shape[0]
    errors = np.zeros((n_latents, len(clusters)), dtype=np.float64)

    cluster_bases: Dict[int, np.ndarray] = {}
    for cluster_id, indices in clusters.items():
        vectors = decoder_np[indices]
        basis = orthonormalize_vectors(vectors)
        cluster_bases[cluster_id] = basis

    cluster_ids = sorted(clusters.keys())
    for i in range(n_latents):
        vec = decoder_np[i]
        for j, cluster_id in enumerate(cluster_ids):
            basis = cluster_bases[cluster_id]
            errors[i, j] = compute_projection_error(vec, basis)

    return errors


def plot_histograms(
    errors: np.ndarray,
    output_dir: Path,
    *,
    max_rank: int | None = None,
    clip_threshold: float | None = None,
    zoom_rank: int | None = None,
    zoom_threshold: float | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_errors = np.sort(errors, axis=1)
    n_clusters = sorted_errors.shape[1]
    ranks_to_plot = min(n_clusters, max_rank if max_rank else n_clusters)

    for rank in range(ranks_to_plot):
        data = sorted_errors[:, rank]
        if clip_threshold is not None:
            data = data[data <= clip_threshold]
        plt.figure(figsize=(7, 4))
        base_bins = min(100, max(10, len(data) // 50 + 10))
        bins = base_bins * 2 if rank == 1 else base_bins
        plt.hist(data, bins=bins, alpha=0.7, color="steelblue")
        plt.xlabel("Projection error")
        plt.ylabel("Count")
        plt.title(f"Histogram of rank {rank + 1} smallest projection errors")
        plt.tight_layout()
        plt.savefig(output_dir / f"projection_error_rank{rank + 1}.png", dpi=200)
        plt.close()

        if zoom_rank is not None and zoom_threshold is not None and rank + 1 == zoom_rank:
            zoom_data = data[data <= zoom_threshold]
            if zoom_data.size > 0:
                plt.figure(figsize=(6, 4))
                base_bins = min(100, max(10, len(zoom_data) // 25 + 10))
                plt.hist(zoom_data, bins=base_bins, alpha=0.8, color="teal")
                plt.xlabel("Projection error")
                plt.ylabel("Count")
                plt.title(f"Zoomed histogram of rank {zoom_rank} errors (<= {zoom_threshold})")
                plt.tight_layout()
                plt.savefig(output_dir / f"projection_error_rank{zoom_rank}_zoom.png", dpi=200)
                plt.close()


def main():
    parser = argparse.ArgumentParser(description="Projection error histograms for cluster subspaces.")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-root", type=Path, required=True)
    parser.add_argument("--cluster-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ignore-clusters", type=int, nargs="*", default=None)
    args = parser.parse_args()

    sae_path = args.sae_root / f"layer_{args.layer}_top_k_k12.pt"
    decoder = load_sae_decoder(sae_path, device=torch.device(args.device))

    summary = json.loads(args.cluster_summary.read_text())
    clusters_raw = summary.get("clusters", {})
    clusters = {int(cid): entry["latent_indices"] for cid, entry in clusters_raw.items()}
    if args.ignore_clusters:
        for cid in args.ignore_clusters:
            clusters.pop(cid, None)

    errors = compute_errors(decoder, clusters)
    plot_histograms(
        errors,
        args.output_dir,
        max_rank=args.max_rank,
        clip_threshold=0.06,
        zoom_rank=2,
        zoom_threshold=0.002,
    )
    np.save(args.output_dir / "errors.npy", errors)


if __name__ == "__main__":
    main()
