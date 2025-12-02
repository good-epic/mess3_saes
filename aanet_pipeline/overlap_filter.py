from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .cluster_summary import AAnetDescriptor


def _orthonormalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((vectors.shape[1], 0))
    u, s, vh = np.linalg.svd(vectors, full_matrices=False)
    rank = np.sum(s > 1e-8)
    if rank == 0:
        return np.zeros((vectors.shape[1], 0))
    return vh[:rank].T


def _projection_error(vec: np.ndarray, basis: np.ndarray) -> float:
    if basis.size == 0:
        return float(np.dot(vec, vec))
    coeffs = basis.T @ vec
    residual = vec - basis @ coeffs
    return float(np.dot(residual, residual))


def _build_cluster_bases(
    decoder_vectors: np.ndarray,
    descriptors: Sequence[AAnetDescriptor],
) -> Dict[int, np.ndarray]:
    bases: Dict[int, np.ndarray] = {}
    for desc in descriptors:
        if desc.latent_indices:
            vectors = decoder_vectors[desc.latent_indices]
            bases[desc.cluster_id] = _orthonormalize(vectors)
        else:
            bases[desc.cluster_id] = np.zeros((decoder_vectors.shape[1], 0))
    return bases


def _compute_second_errors(
    decoder_vectors: np.ndarray,
    descriptors: Sequence[AAnetDescriptor],
    cluster_bases: Dict[int, np.ndarray],
) -> Dict[int, float]:
    cluster_ids = sorted(cluster_bases.keys())
    unique_latents: List[int] = sorted({idx for desc in descriptors for idx in desc.latent_indices})
    second_errors: Dict[int, float] = {}

    for latent_idx in unique_latents:
        vec = decoder_vectors[latent_idx]
        errors = [
            _projection_error(vec, cluster_bases[cluster_id])
            for cluster_id in cluster_ids
        ]
        errors.sort()
        if len(errors) >= 2:
            second_errors[latent_idx] = float(errors[1])
        elif errors:
            second_errors[latent_idx] = float(errors[0])
        else:
            second_errors[latent_idx] = float("inf")
    return second_errors


def drop_overlapping_latents(
    descriptors: Sequence[AAnetDescriptor],
    sae: TopKSAE,
    threshold: float = 0.5,
) -> Sequence[AAnetDescriptor]:
    decoder_np = sae.decoder.detach().cpu().numpy()
    cluster_bases = _build_cluster_bases(decoder_np, descriptors)
    second_errors = _compute_second_errors(decoder_np, descriptors, cluster_bases)

    per_cluster_kept: Dict[int, List[int]] = {}

    for desc in descriptors:
        keep: List[int] = []
        drop: List[int] = []
        for latent_idx in desc.latent_indices:
            if second_errors.get(latent_idx, float("inf")) <= threshold:
                drop.append(latent_idx)
            else:
                keep.append(latent_idx)
        per_cluster_drops[desc.cluster_id] = drop
        per_cluster_kept[desc.cluster_id] = keep
        updated.append(
            replace(desc, latent_indices=keep)
        )

    total_before = sum(len(desc.latent_indices) for desc in descriptors)
    total_after = sum(len(desc.latent_indices) for desc in updated)
    report = {
        "threshold": threshold,
        "total_latents_before": total_before,
        "total_latents_after": total_after,
        "total_dropped": total_before - total_after,
        "per_cluster_dropped": {str(cid): drops for cid, drops in per_cluster_drops.items()},
        "per_cluster_kept": {str(cid): kept for cid, kept in per_cluster_kept.items()},
        "second_error_stats": {
            "min": float(min(second_errors.values(), default=float("inf"))),
            "median": float(np.median(list(second_errors.values())) if second_errors else float("inf")),
            "max": float(max(second_errors.values(), default=float("-inf"))),
        },
    }
    return updated, report
