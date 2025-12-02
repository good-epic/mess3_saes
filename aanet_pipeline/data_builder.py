from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import jax
import numpy as np
import torch

from BatchTopK.sae import TopKSAE

from mess3_gmg_analysis_utils import sae_decode_features, sae_encode_features
from training_and_analysis_utils import _generate_sequences, _tokens_from_observations

from .cluster_summary import AAnetDescriptor, AAnetDatasetResult


def _prepare_cluster_indices(
    descriptors: Sequence[AAnetDescriptor],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    index_map: Dict[int, torch.Tensor] = {}
    for desc in descriptors:
        if desc.latent_indices:
            index_map[desc.cluster_id] = torch.tensor(desc.latent_indices, device=device, dtype=torch.long)
        else:
            index_map[desc.cluster_id] = torch.empty((0,), device=device, dtype=torch.long)
    return index_map


def build_aanet_datasets(
    *,
    model,
    sampler,
    layer_hook: str,
    sae: TopKSAE,
    aanet_descriptors: Sequence[AAnetDescriptor],
    batch_size: int,
    seq_len: int,
    num_batches: int,
    activation_threshold: float,
    device: torch.device,
    max_samples_per_cluster: int | None = None,
    min_cluster_samples: int = 0,
    seed: int = 0,
    token_positions: Sequence[int] | None = None,
) -> Tuple[Dict[int, AAnetDatasetResult], jax.Array]:
    sae.eval()
    model.eval()

    rng_key = jax.random.PRNGKey(seed)
    cluster_indices = _prepare_cluster_indices(aanet_descriptors, device)
    storage: Dict[int, list[torch.Tensor]] = {desc.cluster_id: [] for desc in aanet_descriptors}
    total_counts: Dict[int, int] = {desc.cluster_id: 0 for desc in aanet_descriptors}
    kept_counts: Dict[int, int] = {desc.cluster_id: 0 for desc in aanet_descriptors}

    for _ in range(num_batches):
        rng_key, states, observations = _generate_sequences(
            rng_key,
            batch_size=batch_size,
            sequence_len=seq_len,
            source=sampler,
        )
        tokens = _tokens_from_observations(observations, device=str(device))
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[layer_hook])
            acts = cache[layer_hook]
        if token_positions:
            seq_len_current = acts.shape[1]
            for pos in token_positions:
                if pos < 0 or pos >= seq_len_current:
                    raise ValueError(f"Token index {pos} is out of bounds for sequence length {seq_len_current}")
            acts = acts[:, token_positions, :]
        acts_flat = acts.reshape(-1, acts.shape[-1]).to(device)
        feature_acts, x_mean, x_std = sae_encode_features(sae, acts_flat)
        feature_acts = feature_acts.detach()
        zero_template = torch.zeros_like(feature_acts, device=device)

        for desc in aanet_descriptors:
            indices = cluster_indices[desc.cluster_id]
            if indices.numel() == 0:
                continue
            subset = feature_acts[:, indices]
            if activation_threshold > 0.0:
                active_mask = (subset > activation_threshold).any(dim=1)
            else:
                active_mask = (subset > 0).any(dim=1)
            total_counts[desc.cluster_id] += int(feature_acts.shape[0])
            kept = int(active_mask.sum().item())
            if kept == 0:
                continue
            kept_counts[desc.cluster_id] += kept
            cluster_latents = zero_template.clone()
            cluster_latents[:, indices] = subset
            recon = sae_decode_features(sae, cluster_latents, x_mean, x_std)
            selected = recon[active_mask].detach().cpu()
            storage[desc.cluster_id].append(selected)

    results: Dict[int, AAnetDatasetResult] = {}
    for desc in aanet_descriptors:
        tensors = storage[desc.cluster_id]
        if tensors:
            data = torch.cat(tensors, dim=0)
        else:
            data = torch.empty((0, sae.cfg["act_size"]), dtype=torch.float32)
        if max_samples_per_cluster is not None and data.shape[0] > max_samples_per_cluster:
            perm = torch.randperm(data.shape[0])[:max_samples_per_cluster]
            data = data[perm]
        kept = kept_counts[desc.cluster_id]
        total = total_counts[desc.cluster_id]
        ignored_fraction = 1.0 - (kept / total) if total > 0 else 1.0
        results[desc.cluster_id] = AAnetDatasetResult(
            descriptor=desc,
            data=data,
            kept_samples=kept,
            total_samples=total,
            ignored_fraction=ignored_fraction,
        )

    for desc in aanet_descriptors:
        if results[desc.cluster_id].data.shape[0] < min_cluster_samples and min_cluster_samples > 0:
            # record class but keep data as-is; caller can decide how to handle
            pass

    return results, rng_key
