import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from BatchTopK.sae import TopKSAE, VanillaSAE


@dataclass
class ClusterPCAResult:
    coords: np.ndarray
    pca: PCA
    decoder_coords: Dict[int, np.ndarray]
    scale_factor: float


def load_metrics_summary(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics_summary not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_topk_l2(metrics_summary: Mapping[str, Mapping], site_filter: Iterable[str] | None = None) -> Dict[str, Dict[int, float]]:
    allowed = set(site_filter) if site_filter is not None else None
    l2_by_site: Dict[str, Dict[int, float]] = {}
    for site, site_data in metrics_summary.items():
        if allowed is not None and site not in allowed:
            continue
        topk_data = (
            site_data.get("sequence", {}).get("top_k", {})
            if isinstance(site_data, Mapping)
            else {}
        )
        site_dict: Dict[int, float] = {}
        for k_key, entry in topk_data.items():
            try:
                k_val = int(str(k_key).lstrip("k"))
            except ValueError:
                continue
            l2 = (
                entry.get("avg_last_quarter", {}).get("l2")
                if isinstance(entry, Mapping)
                else None
            )
            if l2 is None:
                continue
            site_dict[k_val] = float(l2)
        if site_dict:
            l2_by_site[site] = site_dict
    return l2_by_site


def compute_average_l2(l2_by_site: Mapping[str, Mapping[int, float]]) -> Dict[int, float]:
    accumulator: Dict[int, List[float]] = defaultdict(list)
    for site_dict in l2_by_site.values():
        for k, val in site_dict.items():
            accumulator[int(k)].append(float(val))
    return {k: float(np.mean(vals)) for k, vals in accumulator.items() if vals}


def find_elbow_k(k_values: Sequence[int], losses: Sequence[float], prefer_high_k: bool = True, tolerance: float = 0.05) -> int:
    if len(k_values) != len(losses):
        raise ValueError("k_values and losses must have the same length")
    if len(k_values) == 0:
        raise ValueError("k_values is empty")
    k_arr = np.array(k_values, dtype=float)
    loss_arr = np.array(losses, dtype=float)
    sort_idx = np.argsort(k_arr)
    k_arr = k_arr[sort_idx]
    loss_arr = loss_arr[sort_idx]

    k_range = k_arr[-1] - k_arr[0]
    loss_range = loss_arr[0] - loss_arr[-1]
    if k_range <= 0 or math.isclose(k_range, 0.0):
        return int(k_arr[0])
    if math.isclose(loss_range, 0.0):
        # Flat curve; bias towards higher k if requested
        return int(k_arr[-1] if prefer_high_k else k_arr[0])

    k_norm = (k_arr - k_arr[0]) / k_range
    loss_norm = (loss_arr - loss_arr[-1]) / loss_range
    p0 = np.array([0.0, loss_norm[0]])
    p1 = np.array([1.0, 0.0])
    denom = math.hypot(*(p1 - p0))
    if math.isclose(denom, 0.0):
        return int(k_arr[-1] if prefer_high_k else k_arr[0])

    distances = np.abs((p1[1] - p0[1]) * k_norm - (p1[0] - p0[0]) * loss_norm + p1[0] * p0[1] - p1[1] * p0[0]) / denom
    max_dist = float(distances.max())
    if math.isclose(max_dist, 0.0):
        return int(k_arr[-1] if prefer_high_k else k_arr[0])

    threshold = max_dist * (1.0 - tolerance)
    candidate_idx = np.where(distances >= threshold)[0]
    if prefer_high_k:
        best_local = candidate_idx[np.argmax(k_arr[candidate_idx])]
    else:
        best_local = candidate_idx[np.argmin(k_arr[candidate_idx])]
    return int(k_arr[best_local])


def plot_l2_bar_chart(l2_by_site: Mapping[str, Mapping[int, float]], output_path: str, *, title: str = "L2 (avg last quarter) — top_k", ylabel: str = "L2 loss") -> None:
    if not l2_by_site:
        raise ValueError("No L2 data provided for plotting")
    sites = sorted(l2_by_site.keys())
    all_k = sorted({k for site_dict in l2_by_site.values() for k in site_dict.keys()})
    if not all_k:
        raise ValueError("No k values available for plotting")

    x = np.arange(len(all_k))
    bar_width = 0.8 / max(len(sites), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, site in enumerate(sites):
        vals = [l2_by_site[site].get(k, np.nan) for k in all_k]
        ax.bar(x + idx * bar_width, vals, width=bar_width, label=site)

    ax.set_xticks(x + bar_width * (len(sites) - 1) / 2)
    ax.set_xticklabels([f"k={k}" for k in all_k])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def sae_encode_features(sae, acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x, x_mean, x_std = sae.preprocess_input(acts)
    x_center = x - sae.b_dec
    if isinstance(sae, TopKSAE):
        k = int(sae.cfg["top_k"])
        hidden = F.relu(x_center @ sae.W_enc)
        if k >= hidden.shape[-1]:
            acts_topk = hidden
        else:
            topk = torch.topk(hidden, k, dim=-1)
            acts_topk = torch.zeros_like(hidden).scatter(-1, topk.indices, topk.values)
        return acts_topk, x_mean, x_std
    if isinstance(sae, VanillaSAE):
        hidden = F.relu(x_center @ sae.W_enc + sae.b_enc)
    else:
        hidden = F.relu(x_center @ sae.W_enc)
    return hidden, x_mean, x_std


def sae_decode_features(sae, feature_acts: torch.Tensor, x_mean: torch.Tensor | None, x_std: torch.Tensor | None) -> torch.Tensor:
    recon = feature_acts @ sae.W_dec + sae.b_dec
    return sae.postprocess_output(recon, x_mean, x_std)


def collect_cluster_reconstructions(
    acts: torch.Tensor,
    sae,
    cluster_assignments: Sequence[int],
    *,
    min_activation: float = 0.0,
    encoded_cache: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
    if encoded_cache is not None:
        feature_acts, x_mean, x_std = encoded_cache
    else:
        feature_acts, x_mean, x_std = sae_encode_features(sae, acts)
    feature_acts = feature_acts.detach()
    if x_mean is not None:
        x_mean = x_mean.detach()
    if x_std is not None:
        x_std = x_std.detach()
    cluster_assignments = torch.tensor(cluster_assignments, device=acts.device, dtype=torch.long)
    unique_clusters = torch.unique(cluster_assignments).tolist()

    cluster_recons: Dict[int, np.ndarray] = {}
    cluster_stats: Dict[int, Dict] = {}

    for cluster_value in unique_clusters:
        cluster_id = int(cluster_value)
        if cluster_id < 0:
            continue
        latent_indices = torch.nonzero(cluster_assignments == cluster_value, as_tuple=False).squeeze(-1)
        if latent_indices.numel() == 0:
            continue
        activations_subset = feature_acts[:, latent_indices]
        active_mask = (activations_subset > min_activation).any(dim=1)
        if active_mask.sum() == 0:
            continue
        cluster_latents = torch.zeros_like(feature_acts)
        cluster_latents[:, latent_indices] = feature_acts[:, latent_indices]
        recon = sae_decode_features(sae, cluster_latents, x_mean, x_std)[active_mask]
        cluster_recons[cluster_id] = recon.detach().cpu().numpy()
        stats = {
            "latent_indices": [int(idx) for idx in latent_indices.tolist()],
            "num_samples": int(active_mask.sum().item()),
            "activation_fraction": float(active_mask.float().mean().item()),
            "mean_activation": float(activations_subset[active_mask].mean().item()),
        }
        cluster_stats[cluster_id] = stats
    return cluster_recons, cluster_stats


def fit_pca_for_clusters(
    cluster_recons: Mapping[int, np.ndarray],
    *,
    n_components: int = 3,
    min_samples: int = 32,
) -> Dict[int, ClusterPCAResult]:
    results: Dict[int, ClusterPCAResult] = {}
    for cluster_id, data in cluster_recons.items():
        if data.shape[0] < max(n_components, min_samples):
            continue
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(data)
        decoder_coords: Dict[int, np.ndarray] = {}
        results[cluster_id] = ClusterPCAResult(coords=coords, pca=pca, decoder_coords=decoder_coords, scale_factor=1.0)
    return results


def project_decoder_directions_to_pca(
    sae,
    pca_results: Dict[int, ClusterPCAResult],
    cluster_stats: Mapping[int, Mapping],
    *,
    normalize: bool = True,
) -> None:
    decoder = sae.W_dec.detach().cpu().numpy()
    for cluster_id, result in pca_results.items():
        stats = cluster_stats.get(cluster_id)
        if stats is None:
            continue
        latent_indices = stats.get("latent_indices", [])
        if not latent_indices:
            continue
        vectors = decoder[latent_indices]
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            vectors = vectors / norms
        centered = vectors - result.pca.mean_
        projected = centered @ result.pca.components_.T
        # scale decoder vectors to match data spread
        data_norm = np.linalg.norm(result.coords, axis=1)
        data_scale = float(np.percentile(data_norm, 90)) if data_norm.size else 1.0
        vector_norm = np.linalg.norm(projected, axis=1)
        max_vec = float(vector_norm.max()) if vector_norm.size else 1.0
        scale = data_scale / max_vec if max_vec > 0 else 1.0
        result.decoder_coords = {latent_indices[i]: projected[i] * scale for i in range(len(latent_indices))}
        result.scale_factor = scale


def plot_cluster_pca(
    site_name: str,
    k_value: int,
    cluster_id: int,
    result: ClusterPCAResult,
    output_path: str,
    *,
    max_points: int | None = None,
    random_state: int | None = None,
) -> None:
    coords = result.coords
    if coords.shape[1] < 3:
        raise ValueError("Cluster PCA result does not have 3 components")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if max_points is not None and coords.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        subset_idx = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords_to_plot = coords[subset_idx]
    else:
        coords_to_plot = coords

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords_to_plot[:, 0], coords_to_plot[:, 1], coords_to_plot[:, 2], s=8, alpha=0.25, color="tab:gray")

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue", "tab:orange", "tab:green", "tab:red"])
    for idx, (latent_idx, vec) in enumerate(result.decoder_coords.items()):
        color = colors[idx % len(colors)]
        ax.quiver(0.0, 0.0, 0.0, vec[0], vec[1], vec[2], color=color, linewidth=1.8)
        ax.text(vec[0], vec[1], vec[2], str(latent_idx), color=color, fontsize=8)

    ax.set_title(f"{site_name} • k={k_value} • cluster {cluster_id}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_activity_histogram(
    values: Sequence[float],
    output_path: str,
    *,
    bins: int = 20,
    title: str | None = None,
) -> None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr, bins=bins, range=(0.0, 1.0), color="tab:blue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Activation rate")
    ax.set_ylabel("Latent count")
    ax.set_xlim(0.0, 1.0)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_activity_histograms_by_site(
    data: Mapping[str, Sequence[float]],
    output_path: str,
    *,
    bins: int = 20,
    suptitle: str | None = None,
) -> None:
    if not data:
        return
    sites = sorted(data.keys())
    n_sites = len(sites)
    fig, axes = plt.subplots(1, n_sites, figsize=(4 * n_sites, 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for idx, site in enumerate(sites):
        ax = axes[idx]
        values = np.asarray(data[site], dtype=float)
        if values.size == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{site}\n(no active latents)", transform=ax.transAxes, ha="center", va="center")
            continue
        ax.hist(values, bins=bins, range=(0.0, 1.0), color="tab:blue", edgecolor="black", alpha=0.8)
        ax.set_title(f"{site} (n={values.size})")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for ax in axes:
        if ax.has_data():
            ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Activation rate")
    axes[0].set_ylabel("Latent count")
    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_activity_histograms_site_clusters(
    site_cluster_rates: Mapping[str, Mapping[int, Sequence[float]]],
    output_path: str,
    *,
    bins: int = 20,
    suptitle: str | None = None,
) -> None:
    if not site_cluster_rates:
        return
    sites = sorted(site_cluster_rates.keys())
    cluster_counts = []
    for site in sites:
        cluster_map = site_cluster_rates.get(site, {})
        count = sum(1 for vals in cluster_map.values() if np.asarray(vals, dtype=float).size > 0)
        cluster_counts.append(count)
    max_clusters = max(cluster_counts) if cluster_counts else 0
    if max_clusters == 0:
        return
    fig, axes = plt.subplots(max_clusters, len(sites), figsize=(4 * len(sites), 3 * max_clusters), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for col, site in enumerate(sites):
        cluster_map = site_cluster_rates.get(site, {})
        cluster_ids = [cid for cid in sorted(cluster_map.keys()) if np.asarray(cluster_map.get(cid, []), dtype=float).size > 0]
        for row in range(max_clusters):
            ax = axes[row, col]
            if row >= len(cluster_ids):
                ax.axis("off")
                continue
            cid = cluster_ids[row]
            values = np.asarray(cluster_map.get(cid, []), dtype=float)
            if values.size == 0:
                ax.axis("off")
                continue
            ax.hist(values, bins=bins, range=(0.0, 1.0), color="tab:orange", edgecolor="black", alpha=0.8)
            ax.set_title(f"{site}\ncluster {cid} (n={values.size})", fontsize=9)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for ax in axes.flatten():
        if ax.has_data():
            ax.set_xlim(0.0, 1.0)
    for row in range(max_clusters):
        axes[row, 0].set_ylabel("Latent count")
    for col in range(len(sites)):
        axes[-1, col].set_xlabel("Activation rate")
    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_cluster_metadata(
    output_path: str,
    cluster_stats: Mapping[int, Mapping],
    selected_k: int,
    average_l2: Mapping[int, float],
    cluster_labels: Sequence[int] | None = None,
    extra_fields: Mapping[str, object] | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "selected_k": int(selected_k),
        "average_l2": {int(k): float(v) for k, v in average_l2.items()},
        "clusters": {
            int(cid): {
                key: (value if not isinstance(value, list) else [int(v) if isinstance(v, (int, np.integer)) else float(v) for v in value])
                for key, value in stats.items()
            }
            for cid, stats in cluster_stats.items()
        },
    }
    if cluster_labels is not None:
        payload["latent_cluster_assignments"] = {
            int(idx): int(lbl)
            for idx, lbl in enumerate(cluster_labels)
        }
    if extra_fields is not None:
        for key, value in extra_fields.items():
            payload[key] = value
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
