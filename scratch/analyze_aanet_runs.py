#!/usr/bin/env python3
"""
Scratch analysis script for inspecting AAnet training runs.

Features:
  * Aggregate loss metrics across layers / clusters / k-values.
  * Plot train/validation reconstruction loss curves to inspect elbow behaviour.
  * Generate per-cluster PCA visualisations with archetypes overlaid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODULE_ROOT = ROOT_DIR / "AAnet"
if MODULE_ROOT.exists():
    import sys

    if str(MODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(MODULE_ROOT))

try:
    from AAnet.AAnet_torch.models import AAnet_vanilla
except ImportError:
    from AAnet_torch.models import AAnet_vanilla  # type: ignore

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from aanet_pipeline import (
    ClusterDescriptor,
    ExtremaConfig,
    build_cluster_datasets,
    load_cluster_summary,
    parse_cluster_descriptors,
)
from BatchTopK.sae import TopKSAE
from transformer_lens import HookedTransformer, HookedTransformerConfig
from multipartite_utils import MultipartiteSampler, build_components_from_config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    layer: int
    cluster_label: str
    cluster_id: int
    component_names: Sequence[str]
    is_noise: bool
    k: int
    status: str
    metrics: Dict[str, float]
    epoch_metrics_path: Path
    model_path: Optional[Path]


@dataclass
class LossCurve:
    epochs: List[int]
    train: List[float]
    val: List[float]
    lr: List[float]


# ---------------------------------------------------------------------------
# Helpers for loading run metadata
# ---------------------------------------------------------------------------


def load_run_records(root: Path, layers: Optional[Iterable[int]] = None) -> Dict[Tuple[int, str], List[RunRecord]]:
    layer_filter = set(layers) if layers is not None else None
    records: Dict[Tuple[int, str], List[RunRecord]] = defaultdict(list)

    for layer_dir in sorted(root.glob("layer_*")):
        try:
            layer_idx = int(layer_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if layer_filter is not None and layer_idx not in layer_filter:
            continue

        for cluster_dir in sorted(layer_dir.iterdir()):
            if not cluster_dir.is_dir():
                continue
            cluster_label = cluster_dir.name
            for k_dir in sorted(cluster_dir.glob("k_*")):
                metrics_path = k_dir / "metrics.json"
                epoch_path = k_dir / "epoch_metrics.csv"
                if not metrics_path.exists():
                    continue
                with metrics_path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                metrics = record.get("metrics", {})
                run = RunRecord(
                    layer=record.get("layer", layer_idx),
                    cluster_label=record.get("cluster_label", cluster_label),
                    cluster_id=record.get("cluster_id", -1),
                    component_names=record.get("component_names", []),
                    is_noise=record.get("is_noise", False),
                    k=record.get("k"),
                    status=record.get("status", "unknown"),
                    metrics=metrics,
                    epoch_metrics_path=epoch_path,
                    model_path=k_dir / "model.pt" if (k_dir / "model.pt").exists() else None,
                )
                records[(layer_idx, cluster_label)].append(run)

    # Sort runs per cluster by k
    for key in records.keys():
        records[key].sort(key=lambda r: r.k)
    return records


def load_epoch_metrics(csv_path: Path) -> LossCurve:
    epochs: List[int] = []
    train: List[float] = []
    val: List[float] = []
    lr: List[float] = []
    if not csv_path.exists():
        return LossCurve(epochs, train, val, lr)
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(float(row.get("epoch", 0))))
            train.append(float(row.get("reconstruction_loss", 0.0)))
            val.append(float(row.get("val_loss", 0.0)))
            lr.append(float(row.get("lr", 0.0)))
    return LossCurve(epochs, train, val, lr)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def plot_loss_curves(records: Dict[Tuple[int, str], List[RunRecord]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for (layer, cluster_label), runs in records.items():
        if not runs:
            continue
        ks = [r.k for r in runs]
        train_losses = [r.metrics.get("reconstruction_loss_final") for r in runs]
        val_losses = [r.metrics.get("val_reconstruction_loss_final") for r in runs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, train_losses, marker="o", label="train reconstruction")
        ax.plot(ks, val_losses, marker="o", label="val reconstruction")
        ax.set_xlabel("k (archetypes)")
        ax.set_ylabel("loss")
        ax.set_title(f"Layer {layer} – {cluster_label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"layer{layer}_{cluster_label}_loss_curve.png", dpi=200)
        plt.close(fig)


def select_best_run(runs: List[RunRecord]) -> Optional[RunRecord]:
    if not runs:
        return None
    # Prefer lowest validation reconstruction loss; fall back to train loss
    best = None
    best_val = float("inf")
    for run in runs:
        val_score = run.metrics.get("val_reconstruction_loss_final")
        if val_score is None:
            val_score = run.metrics.get("reconstruction_loss_final", float("inf"))
        if val_score < best_val:
            best = run
            best_val = val_score
    return best


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def _load_transformer(args, device: torch.device) -> HookedTransformer:
    checkpoint = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    cfg_dict = checkpoint.get("config")
    cfg = None
    if isinstance(cfg_dict, dict):
        try:
            cfg = HookedTransformerConfig.from_dict(cfg_dict)
        except Exception:
            cfg = None
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    inferred_vocab = None
    if state_dict is not None:
        embed_weight = state_dict.get("embed.W_E")
        if embed_weight is not None and hasattr(embed_weight, "shape"):
            inferred_vocab = int(embed_weight.shape[0])
    if cfg is None:
        cfg = HookedTransformerConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_ctx=args.n_ctx,
            d_head=args.d_head,
            act_fn=args.act_fn,
            d_vocab=inferred_vocab,
        )
    model = HookedTransformer(cfg).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _instantiate_sae(path: Path, device: torch.device) -> TopKSAE:
    payload = torch.load(path, map_location=device)
    cfg = dict(payload["cfg"])
    cfg["device"] = str(device)
    sae = TopKSAE(cfg).to(device)
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    return sae


def build_layer_datasets(
    args,
    layer: int,
    model,
    sampler,
    sae: TopKSAE,
    descriptors: Sequence[ClusterDescriptor],
    *,
    device: torch.device,
):
    datasets, _ = build_cluster_datasets(
        model=model,
        sampler=sampler,
        layer_hook=f"blocks.{layer}.hook_resid_post" if layer >= 0 else "hook_embed",
        sae=sae,
        cluster_descriptors=descriptors,
        batch_size=args.analysis_batch_size,
        seq_len=args.analysis_seq_len,
        num_batches=args.analysis_num_batches,
        activation_threshold=args.analysis_activation_threshold,
        device=device,
        max_samples_per_cluster=args.analysis_max_samples,
        min_cluster_samples=0,
        seed=args.analysis_seed + layer,
        token_positions=args.analysis_token_indices,
    )
    return datasets


def encode_latents(
    model,
    data_tensor: torch.Tensor,
    *,
    device: torch.device,
    max_points: int = 60000,
) -> Tuple[np.ndarray, np.ndarray]:
    total = data_tensor.shape[0]
    if total > max_points:
        idx = torch.randperm(total)[:max_points]
        data_sample = data_tensor[idx]
    else:
        data_sample = data_tensor
    batch_size = 2048
    latents_list = []
    bary_list = []
    model.eval()
    with torch.no_grad():
        for start in range(0, data_sample.shape[0], batch_size):
            batch = data_sample[start : start + batch_size].to(device)
            latent = model.encode(batch)
            bary = model.euclidean_to_barycentric(latent)
            latents_list.append(latent.cpu())
            bary_list.append(bary.cpu())
    latents = torch.cat(latents_list, dim=0).numpy()
    bary = torch.cat(bary_list, dim=0).numpy()
    return latents, bary


def compute_pca(data: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    # Use SVD for stability
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projected = centered @ components.T
    var = (s ** 2)
    total_var = var.sum()
    if total_var <= 0:
        var_ratio = np.zeros_like(s)
    else:
        var_ratio = var / total_var
    return mean.squeeze(0), components, projected, var_ratio


def project_points(points: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (points - mean) @ components.T


def plot_pca_with_archetypes(
    data_proj: np.ndarray,
    archetype_sets: Dict[int, np.ndarray],
    output_path: Path,
    title: str,
    var_ratio: Optional[np.ndarray] = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_points = min(20000, data_proj.shape[0])
    if data_proj.shape[0] > num_points:
        sample_idx = np.random.choice(data_proj.shape[0], size=num_points, replace=False)
        data_sample = data_proj[sample_idx]
    else:
        data_sample = data_proj

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data_sample[:, 0],
        data_sample[:, 1],
        data_sample[:, 2],
        s=2,
        alpha=0.15,
        label="data",
    )
    colors = plt.colormaps["tab10"]
    for idx, (k, arche_proj) in enumerate(sorted(archetype_sets.items())):
        color = colors(idx % 10)
        ax.scatter(
            arche_proj[:, 0],
            arche_proj[:, 1],
            arche_proj[:, 2],
            s=80,
            c=[color],
            marker="^",
            label=f"archetypes k={k}",
        )
        for point_idx, point in enumerate(arche_proj):
            ax.text(point[0], point[1], point[2], f"{k}:{point_idx+1}", color=color)

    subtitle = ""
    if var_ratio is not None and var_ratio.size >= 3:
        subtitle = f" (var: {var_ratio[0]:.2%}, {var_ratio[1]:.2%}, {var_ratio[2]:.2%})"
    ax.set_title(title + subtitle)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _pad_to_3d(array: np.ndarray) -> np.ndarray:
    if array.shape[1] >= 3:
        return array[:, :3]
    padded = np.zeros((array.shape[0], 3), dtype=array.dtype)
    padded[:, : array.shape[1]] = array
    return padded


def plot_latent_euclidean_plotly(
    data_latent: np.ndarray,
    simplex_vertices: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if go is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = min(50000, data_latent.shape[0])
    if data_latent.shape[0] > num_samples:
        idx = np.random.choice(data_latent.shape[0], size=num_samples, replace=False)
        data_sample = data_latent[idx]
    else:
        data_sample = data_latent

    data_plot = _pad_to_3d(data_sample)
    simplex_plot = _pad_to_3d(simplex_vertices)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data_plot[:, 0],
            y=data_plot[:, 1],
            z=data_plot[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.2, color="gray"),
            name="data",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=simplex_plot[:, 0],
            y=simplex_plot[:, 1],
            z=simplex_plot[:, 2],
            mode="markers+text",
            marker=dict(size=6, color="red", symbol="diamond"),
            text=[f"v{i+1}" for i in range(simplex_plot.shape[0])],
            name="simplex",
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Dim1", yaxis_title="Dim2", zaxis_title="Dim3"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(output_path)


def plot_pca_plotly(
    data_proj: np.ndarray,
    archetype_sets: Dict[int, np.ndarray],
    output_path: Path,
    title: str,
    var_ratio: Optional[np.ndarray] = None,
) -> None:
    if go is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_points = min(50000, data_proj.shape[0])
    if data_proj.shape[0] > num_points:
        sample_idx = np.random.choice(data_proj.shape[0], size=num_points, replace=False)
        data_sample = data_proj[sample_idx]
    else:
        data_sample = data_proj

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data_sample[:, 0],
            y=data_sample[:, 1],
            z=data_sample[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.15, color="gray"),
            name="data",
        )
    )
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for idx, (k, arche_proj) in enumerate(sorted(archetype_sets.items())):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=arche_proj[:, 0],
                y=arche_proj[:, 1],
                z=arche_proj[:, 2],
                mode="markers+text",
                marker=dict(size=6, color=color, symbol="diamond"),
                text=[f"{k}:{i+1}" for i in range(arche_proj.shape[0])],
                name=f"k={k}",
            )
        )
    subtitle = ""
    if var_ratio is not None and var_ratio.size >= 3:
        subtitle = f" (var: {var_ratio[0]:.2%}, {var_ratio[1]:.2%}, {var_ratio[2]:.2%})"
    fig.update_layout(
        title=title + subtitle,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(output_path)


def plot_latent_euclidean(
    data_latent: np.ndarray,
    simplex_vertices: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = min(20000, data_latent.shape[0])
    if data_latent.shape[0] > num_samples:
        idx = np.random.choice(data_latent.shape[0], size=num_samples, replace=False)
        data_sample = data_latent[idx]
    else:
        data_sample = data_latent

    data_plot = _pad_to_3d(data_sample)
    simplex_plot = _pad_to_3d(simplex_vertices)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data_plot[:, 0], data_plot[:, 1], data_plot[:, 2], s=3, alpha=0.15, label="data")
    ax.scatter(simplex_plot[:, 0], simplex_plot[:, 1], simplex_plot[:, 2], c="black", marker="x", label="simplex")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_latent_barycentric_grid(
    bary_data: np.ndarray,
    bary_arche: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    k = bary_data.shape[1]
    if k != 4:
        return
    num_samples = min(20000, bary_data.shape[0])
    if bary_data.shape[0] > num_samples:
        idx = np.random.choice(bary_data.shape[0], size=num_samples, replace=False)
        data_sample = bary_data[idx]
    else:
        data_sample = bary_data

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    for drop_idx in range(4):
        dims = [d for d in range(4) if d != drop_idx]
        ax = axes[drop_idx]
        ax.scatter(
            data_sample[:, dims[0]],
            data_sample[:, dims[1]],
            data_sample[:, dims[2]],
            s=3,
            alpha=0.15,
            label="data",
        )
        ax.scatter(
            bary_arche[:, dims[0]],
            bary_arche[:, dims[1]],
            bary_arche[:, dims[2]],
            c="red",
            marker="^",
            label="archetypes",
        )
        ax.set_xlabel(f"Dim {dims[0]+1}")
        ax.set_ylabel(f"Dim {dims[1]+1}")
        ax.set_zlabel(f"Dim {dims[2]+1}")
        ax.set_title(f"Drop dim {drop_idx+1}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_barycentric_3d(
    bary_data: np.ndarray,
    bary_arche: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = min(20000, bary_data.shape[0])
    if bary_data.shape[0] > num_samples:
        idx = np.random.choice(bary_data.shape[0], size=num_samples, replace=False)
        data_sample = bary_data[idx]
    else:
        data_sample = bary_data
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data_sample[:, 0], data_sample[:, 1], data_sample[:, 2], s=3, alpha=0.15, label="data")
    ax.scatter(bary_arche[:, 0], bary_arche[:, 1], bary_arche[:, 2], c="red", marker="^", label="archetypes")
    ax.set_xlabel("Bary 1")
    ax.set_ylabel("Bary 2")
    ax.set_zlabel("Bary 3")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_barycentric_plotly(
    bary_data: np.ndarray,
    bary_arche: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if go is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = min(50000, bary_data.shape[0])
    if bary_data.shape[0] > num_samples:
        idx = np.random.choice(bary_data.shape[0], size=num_samples, replace=False)
        data_sample = bary_data[idx]
    else:
        data_sample = bary_data
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data_sample[:, 0],
            y=data_sample[:, 1],
            z=data_sample[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.2, color="gray"),
            name="data",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=bary_arche[:, 0],
            y=bary_arche[:, 1],
            z=bary_arche[:, 2],
            mode="markers+text",
            marker=dict(size=6, color="red", symbol="diamond"),
            text=[f"A{i+1}" for i in range(bary_arche.shape[0])],
            name="archetypes",
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Bary1", yaxis_title="Bary2", zaxis_title="Bary3"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(output_path)


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze AAnet training outputs.")
    parser.add_argument("--aanet-root", type=Path, required=True, help="Root directory for AAnet outputs.")
    parser.add_argument("--process-config", type=Path, required=True)
    parser.add_argument("--process-config-name", type=str, required=True)
    parser.add_argument("--model-ckpt", type=Path, required=True)
    parser.add_argument("--sae-root", type=Path, required=True)
    parser.add_argument("--cluster-summary-dir", type=Path, required=True)
    parser.add_argument("--cluster-summary-pattern", type=str, default="top_r2_run_layer_{layer}_cluster_summary.json")
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/aanet_analysis"))
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Subset of layers to analyze.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model loading.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--act-fn", type=str, default="relu")

    # Analysis sampling parameters
    parser.add_argument("--analysis-topk", type=int, default=12, help="Top-k SAE to load for analysis.")
    parser.add_argument("--analysis-batch-size", type=int, default=128)
    parser.add_argument("--analysis-seq-len", type=int, default=16)
    parser.add_argument("--analysis-num-batches", type=int, default=100)
    parser.add_argument("--analysis-activation-threshold", type=float, default=0.0)
    parser.add_argument("--analysis-max-samples", type=int, default=60000)
    parser.add_argument("--analysis-seed", type=int, default=777)
    parser.add_argument("--analysis-token-indices", type=int, nargs="+", default=[4, 9, 14])
    parser.add_argument(
        "--pca-k-values",
        type=int,
        nargs="+",
        default=None,
        help="Specific archetype counts to visualise (overrides --pca-top-k).",
    )
    parser.add_argument("--pca-top-k", type=int, default=1, help="Number of top runs (by val loss) to visualise per cluster when --pca-k-values is not provided.")

    args = parser.parse_args()

    args.analysis_token_indices = None if args.analysis_token_indices is None else list(args.analysis_token_indices)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_run_records(args.aanet_root, layers=args.layers)
    plot_loss_curves(records, output_dir / "loss_curves")

    # Prepare global model/sampler for dataset reconstruction
    device = torch.device(args.device)
    model = _load_transformer(args, device=device)
    with args.process_config.open("r", encoding="utf-8") as handle:
        process_cfg_all = json.load(handle)
    process_cfg = process_cfg_all[args.process_config_name]
    components = build_components_from_config(process_cfg)
    sampler = MultipartiteSampler(components)

    for (layer, cluster_label), runs in records.items():
        descriptors_path = args.cluster_summary_dir / args.cluster_summary_pattern.format(layer=layer)
        if not descriptors_path.exists():
            continue
        summary = load_cluster_summary(descriptors_path)
        descriptors = parse_cluster_descriptors(summary, include_noise=True)
        sae_path = args.sae_root / f"layer_{layer}_top_k_k{args.analysis_topk}.pt"
        if not sae_path.exists():
            # fallback to k12
            sae_path = args.sae_root / f"layer_{layer}_top_k_k12.pt"
        sae = _instantiate_sae(sae_path, device=device)
        datasets = build_layer_datasets(
            args,
            layer,
            model,
            sampler,
            sae,
            descriptors,
            device=device,
        )

        selected_runs: Dict[int, RunRecord] = {}
        if args.pca_k_values:
            target = set(args.pca_k_values)
            for run in runs:
                if run.k in target and run.k not in selected_runs:
                    selected_runs[run.k] = run
        else:
            best_runs = sorted(
                runs,
                key=lambda r: r.metrics.get("val_reconstruction_loss_final", float("inf")),
            )[: args.pca_top_k]
            for run in best_runs:
                selected_runs[run.k] = run

        if not selected_runs:
            continue

        dataset = None
        if selected_runs:
            sample_run = next(iter(selected_runs.values()))
            dataset = datasets.get(sample_run.cluster_id)
        if dataset is None or dataset.data.shape[0] == 0:
            continue

        data_tensor = dataset.data
        data_np = data_tensor.detach().cpu().numpy()
        mean, components, projected, var_ratio = compute_pca(data_np, n_components=3)

        archetype_sets: Dict[int, np.ndarray] = {}
        latent_info: Dict[int, Dict[str, np.ndarray]] = {}
        for k, run in selected_runs.items():
            dataset = datasets.get(run.cluster_id)
            if dataset is None or dataset.data.shape[0] == 0:
                continue
            if run.model_path is None or not run.model_path.exists():
                continue
            payload = torch.load(run.model_path, map_location=device)
            aanet_cfg = payload.get("aanet_config", {})
            model_vis = AAnet_vanilla(
                input_shape=data_tensor.shape[1],
                n_archetypes=run.k,
                layer_widths=aanet_cfg.get("layer_widths", [256, 128]),
                simplex_scale=aanet_cfg.get("simplex_scale", 1.0),
                noise=0.0,
                device=device,
            )
            model_vis.load_state_dict(payload["state_dict"])
            simplex_scale = aanet_cfg.get("simplex_scale", 1.0)
            archetypes = model_vis.get_archetypes_data().detach().cpu().numpy()
            arche_proj = project_points(archetypes, mean, components)
            archetype_sets[k] = arche_proj
            simplex_vertices = model_vis.get_n_simplex(run.k, simplex_scale).detach().cpu().numpy()
            data_latent, data_bary = encode_latents(
                model_vis,
                data_tensor,
                device=device,
                max_points=args.analysis_max_samples,
            )
            latent_info[k] = {
                "latent": data_latent,
                "bary": data_bary,
                "simplex": simplex_vertices,
            }

        if not archetype_sets:
            continue

        plot_dir = output_dir / "pca"
        k_suffix = "-".join(str(k) for k in sorted(archetype_sets.keys()))
        title = f"Layer {layer} – {sample_run.cluster_label}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        var_payload = {f"pc{i+1}": float(val) for i, val in enumerate(var_ratio[:20])}
        var_path = plot_dir / f"layer{layer}_{sample_run.cluster_label}_variance.json"
        with var_path.open("w", encoding="utf-8") as handle:
            json.dump(var_payload, handle, indent=2)

        plot_path = plot_dir / f"layer{layer}_{sample_run.cluster_label}_k{k_suffix}.png"
        plot_pca_with_archetypes(projected, archetype_sets, plot_path, f"{title} (k={k_suffix})", var_ratio=var_ratio)
        if go is not None:
            plot_pca_plotly(
                projected,
                archetype_sets,
                plot_dir / f"layer{layer}_{sample_run.cluster_label}_k{k_suffix}.html",
                f"{title} (k={k_suffix})",
                var_ratio=var_ratio,
            )
        latent_dir = output_dir / "latent"
        for k, info in latent_info.items():
            plot_latent_euclidean(
                info["latent"],
                info["simplex"],
                latent_dir / f"layer{layer}_{sample_run.cluster_label}_k{k}_euclidean.png",
                f"{title} latent k={k}",
            )
            if go is not None:
                plot_latent_euclidean_plotly(
                    info["latent"],
                    info["simplex"],
                    latent_dir / f"layer{layer}_{sample_run.cluster_label}_k{k}_euclidean.html",
                    f"{title} latent k={k}",
                )
            if k == 3:
                bary = info["bary"][:, :3]
                plot_barycentric_3d(
                    bary,
                    np.eye(3),
                    latent_dir / f"layer{layer}_{sample_run.cluster_label}_k{k}_barycentric.png",
                    f"{title} barycentric k={k}",
                )
                if go is not None:
                    plot_barycentric_plotly(
                        bary,
                        np.eye(3),
                        latent_dir / f"layer{layer}_{sample_run.cluster_label}_k{k}_barycentric.html",
                        f"{title} barycentric k={k}",
                    )
            if k == 4:
                plot_latent_barycentric_grid(
                    info["bary"],
                    np.eye(4),
                    latent_dir / f"layer{layer}_{sample_run.cluster_label}_k{k}_barycentric.png",
                    f"{title} barycentric k={k}",
                )
                if go is not None:
                    for drop_idx in range(4):
                        dims = [d for d in range(4) if d != drop_idx]
                        plot_barycentric_plotly(
                            info["bary"][:, dims],
                            np.eye(4)[:, dims],
                            latent_dir / f"layer{layer}_{sample_run.cluster_label}_k{k}_barycentric_drop{drop_idx+1}.html",
                            f"{title} barycentric k={k} drop {drop_idx+1}",
                        )


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
