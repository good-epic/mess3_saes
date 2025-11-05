#!/usr/bin/env python3

from __future__ import annotations

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

import graphtools as gt

from BatchTopK.sae import TopKSAE
from transformer_lens import HookedTransformer, HookedTransformerConfig

from aanet_pipeline import (
    ClusterDatasetResult,
    ClusterDescriptor,
    ExtremaConfig,
    TrainingConfig,
    build_cluster_datasets,
    compute_diffusion_extrema,
    load_cluster_summary,
    parse_cluster_descriptors,
    train_aanet_model,
)
from multipartite_utils import build_components_from_config, MultipartiteSampler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AAnet models on SAE cluster reconstructions.")

    io_group = parser.add_argument_group("paths")
    io_group.add_argument("--process-config", type=Path, required=True, help="Path to process_configs.json.")
    io_group.add_argument("--process-config-name", type=str, required=True, help="Configuration key to load.")
    io_group.add_argument("--model-ckpt", type=Path, required=True, help="Transformer checkpoint path.")
    io_group.add_argument("--sae-root", type=Path, required=True, help="Directory containing saved SAEs.")
    io_group.add_argument("--cluster-summary-dir", type=Path, required=True, help="Directory holding cluster summary JSON files.")
    io_group.add_argument("--cluster-summary-pattern", type=str, default="top_r2_run_layer_{layer}_cluster_summary.json", help="Filename pattern for cluster summaries.")
    io_group.add_argument("--output-dir", type=Path, required=True, help="Directory to store AAnet outputs.")

    model_group = parser.add_argument_group("transformer")
    model_group.add_argument("--d-model", type=int, default=128)
    model_group.add_argument("--n-heads", type=int, default=4)
    model_group.add_argument("--n-layers", type=int, default=3)
    model_group.add_argument("--n-ctx", type=int, default=16)
    model_group.add_argument("--d-vocab", type=int, default=None)
    model_group.add_argument("--d-head", type=int, default=32)
    model_group.add_argument("--act-fn", type=str, default="relu")
    model_group.add_argument("--device", type=str, default="cuda")

    sampling_group = parser.add_argument_group("sampling")
    sampling_group.add_argument("--layers", type=int, nargs="+", required=True, help="Layer indices to process.")
    sampling_group.add_argument("--topk", type=int, default=12, help="Top-k value used for SAE checkpoints.")
    sampling_group.add_argument("--batch-size", type=int, default=256, help="Batch size for sampler/model runs.")
    sampling_group.add_argument("--seq-len", type=int, default=16, help="Sequence length passed to the sampler.")
    sampling_group.add_argument("--num-batches", type=int, default=64, help="Number of batches to generate per layer.")
    sampling_group.add_argument("--activation-threshold", type=float, default=0.0, help="Activation threshold for keeping samples.")
    sampling_group.add_argument("--max-samples-per-cluster", type=int, default=None, help="Optional cap on samples per cluster.")
    sampling_group.add_argument("--min-cluster-samples", type=int, default=0, help="Minimum samples required to consider training.")
    sampling_group.add_argument("--sampling-seed", type=int, default=123, help="Seed for data generation.")
    sampling_group.add_argument("--token-indices", type=int, nargs="+", default=None, help="Optional 0-based token positions to retain from each sequence.")

    aanet_group = parser.add_argument_group("aanet")
    aanet_group.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8], help="Archetype counts to evaluate.")
    aanet_group.add_argument("--aanet-epochs", type=int, default=100, help="Training epochs per model.")
    aanet_group.add_argument("--aanet-batch-size", type=int, default=256, help="Batch size for AAnet training.")
    aanet_group.add_argument("--aanet-lr", type=float, default=1e-3, help="Learning rate.")
    aanet_group.add_argument("--aanet-weight-decay", type=float, default=0.0, help="Weight decay.")
    aanet_group.add_argument("--aanet-layer-widths", type=int, nargs="+", default=[256, 128], help="Hidden layer widths.")
    aanet_group.add_argument("--aanet-simplex-scale", type=float, default=1.0, help="Simplex scale.")
    aanet_group.add_argument("--aanet-noise", type=float, default=0.05, help="Latent noise value or scale.")
    aanet_group.add_argument("--aanet-noise-relative", action="store_true", help="Interpret --aanet-noise as a multiple of the dataset std.")
    aanet_group.add_argument("--aanet-gamma-reconstruction", type=float, default=1.0)
    aanet_group.add_argument("--aanet-gamma-archetypal", type=float, default=1.0)
    aanet_group.add_argument("--aanet-gamma-extrema", type=float, default=1.0)
    aanet_group.add_argument("--aanet-min-samples", type=int, default=32, help="Minimum dataset size before training.")
    aanet_group.add_argument("--aanet-num-workers", type=int, default=0, help="DataLoader workers for AAnet training.")
    aanet_group.add_argument("--aanet-seed", type=int, default=43, help="Base seed for AAnet training.")
    aanet_group.add_argument("--aanet-val-fraction", type=float, default=0.1, help="Fraction of samples reserved for validation per cluster.")
    aanet_group.add_argument("--aanet-val-min-size", type=int, default=256, help="Minimum number of samples required for a validation split.")
    aanet_group.add_argument("--aanet-early-stop-patience", type=int, default=10, help="Early stopping patience based on validation loss.")
    aanet_group.add_argument("--aanet-early-stop-delta", type=float, default=1e-4, help="Minimum improvement in validation loss to reset patience.")
    aanet_group.add_argument("--aanet-lr-patience", type=int, default=5, help="ReduceLROnPlateau patience in epochs.")
    aanet_group.add_argument("--aanet-lr-factor", type=float, default=0.5, help="Factor to reduce learning rate when plateau is detected.")
    aanet_group.add_argument("--aanet-grad-clip", type=float, default=1.0, help="Gradient clipping norm (set <=0 to disable).")
    aanet_group.add_argument("--aanet-restarts-no-extrema", type=int, default=3, help="Number of random restarts when no warm-start extrema are available.")

    extrema_group = parser.add_argument_group("extrema")
    extrema_group.add_argument("--extrema-enabled", dest="extrema_enabled", action="store_true", default=True, help="Enable Laplacian extrema warm start.")
    extrema_group.add_argument("--no-extrema", dest="extrema_enabled", action="store_false", help="Disable Laplacian extrema warm start.")
    extrema_group.add_argument("--extrema-knn", type=int, default=10, help="kNN value for Laplacian extrema.")
    extrema_group.add_argument("--extrema-disable-subsample", action="store_true", help="Disable internal subsampling.")
    extrema_group.add_argument("--extrema-max-points", type=int, default=10000, help="Maximum samples used for extrema computation.")
    extrema_group.add_argument("--extrema-seed", type=int, default=0, help="Seed for extrema subsampling.")

    parser.add_argument("--save-models", action="store_true", help="Persist trained AAnet weights.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    return parser.parse_args()


def _slugify(label: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    safe = safe.strip("_")
    return safe or "cluster"


def _load_transformer(args: argparse.Namespace, device: torch.device) -> HookedTransformer:
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
        d_vocab = args.d_vocab if args.d_vocab is not None else inferred_vocab
        if d_vocab is None:
            raise ValueError(
                "Unable to infer vocabulary size; specify --d-vocab explicitly."
            )
        cfg = HookedTransformerConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_ctx=args.n_ctx,
            d_head=args.d_head,
            act_fn=args.act_fn,
            d_vocab=d_vocab,
        )
    model = HookedTransformer(cfg).to(device)
    if state_dict is None:
        raise KeyError("Checkpoint missing Transformer weights.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_sampler(args: argparse.Namespace) -> MultipartiteSampler:
    with args.process_config.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, dict):
        if args.process_config_name not in loaded:
            raise KeyError(f"process_config_name '{args.process_config_name}' not found.")
        config_raw = loaded[args.process_config_name]
    else:
        raise TypeError("process_config must contain a mapping of named configurations.")
    if isinstance(config_raw, dict) and "type" in config_raw:
        config_list = [config_raw]
    elif isinstance(config_raw, list):
        config_list = config_raw
    else:
        raise ValueError("Unsupported process configuration format.")
    components = build_components_from_config(config_list)
    return MultipartiteSampler(components)


def _instantiate_sae(path: Path, device: torch.device) -> TopKSAE:
    payload = torch.load(path, map_location=device)
    cfg = payload["cfg"]
    sae = TopKSAE(cfg)
    sae.load_state_dict(payload["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def _ensure_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_cluster_plot(cluster_dir: Path, cluster_label: str, records: List[dict]) -> None:
    if not records:
        return
    records = sorted(records, key=lambda item: item["k"])
    ks = [item["k"] for item in records]
    total_loss = [item["metrics"].get("loss_final") for item in records]
    recon_loss = [item["metrics"].get("reconstruction_loss_final") for item in records]
    arche_loss = [item["metrics"].get("archetypal_loss_final") for item in records]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, total_loss, marker="o", label="total loss")
    plt.plot(ks, recon_loss, marker="o", label="reconstruction")
    plt.plot(ks, arche_loss, marker="o", label="archetypal")
    plt.xlabel("k (archetypes)")
    plt.ylabel("loss")
    plt.title(f"Elbow curve – {cluster_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cluster_dir / "elbow.png", dpi=200)
    plt.close()


def _assemble_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.aanet_epochs,
        batch_size=args.aanet_batch_size,
        learning_rate=args.aanet_lr,
        weight_decay=args.aanet_weight_decay,
        gamma_reconstruction=args.aanet_gamma_reconstruction,
        gamma_archetypal=args.aanet_gamma_archetypal,
        gamma_extrema=args.aanet_gamma_extrema,
        simplex_scale=args.aanet_simplex_scale,
        noise=args.aanet_noise,
        layer_widths=args.aanet_layer_widths,
        min_samples=args.aanet_min_samples,
        num_workers=args.aanet_num_workers,
        shuffle=True,
        val_fraction=args.aanet_val_fraction,
        min_val_size=args.aanet_val_min_size,
        early_stop_patience=args.aanet_early_stop_patience,
        early_stop_delta=args.aanet_early_stop_delta,
        lr_patience=args.aanet_lr_patience,
        lr_factor=args.aanet_lr_factor,
        grad_clip=args.aanet_grad_clip,
        restarts_no_extrema=args.aanet_restarts_no_extrema,
    )


def _assemble_extrema_config(args: argparse.Namespace) -> ExtremaConfig:
    return ExtremaConfig(
        enabled=args.extrema_enabled,
        knn=args.extrema_knn,
        subsample=not args.extrema_disable_subsample,
        max_points=args.extrema_max_points,
        random_seed=args.extrema_seed,
    )


def _summarize_dataset(
    result: ClusterDatasetResult,
    token_positions: Sequence[int] | None,
    noise_value: float,
    noise_mode: str,
) -> dict:
    return {
        "cluster_id": result.descriptor.cluster_id,
        "cluster_label": result.descriptor.label,
        "kept_samples": result.kept_samples,
        "total_samples": result.total_samples,
        "ignored_fraction": result.ignored_fraction,
        "num_features": int(result.data.shape[1]) if result.data.ndim == 2 else 0,
        "num_rows": int(result.data.shape[0]),
        "component_names": list(result.descriptor.component_names),
        "is_noise": result.descriptor.is_noise,
        "token_indices": list(token_positions) if token_positions is not None else None,
        "aanet_noise_value": float(noise_value),
        "aanet_noise_mode": noise_mode,
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    # args.output_dir is a Path object
    output_root = args.output_dir
    _ensure_dir(output_root, overwrite=True)

    model = _load_transformer(args, device)
    sampler = _load_sampler(args)
    training_config = _assemble_training_config(args)
    extrema_config = _assemble_extrema_config(args)
    max_k = max(args.k_values)

    layer_summaries: Dict[int, List[dict]] = {}

    for layer in args.layers:
        print(f"Processing layer {layer}")
        hook_name = f"blocks.{layer}.hook_resid_post"
        sae_path = args.sae_root / f"layer_{layer}_top_k_k{args.topk}.pt"
        if not sae_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")
        sae = _instantiate_sae(sae_path, device)
        summary_path = args.cluster_summary_dir / args.cluster_summary_pattern.format(layer=layer)
        if not summary_path.exists():
            raise FileNotFoundError(f"Cluster summary not found: {summary_path}")

        summary = load_cluster_summary(summary_path)
        descriptors = parse_cluster_descriptors(summary, include_noise=True)
        if not descriptors:
            print(f"No clusters found for layer {layer}, skipping.")
            continue

        datasets, _ = build_cluster_datasets(
            model=model,
            sampler=sampler,
            layer_hook=hook_name,
            sae=sae,
            cluster_descriptors=descriptors,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_batches=args.num_batches,
            activation_threshold=args.activation_threshold,
            device=device,
            max_samples_per_cluster=args.max_samples_per_cluster,
            min_cluster_samples=args.min_cluster_samples,
            seed=args.sampling_seed + layer,
            token_positions=args.token_indices,
        )

        layer_records: List[dict] = []
        layer_dir = output_root / f"layer_{layer}"
        _ensure_dir(layer_dir, overwrite=args.overwrite)

        extrema_cache: Dict[int, torch.Tensor | None] = {}
        for desc in descriptors:
            dataset = datasets[desc.cluster_id]
            print(f"Cluster {desc.cluster_id} ({desc.label}): computing extrema candidates")
            if not extrema_config.enabled:
                extrema_cache[desc.cluster_id] = None
                continue

            data_array_full = dataset.data.detach().cpu().numpy()
            filtered_array = data_array_full
            base_cfg = ExtremaConfig(
                enabled=True,
                knn=extrema_config.knn,
                subsample=extrema_config.subsample,
                max_points=extrema_config.max_points,
                random_seed=extrema_config.random_seed + layer * 101 + desc.cluster_id * 13,
            )

            cfg_current = base_cfg
            component_checked = False
            extrema_tensor = None
            bumped_once = False

            while True:
                try:
                    extrema_tensor = compute_diffusion_extrema(
                        filtered_array,
                        max_k=max_k,
                        config=cfg_current,
                    )
                    break
                except nx.NetworkXError as err:
                    if "graph is not connected" not in str(err):
                        raise

                    if not component_checked:
                        component_checked = True
                        knn_for_graph = cfg_current.knn
                        if filtered_array.shape[0] > 1:
                            knn_for_graph = min(cfg_current.knn, max(filtered_array.shape[0] - 1, 1))
                        else:
                            knn_for_graph = 1
                        try:
                            graph = gt.Graph(filtered_array, use_pygsp=True, decay=None, knn=knn_for_graph)
                            graph_nx = nx.convert_matrix.from_scipy_sparse_array(graph.W)
                            components = list(nx.connected_components(graph_nx))
                        except Exception as graph_err:
                            print(
                                f"Cluster {desc.cluster_id}: failed to analyze connectivity ({graph_err});"
                                " proceeding to retry with bumped parameters."
                            )
                            components = [set(range(filtered_array.shape[0]))]

                        if len(components) > 1:
                            largest = max(components, key=len)
                            largest_size = len(largest)
                            if largest_size < filtered_array.shape[0]:
                                idx = np.array(sorted(largest), dtype=np.int64)
                                print(
                                    f"Cluster {desc.cluster_id}: keeping largest connected component "
                                    f"({largest_size}/{filtered_array.shape[0]} samples)."
                                )
                                filtered_array = filtered_array[idx]
                                continue  # retry with same config on filtered data

                    # Either component already checked or dropping didn't help -> bump params once
                    if not bumped_once:
                        bumped_once = True
                        bumped_knn = int(math.ceil(cfg_current.knn * 1.5))
                        bumped_max = (
                            None
                            if cfg_current.max_points is None
                            else int(math.ceil(cfg_current.max_points * 1.5))
                        )
                        cfg_current = ExtremaConfig(
                            enabled=True,
                            knn=bumped_knn,
                            subsample=cfg_current.subsample,
                            max_points=bumped_max,
                            random_seed=cfg_current.random_seed + 1,
                        )
                        print(
                            f"Cluster {desc.cluster_id}: extrema graph still disconnected; retrying with "
                            f"knn={bumped_knn}, max_points={bumped_max if bumped_max is not None else 'None'}."
                        )
                        continue

                    print(f"Cluster {desc.cluster_id}: extrema computation failed after retries; aborting.")
                    raise

            extrema_cache[desc.cluster_id] = extrema_tensor

        for desc in descriptors:
            print(f"Cluster {desc.cluster_id} ({desc.label}): training AAnet models")
            dataset = datasets[desc.cluster_id]
            data_tensor = dataset.data.to(device)
            cluster_dir = layer_dir / _slugify(desc.label)
            _ensure_dir(cluster_dir, overwrite=args.overwrite)

            if args.aanet_noise_relative and data_tensor is not None and data_tensor.numel() > 0:
                data_std = float(data_tensor.std().item())
                noise_value = args.aanet_noise * data_std
            else:
                noise_value = args.aanet_noise
            noise_mode = "relative" if args.aanet_noise_relative else "absolute"

            dataset_summary = _summarize_dataset(dataset, args.token_indices, noise_value, noise_mode)
            extrema_tensor = extrema_cache.get(desc.cluster_id)
            if extrema_tensor is not None:
                extrema_tensor = extrema_tensor.to(device)

            cluster_results: List[dict] = []
            for k_idx, k in enumerate(sorted(args.k_values)):
                run_seed = args.aanet_seed + k_idx + layer * 101 + desc.cluster_id * 17
                print(
                    f"Cluster {desc.cluster_id}: training AAnet for simplex size {k-1} "
                    f"(max {max_k-1})"
                )
                result = train_aanet_model(
                    data_tensor,
                    k=k,
                    config=training_config,
                    device=device,
                    diffusion_extrema=extrema_tensor,
                    seed=run_seed,
                    noise_override=noise_value,
                )
                run_dir = cluster_dir / f"k_{k}"
                _ensure_dir(run_dir, overwrite=args.overwrite)

                record = {
                    "layer": layer,
                    "cluster_id": desc.cluster_id,
                    "cluster_label": desc.label,
                    "component_names": list(desc.component_names),
                    "is_noise": desc.is_noise,
                    "k": k,
                    "status": result.status,
                    "ignored_fraction": dataset.ignored_fraction,
                    "kept_samples": dataset.kept_samples,
                    "total_samples": dataset.total_samples,
                    "token_indices": list(args.token_indices) if args.token_indices is not None else None,
                    "metrics": result.metrics,
                    "noise_value": noise_value,
                    "noise_mode": noise_mode,
                }
                cluster_results.append(record)
                layer_records.append(record)

                _write_json(run_dir / "metrics.json", record)
                _write_csv(run_dir / "epoch_metrics.csv", result.epoch_metrics)

                if args.save_models and result.status == "ok":
                    payload = {
                        "state_dict": result.model_state_dict,
                        "aanet_config": {
                            "k": k,
                            "layer_widths": training_config.layer_widths,
                            "simplex_scale": training_config.simplex_scale,
                            "noise": training_config.noise,
                            "epochs": training_config.epochs,
                            "ignored_fraction": dataset.ignored_fraction,
                        },
                        "cluster": {
                            "id": desc.cluster_id,
                            "label": desc.label,
                            "is_noise": desc.is_noise,
                        },
                    }
                    torch.save(payload, run_dir / "model.pt")

            _write_json(cluster_dir / "dataset_summary.json", dataset_summary)
            _make_cluster_plot(cluster_dir, desc.label, cluster_results)

        layer_summaries[layer] = layer_records
        _write_json(layer_dir / "layer_summary.json", {"runs": layer_records})

    overall = [{"layer": layer, "runs": records} for layer, records in layer_summaries.items()]
    _write_json(output_root / "summary.json", {"layers": overall})


if __name__ == "__main__":
    main()
