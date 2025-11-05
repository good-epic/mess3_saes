#!/usr/bin/env python3
"""
Analyze SubspacePartition rotations on multipartite transformer checkpoints.

For each requested activation site, this script:
  * loads the learned orthogonal matrix R and partition metadata,
  * samples activations/belief states from the configured generative process,
  * rotates activations into subspace coordinates and gathers activity stats,
  * computes per-subspace belief-state R² scores,
  * (optionally) evaluates Gromov-Wasserstein fits against reference geometries,
  * writes consolidated CSV + JSON summaries.

The goal is to mirror the downstream analysis previously built around
Sparse Autoencoders, without depending on SAE checkpoints.
"""
from __future__ import annotations

## REQUIRED. NEVER DELETE THIS ##
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
##################################

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from clustering.analysis import ClusterAnalyzer
from clustering.config import AnalysisConfig, GeometryFittingConfig
from clustering.geometry_fitting import GeometryFitter
from clustering.geometries import CircleGeometry, HypersphereGeometry, SimplexGeometry
from multipartite_utils import (
    MultipartiteSampler,
    _load_process_stack,
    _load_transformer,
    _resolve_device,
)
from training_and_analysis_utils import _generate_sequences, _tokens_from_observations


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

SITE_HOOK_MAP = {
    "embeddings": "hook_embed",
    "layer_0": "blocks.0.hook_resid_post",
    "layer_1": "blocks.1.hook_resid_post",
    "layer_2": "blocks.2.hook_resid_post",
    "layer_3": "blocks.3.hook_resid_post",
}


def default_sp_short_name(site: str) -> str:
    """Map a site name to the short-name convention used by SubspacePartition."""
    if site.startswith("layer_"):
        layer_idx = int(site.split("_")[1])
        return f"x{layer_idx}.post"
    if site == "embeddings":
        return "x0.pre"
    raise ValueError(f"Unsupported site name: {site}")


@dataclass
class SubspacePartition:
    R: torch.Tensor  # (d, d)
    partition: List[int]

    def to_device(self, device: str) -> "SubspacePartition":
        R_device = self.R.to(device)
        return SubspacePartition(R=R_device, partition=list(self.partition))

    @property
    def dimension(self) -> int:
        return self.R.shape[0]

    @property
    def num_subspaces(self) -> int:
        return len(self.partition)

    def edges(self) -> List[int]:
        edges = [0]
        total = 0
        for dim in self.partition:
            total += dim
            edges.append(total)
        return edges

    def rotate(self, acts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return acts.to(self.R.device, dtype=dtype) @ self.R.to(dtype)

    def reconstruct(self, rotated: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return rotated.to(self.R.device, dtype=dtype) @ self.R.t().to(dtype)


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def load_subspace_partition(
    base_dir: Path,
    model_tag: str,
    site_short_name: str,
    device: str,
) -> SubspacePartition:
    """Load rotation matrix and partition metadata for a given site."""
    config_path = base_dir / f"R_config-{model_tag}-{site_short_name}.json"
    weights_path = base_dir / f"R-{model_tag}-{site_short_name}.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Partition config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Rotation weights not found: {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    partition = config.get("partition")
    if not partition:
        raise ValueError(f"Partition list missing in {config_path}")

    state = torch.load(weights_path, map_location=device)
    key = "R.parametrizations.weight.0.base"
    if key not in state:
        raise KeyError(f"{weights_path} missing expected tensor key '{key}'")
    R = state[key].clone().to(device)

    return SubspacePartition(R=R, partition=partition)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def flatten_component_beliefs(
    component_order: List[str],
    component_arrays: List[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Concatenate component belief matrices and record their slices."""
    slices: Dict[str, Tuple[int, int]] = {}
    flattened = []
    offset = 0
    for name, arr in zip(component_order, component_arrays):
        flat = arr.reshape(-1, arr.shape[-1])
        flattened.append(flat)
        slices[name] = (offset, offset + flat.shape[-1])
        offset += flat.shape[-1]
    return np.concatenate(flattened, axis=1), slices


def collect_activations_and_beliefs(
    model,
    data_source,
    hook_name: str,
    total_required: int,
    sample_positions: Sequence[int],
    batch_size: int,
    seq_len: int,
    device: str,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, np.ndarray], List[str]]:
    """Sample activations and component belief arrays for a single site."""
    if not sample_positions:
        raise ValueError("At least one sample position must be provided")

    max_position = max(sample_positions)
    if seq_len <= max_position:
        raise ValueError(
            f"Sequence length {seq_len} too short for sample positions {sample_positions}"
        )

    key = jax.random.PRNGKey(seed)
    total_collected = 0

    component_buffers: Dict[str, List[np.ndarray]] = {}
    activations: List[torch.Tensor] = []

    # Determine component metadata from data_source
    if isinstance(data_source, MultipartiteSampler):
        component_order = [str(comp.name) for comp in data_source.components]
        component_dims = [comp.state_dim for comp in data_source.components]
        for comp in data_source.components:
            component_buffers[str(comp.name)] = []
    else:
        component_order = [getattr(data_source, "name", "process")]
        component_dims = [data_source.state_dim]
        component_buffers[component_order[0]] = []

    pbar = tqdm(total=total_required, desc=f"Collecting {hook_name}", unit="acts")

    while total_collected < total_required:
        remaining = total_required - total_collected
        per_sequence = len(sample_positions)
        sequences_needed = math.ceil(remaining / per_sequence)
        current_batch = min(batch_size, max(1, sequences_needed))

        if isinstance(data_source, MultipartiteSampler):
            key, beliefs_batch, product_tokens, _ = data_source.sample(key, current_batch, seq_len)
            tokens = torch.from_numpy(np.array(product_tokens)).long().to(device)
            beliefs_np = np.array(beliefs_batch)
        else:
            key, states_eval, observations = _generate_sequences(
                key,
                batch_size=current_batch,
                sequence_len=seq_len,
                source=data_source,
            )
            tokens = _tokens_from_observations(observations, device=device)
            beliefs_np = np.array(states_eval)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None)

        acts = cache[hook_name][:, sample_positions, :]
        acts = acts.reshape(-1, acts.shape[-1]).detach().cpu()
        activations.append(acts)

        belief_subset = beliefs_np[:, sample_positions, :]
        samples_added = belief_subset.shape[0] * belief_subset.shape[1]

        cursor = 0
        for comp_name, dim in zip(component_order, component_dims):
            comp_slice = belief_subset[..., cursor : cursor + dim]
            component_buffers[comp_name].append(comp_slice.reshape(-1, dim))
            cursor += dim

        total_collected += samples_added
        pbar.update(samples_added)
        model.reset_hooks()

    pbar.close()

    acts_tensor = torch.cat(activations, dim=0)
    desired_samples = min(total_required, acts_tensor.shape[0])
    acts_tensor = acts_tensor[:desired_samples].contiguous()

    component_arrays: Dict[str, np.ndarray] = {}
    for comp_name, buffers in component_buffers.items():
        comp_arr = np.concatenate(buffers, axis=0)
        component_arrays[comp_name] = comp_arr[:desired_samples]

    return acts_tensor, component_arrays, component_order


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_activity_stats(
    rotated_acts: np.ndarray,
    partition: List[int],
    activation_eps: float,
) -> List[Dict[str, Any]]:
    """Compute activity metrics for each subspace."""
    stats: List[Dict[str, Any]] = []
    edges = np.cumsum([0] + partition)

    for idx, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        sub_acts = rotated_acts[:, start:end]
        abs_vals = np.abs(sub_acts)
        activation_mask = np.any(abs_vals > activation_eps, axis=1)

        stats.append(
            {
                "subspace_index": idx,
                "dimension": int(end - start),
                "activation_rate": float(activation_mask.mean()),
                "mean_abs_activation": float(abs_vals.mean()),
                "variance_sum": float(np.var(sub_acts, axis=0).sum()),
                "max_abs_activation": float(abs_vals.max()),
            }
        )
    return stats


def prepare_cluster_labels(partition: List[int]) -> np.ndarray:
    """Create cluster-label array assigning each rotated axis to its subspace."""
    labels = []
    for idx, dim in enumerate(partition):
        labels.extend([idx] * dim)
    return np.array(labels, dtype=int)


def build_geometry_candidates(config: GeometryFittingConfig) -> List[Any]:
    """Instantiate geometry objects based on configuration."""
    geometries: List[Any] = []
    k_min, k_max = config.simplex_k_range
    for k in range(k_min, k_max + 1):
        geometries.append(SimplexGeometry(k))
    if config.include_circle:
        geometries.append(CircleGeometry())
    if config.include_hypersphere:
        for dim in config.hypersphere_dims:
            geometries.append(HypersphereGeometry(dim))
    return geometries


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SubspacePartition rotations for multipartite transformers",
        allow_abbrev=False,
    )

    # Paths and devices
    parser.add_argument("--subspace_dir", type=str, required=True, help="Directory containing trained SubspacePartition matrices (e.g., SubspacePartition/trainedRs/<exp>)")
    parser.add_argument("--subspace_model_tag", type=str, required=True, help="Model tag used in SubspacePartition filenames (e.g., gpt2)")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Transformer checkpoint path")
    parser.add_argument("--process_config", type=str, default="process_configs.json", help="Process configuration JSON")
    parser.add_argument("--process_config_name", type=str, default="3xmess3_2xtquant_003", help="Named configuration within process_config")
    parser.add_argument("--process_preset", type=str, default=None, help="Optional preset configuration name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for analysis outputs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")

    # Transformer config fallbacks (used when the checkpoint lacks a saved config)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--d_vocab", type=int, default=None)
    parser.add_argument("--act_fn", type=str, default="relu")

    # Sampling settings
    parser.add_argument("--sites", type=str, nargs="+", required=True, help="List of site names (e.g., layer_0 layer_1)")
    parser.add_argument("--sample_positions", type=int, nargs="+", default=[4, 14], help="Token positions (0-indexed) to sample per sequence")
    parser.add_argument("--sample_seq_len", type=int, default=None, help="Sequence length for sampling (defaults to model n_ctx)")
    parser.add_argument("--total_samples_per_site", type=int, default=25000, help="Approximate number of activation positions to collect per site")
    parser.add_argument("--batch_size", type=int, default=2048, help="Sequences per sampling batch")
    parser.add_argument("--activation_eps", type=float, default=1e-6, help="Threshold for counting activations as non-zero")

    # Belief prediction
    parser.add_argument("--ridge_alpha", type=float, default=1e-3, help="Ridge regularization for belief R² fits")

    # Geometry fitting options
    parser.add_argument("--geometry", action="store_true", help="Enable geometry fitting via Gromov-Wasserstein")
    parser.add_argument("--geo_simplex_k_min", type=int, default=1)
    parser.add_argument("--geo_simplex_k_max", type=int, default=6)
    parser.add_argument("--geo_include_circle", action="store_true")
    parser.add_argument("--geo_include_hypersphere", action="store_true")
    parser.add_argument("--geo_hypersphere_dims", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--geo_n_target_samples", type=int, default=1000)
    parser.add_argument("--geo_sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument("--geo_sinkhorn_max_iter", type=int, default=1000)
    parser.add_argument("--geo_cost_fn", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--geo_per_point_threshold", type=float, default=0.5)
    parser.add_argument("--geo_threshold_mode", type=str, default="normalized", choices=["normalized", "raw"])

    if argv is None:
        args, _unknown = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    subspace_dir = Path(args.subspace_dir)

    process_cfg_raw, components, data_source = _load_process_stack(args, None)
    model, model_cfg = _load_transformer(args, device, getattr(data_source, "vocab_size", None))
    model.eval()

    seq_len = args.sample_seq_len if args.sample_seq_len is not None else model_cfg.n_ctx

    analyzer = ClusterAnalyzer(AnalysisConfig(skip_pca_plots=True))

    geometry_config = GeometryFittingConfig(
        enabled=args.geometry,
        simplex_k_range=(args.geo_simplex_k_min, args.geo_simplex_k_max),
        include_circle=args.geo_include_circle,
        include_hypersphere=args.geo_include_hypersphere,
        hypersphere_dims=args.geo_hypersphere_dims,
        n_target_samples=args.geo_n_target_samples,
        sinkhorn_epsilon=args.geo_sinkhorn_epsilon,
        sinkhorn_max_iter=args.geo_sinkhorn_max_iter,
        cost_fn=args.geo_cost_fn,
        per_point_threshold=args.geo_per_point_threshold,
        threshold_mode=args.geo_threshold_mode,
    )
    geometry_fitter = GeometryFitter(geometry_config)
    geometry_candidates = build_geometry_candidates(geometry_config) if args.geometry else []

    all_rows: List[Dict[str, Any]] = []
    geometry_details: Dict[str, Any] = {}

    for site in args.sites:
        if site not in SITE_HOOK_MAP:
            raise ValueError(f"Unknown site '{site}'. Available: {list(SITE_HOOK_MAP)}")

        hook_name = SITE_HOOK_MAP[site]
        sp_short = default_sp_short_name(site)

        partition = load_subspace_partition(
            base_dir=subspace_dir,
            model_tag=args.subspace_model_tag,
            site_short_name=sp_short,
            device=device,
        )

        acts_flat, component_arrays, component_order = collect_activations_and_beliefs(
            model=model,
            data_source=data_source,
            hook_name=hook_name,
            total_required=args.total_samples_per_site,
            sample_positions=args.sample_positions,
            batch_size=args.batch_size,
            seq_len=seq_len,
            device=device,
            seed=args.seed,
        )

        acts_flat = acts_flat.to(device)

        component_concat, component_slices = flatten_component_beliefs(
            component_order, [component_arrays[name] for name in component_order]
        )
        component_beliefs_flat = {
            name: component_concat[:, start:end]
            for name, (start, end) in component_slices.items()
        }

        rotated = partition.rotate(acts_flat)
        rotated_np = rotated.detach().cpu().numpy()

        activity_stats = compute_activity_stats(rotated_np, partition.partition, args.activation_eps)

        cluster_labels = prepare_cluster_labels(partition.partition)
        r2_summary = analyzer.compute_belief_r2(
            acts_flat,
            rotated_np,
            partition.R.detach().cpu().numpy().T,
            cluster_labels,
            partition.num_subspaces,
            component_beliefs_flat,
            component_order,
            args.ridge_alpha,
            site,
            soft_assignments=None,
            assignment_name="hard",
        )

        edges = partition.edges()
        for stat in activity_stats:
            idx = stat["subspace_index"]
            start, end = edges[idx], edges[idx + 1]
            subspace_key = f"{site}:{idx}"

            row: Dict[str, Any] = {
                "site": site,
                "subspace_index": idx,
                "dimension": stat["dimension"],
                "activation_rate": stat["activation_rate"],
                "mean_abs_activation": stat["mean_abs_activation"],
                "variance_sum": stat["variance_sum"],
                "max_abs_activation": stat["max_abs_activation"],
            }

            if r2_summary and idx in r2_summary:
                for comp_name, comp_metrics in r2_summary[idx].items():
                    row[f"r2_mean::{comp_name}"] = comp_metrics.get("mean_r2", float("nan"))
                    row[f"r2_samples::{comp_name}"] = comp_metrics.get("n_active_samples", 0)
                    row[f"r2_dims_dropped::{comp_name}"] = comp_metrics.get("dropped_constant_dims", 0)

            if args.geometry:
                cluster_points = partition.R[:, start:end].detach().cpu().numpy().T
                geom_results = geometry_fitter.fit_cluster_to_geometries(cluster_points, geometry_candidates)
                if geom_results:
                    best_geom = min(geom_results.items(), key=lambda item: item[1].optimal_distance)
                    row["geometry_best_name"] = best_geom[0]
                    row["geometry_best_distance"] = float(best_geom[1].optimal_distance)
                    geometry_details[subspace_key] = {
                        "site": site,
                        "subspace_index": idx,
                        "results": {
                            name: {
                                "optimal_distance": float(res.optimal_distance),
                                "gw_distortion_stats": {
                                    "mean": float(np.mean(res.gw_distortion_contributions)),
                                    "std": float(np.std(res.gw_distortion_contributions)),
                                    "min": float(np.min(res.gw_distortion_contributions)),
                                    "max": float(np.max(res.gw_distortion_contributions)),
                                },
                            }
                            for name, res in geom_results.items()
                        },
                    }

            all_rows.append(row)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    import pandas as pd  # Local import to avoid dependency if unused

    df = pd.DataFrame(all_rows)
    df.sort_values(by=["site", "subspace_index"], inplace=True)
    df_path = output_dir / "subspace_partition_metrics.csv"
    df.to_csv(df_path, index=False)
    print(f"Wrote per-subspace metrics to {df_path}")

    if geometry_details:
        geometry_path = output_dir / "subspace_partition_geometry.json"
        with open(geometry_path, "w", encoding="utf-8") as f:
            json.dump(geometry_details, f, indent=2)
        print(f"Wrote geometry fit details to {geometry_path}")


if __name__ == "__main__":
    main()
