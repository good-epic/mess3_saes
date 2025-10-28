#!/usr/bin/env python3
"""
Per-latent analysis of belief prediction strength and Lasso selections.

This script mirrors the loading utilities used in `fit_mess3_gmg.py` to:
1. Load a trained transformer and the associated SAE checkpoints.
2. Sample sequences from the configured process and collect activations.
3. For each latent in each selected site, compute:
   - Average single-latent R² when predicting component belief states.
   - Binary indicators for whether the latent is selected by Lasso fits
     (over a sweep of user-specified lambda values) for each component.
4. Emit a table with one row per latent containing the above metrics.
"""

#%%
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import sys
import argparse
import json
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend only when NOT running in IPython/Jupyter.
_IN_IPYTHON = False
try:
    from IPython import get_ipython  # type: ignore
    _IN_IPYTHON = get_ipython() is not None
except Exception:
    _IN_IPYTHON = False

if not _IN_IPYTHON:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import jax
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import pickle

from multipartite_utils import (
    MultipartiteSampler,
    _load_process_stack,
    _load_transformer,
    _resolve_device,
    _select_sites,
)
from mess3_gmg_analysis_utils import fit_residual_to_belief_map, lasso_select_latents, load_metrics_summary, sae_encode_features
from training_and_analysis_utils import _generate_sequences, _tokens_from_observations


SITE_HOOK_MAP = {
    "embeddings": "hook_embed",
    "layer_0": "blocks.0.hook_resid_post",
    "layer_1": "blocks.1.hook_resid_post",
    "layer_2": "blocks.2.hook_resid_post",
    "layer_3": "blocks.3.hook_resid_post",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-latent belief geometry diagnostics", allow_abbrev=False)

    # Paths and model configuration
    parser.add_argument("--sae_folder", type=str, default="outputs/saes/multipartite_003e", help="Directory containing SAE checkpoints")
    parser.add_argument("--metrics_summary", type=str, default="outputs/saes/multipartite_003e/metrics_summary.json", help="Optional path to metrics_summary.json")
    parser.add_argument("--model_ckpt", type=str, default="outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt", help="Transformer checkpoint path")
    parser.add_argument("--output_path", type=str, default="outputs/reports/multipartite_003e/ground_truth_diagnostics/", help="Path to save output table (CSV or Parquet)")
    #parser.add_argument("--input_csv", type=str, default="outputs/reports/multipartite_003e/ground_truth_diagnostics/ground_truth_metrics.csv", help="Optional existing CSV/Parquet to load instead of recomputing metrics")
    parser.add_argument("--input_csv", type=str, default=None, help="Optional existing CSV/Parquet to load instead of recomputing metrics")
    parser.add_argument("--latent_distance_pkl", type=str, default="latent_distance_matrices.pkl", help="Optional existing latent distance matrices pickle file")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"], help="Computation device")

    # Process config
    parser.add_argument("--process_config", type=str, default="process_configs.json", help="Process configuration JSON")
    parser.add_argument("--process_config_name", type=str, default="3xmess3_2xtquant_003", help="Named configuration within process_config")
    parser.add_argument("--process_preset", type=str, default=None, help="Alternative preset configuration name")

    # Transformer config fallbacks
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--d_vocab", type=int, default=None)
    parser.add_argument("--act_fn", type=str, default="relu")

    # SAE controls
    parser.add_argument("--top_k_vals", type=int, nargs="+", default=[8], help="List of K values for top-k SAEs")
    parser.add_argument("--lambda_vals", type=float, nargs="+", default=None, help="List of lambda values for vanilla SAEs")
    parser.add_argument("--sites", type=str, nargs="+", default=None, help="Subset of site names to analyze")

    # Sampling
    parser.add_argument("--transformer_batch_size", type=int, default=2048, help="Number of sequences per forward pass when sampling activations")
    parser.add_argument("--transformer_total_sample_size", type=int, default=25000, help="Target number of token positions to collect for regression and activation statistics")
    parser.add_argument("--sample_seq_len", type=int, default=None, help="Sequence length for sampling (default: model n_ctx)")
    parser.add_argument("--activation_eps", type=float, default=1e-6, help="Threshold when counting latent activations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--act_rate_threshold", type=float, default=0.0, help="Minimum activation rate to include a latent in distance metrics")

    # Elastic net / stability selection sweep
    parser.add_argument("--lasso_lambdas", type=float, nargs="+", default=[5e-4, 1e-3, 2e-3, 3e-3, 4e-3], help="Elastic net alpha values to test")
    parser.add_argument("--lasso_cv", type=int, default=8, help="Legacy fallback CV folds when alpha is None")
    parser.add_argument("--lasso_max_iter", type=int, default=10000, help="Max iterations for ElasticNet solver")
    parser.add_argument("--elasticnet_l1_ratio", type=float, default=0.9, help="Elastic net l1_ratio (fraction of L1 penalty)")
    parser.add_argument("--elasticnet_stability_fraction", type=float, default=0.5, help="Fraction of samples per stability-selection resample")
    parser.add_argument("--elasticnet_stability_runs", type=int, default=50, help="Number of resampled fits for stability selection")
    parser.add_argument("--elasticnet_selection_threshold", type=float, default=0.7, help="Minimum selection frequency to keep a latent")
    parser.add_argument("--r2_threshold", type=float, default=0.001, help="Threshold for counting component R² hits")
    # Output format
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet"], help="Explicit output format (otherwise inferred from path)")

    # In interactive environments (e.g., Jupyter/VSCode #%%), ipykernel passes
    # extra args like "-f <kernel.json>" which can confuse argparse. Use
    # parse_known_args to ignore unknowns by default when argv is None.
    if argv is None:
        args, _unknown = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)

    if (not args.top_k_vals or len(args.top_k_vals) == 0) and (not args.lambda_vals or len(args.lambda_vals) == 0):
        parser.error("Provide at least one value via --top_k_vals or --lambda_vals")

    return args


def determine_output_format(path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return "parquet"
    return "csv"


def load_metrics_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unrecognized file extension for {path}")


def select_sites(metrics_summary: Optional[Dict[str, Any]]) -> List[str]:
    if metrics_summary:
        available = [site for site in SITE_HOOK_MAP if site in metrics_summary]
        if available:
            return available
    return list(SITE_HOOK_MAP.keys())


def load_sae_checkpoint(
    sae_folder: str,
    site: str,
    sae_type: str,
    param_value: float,
    device: str,
):
    if sae_type == "top_k":
        filename = f"{site}_top_k_k{int(param_value)}.pt"
        from BatchTopK.sae import TopKSAE as SAEClass
    else:
        filename = f"{site}_vanilla_lambda_{param_value}.pt"
        from BatchTopK.sae import VanillaSAE as SAEClass

    path = os.path.join(sae_folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SAE checkpoint not found at {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["cfg"])
    cfg["device"] = "cuda" if device.startswith("cuda") else "cpu"
    sae = SAEClass(cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    decoder_dirs = sae.W_dec.detach().cpu().numpy()
    return sae, decoder_dirs


def flatten_component_beliefs(
    component_order: List[str],
    component_arrays: List[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Concatenate component belief matrices and record slices."""
    slices: Dict[str, Tuple[int, int]] = {}
    flattened = []
    offset = 0
    for name, arr in zip(component_order, component_arrays):
        flat = arr.reshape(-1, arr.shape[-1])
        flattened.append(flat)
        start, end = offset, offset + flat.shape[1]
        slices[name] = (start, end)
        offset = end
    belief_concat = np.concatenate(flattened, axis=1) if flattened else np.empty((0, 0))
    return belief_concat, slices


def infer_component_display_names(component_order: List[str], component_meta: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    counters: Dict[str, int] = defaultdict(int)
    display: Dict[str, str] = {}
    for name in component_order:
        meta = component_meta.get(name, {})
        comp_type = meta.get("type") or name.split("_")[0]
        idx = counters[comp_type]
        if idx == 0:
            label = comp_type
        else:
            label = f"{comp_type}_{idx}"
        counters[comp_type] += 1
        display[name] = label
    return display


def component_dim_subset(component_label: str, slice_range: Tuple[int, int]) -> List[int]:
    """Return global dimension indices to include for averaging per component."""
    start, end = slice_range
    dims = list(range(start, end))
    if component_label.startswith("tom_quantum") or component_label.startswith("tom") or "tom_quantum" in component_label:
        # Use second and third dims if available
        if end - start >= 3:
            return [start + 1, start + 2]
        if end - start >= 2:
            return [start + 1]
    return dims


def run_analysis(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, float], Dict[str, Any]]]:
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    metrics_path = args.metrics_summary or os.path.join(args.sae_folder, "metrics_summary.json")
    metrics_summary = load_metrics_summary(metrics_path)
    if metrics_summary is None:
        print(f"Warning: unable to load metrics summary from {metrics_path}")

    process_cfg_raw, components, data_source = _load_process_stack(args, None)

    if isinstance(data_source, MultipartiteSampler):
        vocab_size = data_source.vocab_size
        component_order = [str(comp.name) for comp in data_source.components]
        component_dims: List[int] = [comp.state_dim for comp in data_source.components]
        component_meta_map: Dict[str, Dict[str, Any]] = {}
        for comp in data_source.components:
            component_meta_map[str(comp.name)] = {
                "name": str(comp.name),
                "type": comp.process_type,
                "state_dim": comp.state_dim,
            }
    else:
        vocab_size = data_source.vocab_size
        component_order = [getattr(data_source, "name", "process")]
        component_dims = []
        component_meta_map: Dict[str, Dict[str, Any]] = {}

    model, cfg = _load_transformer(args, device, vocab_size)
    model.eval()

    # Determine sample length
    seq_len = args.sample_seq_len if args.sample_seq_len is not None else cfg.n_ctx

    sample_positions = [4, 14]  # 5th and 15th tokens (0-indexed)
    max_position = max(sample_positions)
    if seq_len <= max_position:
        raise ValueError(
            f"Sequence length {seq_len} too short to sample positions {sample_positions}. "
            "Increase --sample_seq_len or transformer context."
        )

    total_required = max(1, args.transformer_total_sample_size)
    batch_size = max(1, args.transformer_batch_size)
    key = jax.random.PRNGKey(args.seed)

    site_filter = args.sites if args.sites else None
    if metrics_summary:
        available_sites = _select_sites(metrics_summary, site_filter, SITE_HOOK_MAP)
    else:
        available_sites = [site for site in site_filter if site in SITE_HOOK_MAP]
    if not available_sites:
        raise ValueError("No valid sites selected for analysis")

    site_hook_map_subset = {site: SITE_HOOK_MAP[site] for site in available_sites}

    component_buffers: Dict[str, List[np.ndarray]] = {name: [] for name in component_order}
    site_activation_buffers: Dict[str, List[torch.Tensor]] = {site: [] for site in site_hook_map_subset}
    total_samples_collected = 0

    while total_samples_collected < total_required:
        remaining = total_required - total_samples_collected
        per_sequence_samples = len(sample_positions)
        sequences_needed = math.ceil(remaining / per_sequence_samples)
        current_batch_size = min(batch_size, max(1, sequences_needed))

        if isinstance(data_source, MultipartiteSampler):
            key, belief_states, product_tokens, _ = data_source.sample(key, current_batch_size, seq_len)
            tokens_tensor = torch.from_numpy(np.array(product_tokens)).long().to(device)
            beliefs_batch_np = np.array(belief_states)
        else:
            key, states_eval, observations = _generate_sequences(
                key,
                batch_size=current_batch_size,
                sequence_len=seq_len,
                source=data_source,
            )
            tokens_tensor = _tokens_from_observations(observations, device=device)
            beliefs_batch_np = np.array(states_eval)

        if beliefs_batch_np.shape[1] <= max_position:
            raise ValueError(
                f"Collected beliefs with length {beliefs_batch_np.shape[1]} shorter than required index {max_position}."
            )

        belief_subset = beliefs_batch_np[:, sample_positions, :]
        samples_added = belief_subset.shape[0] * belief_subset.shape[1]

        if not component_dims:
            belief_dim = belief_subset.shape[-1]
            component_dims = [belief_dim]
            comp_name = component_order[0]
            component_meta_map[comp_name] = {
                "name": comp_name,
                "type": getattr(data_source, "process_type", comp_name),
                "state_dim": belief_dim,
            }

        cursor = 0
        for comp_name, dim in zip(component_order, component_dims):
            comp_slice = belief_subset[..., cursor: cursor + dim]
            component_buffers[comp_name].append(comp_slice.reshape(-1, dim))
            cursor += dim

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_tensor, return_type=None)

        for site, hook_name in site_hook_map_subset.items():
            acts = cache[hook_name][:, sample_positions, :]
            acts = acts.reshape(-1, acts.shape[-1]).detach().cpu()
            site_activation_buffers[site].append(acts)

        total_samples_collected += samples_added
        del cache

    component_arrays_flat: Dict[str, np.ndarray] = {}
    component_arrays_list: List[np.ndarray] = []
    for idx, comp_name in enumerate(component_order):
        dim = component_dims[idx]
        if component_buffers[comp_name]:
            comp_arr = np.concatenate(component_buffers[comp_name], axis=0)
        else:
            comp_arr = np.empty((0, dim), dtype=np.float32)
        component_arrays_flat[comp_name] = comp_arr
        component_arrays_list.append(comp_arr)

    if not component_arrays_list or component_arrays_list[0].shape[0] == 0:
        raise RuntimeError("No samples collected; adjust transformer batch size or total sample size.")

    available_samples = component_arrays_list[0].shape[0]
    desired_samples = min(available_samples, total_required)

    for comp_name in component_order:
        arr = component_arrays_flat[comp_name]
        if arr.shape[0] > desired_samples:
            component_arrays_flat[comp_name] = arr[:desired_samples]

    belief_concat, component_slices = flatten_component_beliefs(
        component_order,
        [component_arrays_flat[name] for name in component_order],
    )

    site_activation_tensors: Dict[str, torch.Tensor] = {}
    for site, buffers in site_activation_buffers.items():
        if not buffers:
            raise RuntimeError(f"No activations collected for site '{site}'.")
        acts_cat = torch.cat(buffers, dim=0)
        if acts_cat.shape[0] > desired_samples:
            acts_cat = acts_cat[:desired_samples]
        site_activation_tensors[site] = acts_cat

    sites = available_sites

    sae_specs: List[Tuple[str, float]] = []
    if args.top_k_vals:
        for k in args.top_k_vals:
            sae_specs.append(("top_k", float(k)))
    if args.lambda_vals:
        for lam in args.lambda_vals:
            sae_specs.append(("vanilla", float(lam)))

    rows: List[Dict[str, Any]] = []
    
    # Dictionary to collect L0 statistics for vanilla SAEs
    # Key: (site, sae_type, param_value), Value: list of L0 values
    l0_collections: Dict[Tuple[str, str, float], List[int]] = {}
    
    metric_collections: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    
    display_names = infer_component_display_names(component_order, component_meta_map)
    for site in sites:
        print(f"\nProcessing site '{site}'")
    
        acts_flat_cpu = site_activation_tensors[site]
        acts_np = acts_flat_cpu.numpy()
    
        if belief_concat.shape[0] != acts_np.shape[0]:
            raise ValueError(
                f"Belief samples ({belief_concat.shape[0]}) do not match activation samples ({acts_np.shape[0]})"
            )
    
        for sae_type, param_value in sae_specs:
            print(f"  SAE: {sae_type} param={param_value}")

            sae, decoder_dirs = load_sae_checkpoint(
                args.sae_folder,
                site,
                sae_type,
                param_value,
                device,
            )

            with torch.no_grad():
                feature_acts, _, _ = sae_encode_features(sae, acts_flat_cpu.to(device))
            feature_np = feature_acts.detach().cpu().numpy()
            activity_rates = (np.abs(feature_np) > args.activation_eps).mean(axis=0)

            # Collect L0 statistics for vanilla SAEs (top-k always has k non-zero latents)
            if sae_type == "vanilla":
                l0_per_sample = np.sum(np.abs(feature_np) > args.activation_eps, axis=1)
                l0_collections[(site, sae_type, param_value)] = l0_per_sample.tolist()

            residual_map, _ = fit_residual_to_belief_map(acts_np, belief_concat, alpha=1e-3)
            latent_sens = decoder_dirs @ residual_map  # (n_latents, total_belief_dim)

            n_latents = decoder_dirs.shape[0]
            total_belief_dim = latent_sens.shape[1]
    
            r2_per_dim = np.full((n_latents, total_belief_dim), np.nan, dtype=np.float64)
            lambda_values = [float(lam) for lam in args.lasso_lambdas]
            selection_freqs: Dict[float, np.ndarray] = {
                lam: np.zeros((n_latents, total_belief_dim), dtype=np.float64) for lam in lambda_values
            }
            selection_coefs: Dict[float, np.ndarray] = {
                lam: np.zeros((n_latents, total_belief_dim), dtype=np.float64) for lam in lambda_values
            }
    
            per_lambda_component_selection: Dict[float, Dict[str, np.ndarray]] = {}
            for lam in args.lasso_lambdas:
                component_selection = {}
                for comp_name in component_order:
                    sl_start, sl_end = component_slices[comp_name]
                    dims_to_use = component_dim_subset(display_names[comp_name], (sl_start, sl_end))
                    if not dims_to_use:
                        component_selection[comp_name] = np.zeros(n_latents, dtype=int)
                        continue
    
                    selected_mask = np.zeros(n_latents, dtype=bool)
                    for dim_idx in dims_to_use:
                        sens_vec = latent_sens[:, dim_idx]
                        design = feature_np * sens_vec[np.newaxis, :]
                        if not np.any(design):
                            continue
                        stats_payload: Dict[str, Any] = {}
                        try:
                            coef, _, _ = lasso_select_latents(
                                design,
                                belief_concat[:, dim_idx],
                                cv=args.lasso_cv,
                                max_iter=args.lasso_max_iter,
                                random_state=args.seed,
                                alpha=lam,
                                l1_ratio=args.elasticnet_l1_ratio,
                                stability_fraction=args.elasticnet_stability_fraction,
                                stability_runs=args.elasticnet_stability_runs,
                                stability_threshold=args.elasticnet_selection_threshold,
                                selection_stats=stats_payload,
                            )
                        except Exception as exc:
                            print(f"{site}: Lasso failed for component {comp_name} dim {dim_idx} (lambda={lam}): {exc}")
                            continue
                        freq_arr = stats_payload.get("frequency")
                        if freq_arr is not None:
                            selection_freqs[float(lam)][:, dim_idx] = freq_arr.astype(float)
                        coef_arr = stats_payload.get("avg_coef")
                        if coef_arr is not None:
                            selection_coefs[float(lam)][:, dim_idx] = coef_arr.astype(float)
                        selected_mask |= np.abs(coef) > 0.0
                    component_selection[comp_name] = selected_mask.astype(int)
                per_lambda_component_selection[lam] = component_selection
    
            for latent_idx in range(n_latents):
                row: Dict[str, Any] = {
                    "site": site,
                    "latent_index": latent_idx,
                    "sae_type": sae_type,
                    "top_k_k": float(param_value) if sae_type == "top_k" else math.nan,
                    "vanilla_lambda": float(param_value) if sae_type == "vanilla" else math.nan,
                }
                if latent_idx < len(activity_rates):
                    row["activation_rate"] = float(activity_rates[latent_idx])
                else:
                    row["activation_rate"] = float("nan")

                latent_activation = feature_np[:, latent_idx]

                for comp_name in component_order:
                    label = display_names[comp_name]
                    sl_start, sl_end = component_slices[comp_name]
                    dims_to_use = component_dim_subset(label, (sl_start, sl_end))
                    r2_values: List[float] = []

                    for dim_idx in dims_to_use:
                        sens_scalar = latent_sens[latent_idx, dim_idx]
                        predictor = latent_activation * sens_scalar
                        if not np.any(predictor):
                            continue
                        X = predictor.reshape(-1, 1)
                        y = belief_concat[:, dim_idx]
                        model = LinearRegression()
                        try:
                            model.fit(X, y)
                            r2 = model.score(X, y)
                        except Exception:
                            r2 = float("nan")
                        r2_values.append(r2)
                        r2_per_dim[latent_idx, dim_idx] = r2

                    if r2_values:
                        row[f"r2_{label}"] = float(np.nanmean(r2_values))
                    else:
                        row[f"r2_{label}"] = float("nan")

                for lam in args.lasso_lambdas:
                    selection_map = per_lambda_component_selection[lam]
                    for comp_name in component_order:
                        label = display_names[comp_name]
                        indicator = selection_map[comp_name][latent_idx]
                        row[f"lasso_{lam:g}_{label}"] = int(indicator)

                rows.append(row)

                metric_collections[(site, sae_type, float(param_value))] = {
                    "activation_rates": activity_rates.astype(np.float64),
                    "feature_matrix": feature_np.astype(np.float32, copy=False),
                    "latent_sens": latent_sens.astype(np.float64, copy=False),
                    "r2_per_dim": r2_per_dim.copy(),
                    "decoder_dirs": decoder_dirs.astype(np.float32, copy=False),
                    "belief_matrix": belief_concat.astype(np.float32, copy=False),
                    "selection_freqs": {lam: arr.copy() for lam, arr in selection_freqs.items()},
                    "selection_coefs": {lam: arr.copy() for lam, arr in selection_coefs.items()},
                    "component_order": list(component_order),
                    "component_slices": dict(component_slices),
                    "component_display_names": dict(display_names),
                    "lambda_values": lambda_values,
                    "activation_eps": args.activation_eps,
                }
    
    if not rows:
        raise RuntimeError("No data rows produced; check site selection and SAE parameters.")

    df = pd.DataFrame(rows)

    # Compute L0 statistics for vanilla SAEs
    l0_stats_rows: List[Dict[str, Any]] = []
    percentiles = [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9]

    for (site, sae_type, param_value), l0_values in l0_collections.items():
        if not l0_values:
            continue

        l0_array = np.array(l0_values)
        percentile_values = np.percentile(l0_array, percentiles)

        stats_row = {
            "site": site,
            "sae_type": sae_type,
            "vanilla_lambda": param_value,
            "mean": float(np.mean(l0_array)),
            "variance": float(np.var(l0_array)),
        }

        # Add percentile columns
        for p, val in zip(percentiles, percentile_values):
            col_name = f"p{p}".replace(".", "_")  # e.g., p0_1, p1, p5, etc.
            stats_row[col_name] = float(val)

        l0_stats_rows.append(stats_row)

    l0_stats_df = pd.DataFrame(l0_stats_rows) if l0_stats_rows else pd.DataFrame()

    return df, l0_stats_df, metric_collections


def _pairwise_matrix(values: np.ndarray, distance_fn) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if values.shape[0] < 2:
        return np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    return distance_fn(values)


def compute_latent_distance_matrices(
    metric_collections: Dict[Tuple[str, str, float], Dict[str, Any]],
    activation_threshold: float,
    activation_eps: float,
) -> Dict[Tuple[str, str, float], Dict[str, Any]]:
    results: Dict[Tuple[str, str, float], Dict[str, Any]] = {}

    for key, data in metric_collections.items():
        activation_rates = data["activation_rates"]
        mask = activation_rates >= activation_threshold
        indices = np.nonzero(mask)[0]

        entry: Dict[str, Any] = {"latent_indices": indices}

        if indices.size == 0:
            results[key] = entry
            continue

        # R² fingerprint distances (5D by component)
        r2_vectors = []
        for idx in indices:
            vec: List[float] = []
            for comp_name in data["component_order"]:
                start, end = data["component_slices"][comp_name]
                comp_vals = data["r2_per_dim"][idx, start:end]
                if comp_vals.size == 0:
                    vec.append(float("nan"))
                elif np.all(np.isnan(comp_vals)):
                    vec.append(0.0)
                else:
                    vec.append(float(np.nanmean(comp_vals)))
            r2_vectors.append(vec)
        r2_array = np.nan_to_num(np.asarray(r2_vectors, dtype=np.float64), nan=0.0)
        entry["r2_distances"] = {
            "cosine": _pairwise_matrix(r2_array, cosine_distances),
            "euclidean": _pairwise_matrix(r2_array, euclidean_distances),
        }

        # Sensitivity profile distances (full belief vector)
        sens_vectors = data["latent_sens"][indices]
        entry["sensitivity_distances"] = {
            "cosine": _pairwise_matrix(sens_vectors, cosine_distances),
            "euclidean": _pairwise_matrix(sens_vectors, euclidean_distances),
        }

        # Selection frequency vectors (mean across lambdas)
        if data["selection_freqs"]:
            freq_stack = np.stack([mat for mat in data["selection_freqs"].values()], axis=0)
            freq_mean = freq_stack.mean(axis=0)
            freq_vectors = freq_mean[indices]
            entry["selection_frequency_mean"] = {
                "cosine": _pairwise_matrix(freq_vectors, cosine_distances),
                "euclidean": _pairwise_matrix(freq_vectors, euclidean_distances),
            }
            per_lambda = {}
            for lam, mat in data["selection_freqs"].items():
                vectors = mat[indices]
                per_lambda[lam] = {
                    "cosine": _pairwise_matrix(vectors, cosine_distances),
                    "euclidean": _pairwise_matrix(vectors, euclidean_distances),
                }
            entry["selection_frequency_per_lambda"] = per_lambda

        # Selection coefficient vectors (mean across lambdas)
        if data["selection_coefs"]:
            coef_stack = np.stack([mat for mat in data["selection_coefs"].values()], axis=0)
            coef_mean = coef_stack.mean(axis=0)
            coef_vectors = coef_mean[indices]
            entry["selection_coefficient_mean"] = {
                "cosine": _pairwise_matrix(coef_vectors, cosine_distances),
                "euclidean": _pairwise_matrix(coef_vectors, euclidean_distances),
            }
            per_lambda_coef = {}
            for lam, mat in data["selection_coefs"].items():
                vectors = mat[indices]
                per_lambda_coef[lam] = {
                    "cosine": _pairwise_matrix(vectors, cosine_distances),
                    "euclidean": _pairwise_matrix(vectors, euclidean_distances),
                }
            entry["selection_coefficient_per_lambda"] = per_lambda_coef

        # Activation covariance / correlation with non-zero filtering
        feature_matrix = data["feature_matrix"][:, indices]
        num_latents = indices.size
        cov_matrix = np.zeros((num_latents, num_latents), dtype=np.float64)
        corr_matrix = np.zeros((num_latents, num_latents), dtype=np.float64)

        for i in range(num_latents):
            xi = feature_matrix[:, i]
            for j in range(i, num_latents):
                xj = feature_matrix[:, j]
                active_mask = (np.abs(xi) > activation_eps) | (np.abs(xj) > activation_eps)
                if not np.any(active_mask):
                    cov = 0.0
                    corr = 0.0
                else:
                    xi_sel = xi[active_mask]
                    xj_sel = xj[active_mask]
                    if xi_sel.size < 2 or xj_sel.size < 2:
                        cov = 0.0
                        corr = 0.0
                    else:
                        cov = float(np.cov(xi_sel, xj_sel, ddof=0)[0, 1])
                        std_i = float(np.std(xi_sel, ddof=0))
                        std_j = float(np.std(xj_sel, ddof=0))
                        if std_i == 0.0 or std_j == 0.0:
                            corr = 0.0
                        else:
                            corr = cov / (std_i * std_j)
                cov_matrix[i, j] = cov_matrix[j, i] = cov
                corr_matrix[i, j] = corr_matrix[j, i] = corr

        entry["activation_covariance"] = cov_matrix
        entry["activation_correlation"] = corr_matrix

        results[key] = entry

    return results



def save_bar_plot(series: pd.Series, max_count: int, title: str, path: str) -> None:
    values = list(range(max_count + 1))
    counts = series.value_counts().reindex(values, fill_value=0).sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color="#4C72B0")
    plt.xticks(values)
    plt.xlabel("Count")
    plt.ylabel("Latent count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def augment_analysis(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()
    plots_dir = args.output_path

    # R² indicator sums
    r2_columns = sorted([col for col in df.columns if col.startswith("r2_") and not col.endswith("_indicator")])
    r2_indicator_cols: List[str] = []
    if r2_columns:
        for col in r2_columns:
            indicator_col = f"{col}_indicator"
            df[indicator_col] = df[col].ge(args.r2_threshold).astype(int)
            r2_indicator_cols.append(indicator_col)

        r2_sum_col = "r2_indicator_sum"
        df[r2_sum_col] = df[r2_indicator_cols].sum(axis=1)
        save_bar_plot(
            df[r2_sum_col],
            max_count=len(r2_columns),
            title=f"R² matches (threshold={args.r2_threshold})",
            path=os.path.join(plots_dir, f"{r2_sum_col}_hist.png"),
        )

    # Lasso indicator sums per lambda
    lasso_columns = [col for col in df.columns if col.startswith("lasso_")]
    lambda_to_cols: Dict[str, List[str]] = defaultdict(list)
    for col in lasso_columns:
        parts = col.split("_", 2)
        if len(parts) < 3:
            continue
        lam_str = parts[1]
        lambda_to_cols[lam_str].append(col)

    for lam_str, cols in sorted(lambda_to_cols.items(), key=lambda item: float(item[0])):
        sum_col = f"lasso_hits_{lam_str}"
        df[sum_col] = df[cols].sum(axis=1)
        safe_lam = lam_str.replace(".", "p")
        save_bar_plot(
            df[sum_col],
            max_count=len(cols),
            title=f"Lasso selections (lambda={lam_str})",
            path=os.path.join(plots_dir, f"{sum_col}_{safe_lam}.png"),
        )

    return df


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.output_path or ".", exist_ok=True)
    if args.input_csv is None:
        df, l0_stats_df, metric_store = run_analysis(args)
    else:
        df = load_metrics_table(args.input_csv)
        l0_stats_df = pd.DataFrame()  # Empty if loading from existing CSV
        metric_store = {}

    output_format = determine_output_format(args.output_path, args.format)
    if output_format == "csv":
        df.to_csv(os.path.join(args.output_path, "ground_truth_metrics.csv"), index=False)
    else:
        df.to_parquet(os.path.join(args.output_path, "ground_truth_metrics.parquet"), index=False)
    print(f"Saved latent metrics to {args.output_path}")

    # Save L0 statistics if available
    if not l0_stats_df.empty:
        if output_format == "csv":
            l0_stats_df.to_csv(os.path.join(args.output_path, "l0_statistics.csv"), index=False)
        else:
            l0_stats_df.to_parquet(os.path.join(args.output_path, "l0_statistics.parquet"), index=False)
        print(f"Saved L0 statistics to {args.output_path}")

    if metric_store:
        distance_metrics = compute_latent_distance_matrices(
            metric_store,
            activation_threshold=args.act_rate_threshold,
            activation_eps=args.activation_eps,
        )
        metrics_path = os.path.join(args.output_path, "latent_distance_matrices.pkl")
        with open(metrics_path, "wb") as fh:
            pickle.dump(distance_metrics, fh)
        print(f"Saved latent distance matrices to {metrics_path}")


if __name__ == "__main__":
    main()
