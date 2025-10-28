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
import re

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
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.image import AxesImage
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


WHITE_ROSE_CMAP = LinearSegmentedColormap.from_list(
    "white_rose",
    [
        (0.0, (1.0, 1.0, 1.0, 1.0)),
        (0.35, (0.992, 0.925, 0.937, 1.0)),
        (0.7, (0.957, 0.588, 0.651, 1.0)),
        (1.0, (0.647, 0.0, 0.149, 1.0)),
    ],
)

WHITE_ROSE_NEG_CMAP = LinearSegmentedColormap.from_list(
    "white_rose_neg",
    [
        (0.0, (0.133, 0.0, 0.251, 1.0)),
        (0.4, (0.620, 0.447, 0.659, 1.0)),
        (1.0, (1.0, 1.0, 1.0, 1.0)),
    ],
)



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
    parser.add_argument("--input_csv", type=str, default="outputs/reports/multipartite_003e/ground_truth_diagnostics/ground_truth_metrics.csv", help="Optional existing CSV/Parquet to load instead of recomputing metrics")
    #parser.add_argument("--input_csv", type=str, default=None, help="Optional existing CSV/Parquet to load instead of recomputing metrics")
    #arser.add_argument("--latent_distance_pkl", type=str, default="outputs/reports/multipartite_003e/ground_truth_diagnostics/latent_distance_matrices.pkl", help="Optional existing latent distance matrices pickle file")
    parser.add_argument("--latent_distance_pkl", type=str, default=None, help="Optional existing latent distance matrices pickle file")
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
    parser.add_argument("--top_k_vals", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16, 19, 22], help="List of K values for top-k SAEs")
    parser.add_argument("--lambda_vals", type=float, nargs="+", default=[0.005, 0.02, 0.04, 0.07, 0.1, 0.125, 0.15], help="List of lambda values for vanilla SAEs")
    parser.add_argument("--sites", type=str, nargs="+",  default=["layer_0", "layer_1", "layer_2"], help="Subset of site names to analyze")

    # Sampling
    parser.add_argument("--transformer_batch_size", type=int, default=2048, help="Number of sequences per forward pass when sampling activations")
    parser.add_argument("--transformer_total_sample_size", type=int, default=25000, help="Target number of token positions to collect for regression and activation statistics")
    parser.add_argument("--sample_seq_len", type=int, default=None, help="Sequence length for sampling (default: model n_ctx)")
    parser.add_argument("--activation_eps", type=float, default=1e-6, help="Threshold when counting latent activations")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--act_rate_threshold", type=float, default=0.01, help="Minimum activation rate to include a latent in distance metrics")

    # Elastic net / stability selection sweep
    parser.add_argument("--lasso_lambdas", type=float, nargs="+", default=[0.002, 0.004, 0.006, 0.008, 0.01],  help="Elastic net alpha values to test")
    parser.add_argument("--lasso_cv", type=int, default=8, help="Legacy fallback CV folds when alpha is None")
    parser.add_argument("--lasso_max_iter", type=int, default=10000, help="Max iterations for ElasticNet solver")
    parser.add_argument("--elasticnet_l1_ratio", type=float, default=0.9, help="Elastic net l1_ratio (fraction of L1 penalty)")
    parser.add_argument("--elasticnet_stability_fraction", type=float, default=0.5, help="Fraction of samples per stability-selection resample")
    parser.add_argument("--elasticnet_stability_runs", type=int, default=25, help="Number of resampled fits for stability selection")
    parser.add_argument("--elasticnet_selection_threshold", type=float, default=0.5, help="Minimum selection frequency to keep a latent")
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


def _safe_label(label: str) -> str:
    label = label.replace("λ", "lambda")
    label = re.sub(r"[^0-9A-Za-z]+", "_", label)
    return label.strip("_") or "config"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _matrix_df_from_array(matrix: Optional[np.ndarray], indices: Optional[np.ndarray]) -> Optional[pd.DataFrame]:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    if arr.shape[0] != arr.shape[1]:
        return None
    if indices is not None:
        idx_arr = np.asarray(indices)
        if idx_arr.ndim == 1 and idx_arr.size == arr.shape[0]:
            labels = [int(x) for x in idx_arr.tolist()]
            return pd.DataFrame(arr, index=labels, columns=labels)
    return pd.DataFrame(arr)


def _prepare_heatmap_data(
    matrix_df: pd.DataFrame,
    max_dim: Optional[int],
    *,
    bin_count: Optional[int] = 50,
    sort_rows: bool = True,
) -> Optional[Tuple[np.ndarray, List[Any], List[Any]]]:
    if matrix_df.empty:
        return None
    matrix = matrix_df.to_numpy(dtype=float)
    if matrix.size == 0 or np.all(np.isnan(matrix)):
        return None

    row_labels = list(matrix_df.index)
    col_labels = list(matrix_df.columns)

    if max_dim and (matrix.shape[0] > max_dim or matrix.shape[1] > max_dim):
        row_idx = np.linspace(0, matrix.shape[0] - 1, min(max_dim, matrix.shape[0]), dtype=int)
        col_idx = np.linspace(0, matrix.shape[1] - 1, min(max_dim, matrix.shape[1]), dtype=int)
        matrix = matrix[np.ix_(row_idx, col_idx)]
        row_labels = [row_labels[idx] for idx in row_idx]
        col_labels = [col_labels[idx] for idx in col_idx]
        print(f"Downsampling heatmap for shape {matrix_df.shape} to {matrix.shape} before plotting.")

    matrix = np.nan_to_num(matrix, nan=0.0).astype(np.float32, copy=False)

    if bin_count and matrix.size > 0:
        data_min = float(np.nanmin(matrix))
        data_max = float(np.nanmax(matrix))
        if math.isfinite(data_min) and math.isfinite(data_max) and data_max > data_min:
            start = 0.0 if data_min >= 0.0 else data_min
            end = 0.0 if data_max <= 0.0 else data_max
            if end == start:
                end = start + 1.0
            bin_edges = np.linspace(start, end, bin_count + 1, dtype=np.float32)
            indices = np.digitize(matrix, bin_edges, right=False)
            indices = np.clip(indices, 1, bin_count) - 1
            matrix = bin_edges[indices]

    if sort_rows and matrix.shape[0] > 1:
        row_order = sorted(range(matrix.shape[0]), key=lambda idx: tuple(matrix[idx]))
        matrix = matrix[row_order, :]
        row_labels = [row_labels[idx] for idx in row_order]
        if matrix.shape[0] == matrix.shape[1]:
            matrix = matrix[:, row_order]
            col_labels = [col_labels[idx] for idx in row_order]

    return matrix, row_labels, col_labels


def _build_heatmap_cmap_and_norm(
    matrix: np.ndarray,
    *,
    cmap: Optional[str],
    center: Optional[float],
) -> Tuple[LinearSegmentedColormap, Normalize]:
    data_min = float(np.nanmin(matrix))
    data_max = float(np.nanmax(matrix))

    has_neg = data_min < 0.0
    has_pos = data_max > 0.0

    if data_min == 0.0 and data_max == 0.0:
        base_name = cmap or "YlOrRd"
        white_only = LinearSegmentedColormap.from_list(f"{base_name}_all_white", [(0.0, (1.0, 1.0, 1.0, 1.0)), (1.0, (1.0, 1.0, 1.0, 1.0))])
        norm = Normalize(vmin=0.0, vmax=1.0)
        return white_only, norm

    if center is not None:
        center_value = center
    elif has_neg and has_pos:
        center_value = 0.0
    else:
        center_value = None

    if center_value is not None:
        base_name = cmap or "coolwarm"
        base_cmap = plt.get_cmap(base_name)
        white = (1.0, 1.0, 1.0, 1.0)
        span_low = abs(center_value - data_min)
        span_high = abs(data_max - center_value)
        if span_low == 0.0 and span_high == 0.0:
            span_low = span_high = 1.0
        vmin = center_value - span_low
        vmax = center_value + span_high
        colors = [
            (0.0, base_cmap(0.0)),
            (0.5, white),
            (1.0, base_cmap(1.0)),
        ]
        new_cmap = LinearSegmentedColormap.from_list(f"{base_name}_white_center", colors)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
    elif has_pos:
        vmax = data_max if data_max > 0.0 else 1.0
        if cmap is None:
            new_cmap = WHITE_ROSE_CMAP
        else:
            new_cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=0.0, vmax=vmax)
    else:
        vmin = data_min if data_min < 0.0 else -1.0
        if cmap is None:
            new_cmap = WHITE_ROSE_NEG_CMAP
        else:
            new_cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=0.0)

    return new_cmap, norm


def render_matrix_heatmap(
    ax: plt.Axes,
    matrix_df: pd.DataFrame,
    title: str,
    *,
    center: Optional[float] = None,
    cmap: Optional[str] = None,
    max_dim: Optional[int] = None,
    precomputed: Optional[Tuple[np.ndarray, List[Any], List[Any]]] = None,
) -> Optional[AxesImage]:
    prepared = precomputed or _prepare_heatmap_data(matrix_df, max_dim)
    if prepared is None:
        ax.axis("off")
        ax.set_title(f"{title}\nNo Data", fontsize=10)
        return None

    matrix, row_labels, col_labels = prepared
    colormap, norm = _build_heatmap_cmap_and_norm(matrix, cmap=cmap, center=center)

    im = ax.imshow(matrix, cmap=colormap, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=9)

    if len(col_labels) <= 60:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticks([])

    if len(row_labels) <= 60:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
    else:
        ax.set_yticks([])

    return im


def save_matrix_heatmap(
    matrix_df: pd.DataFrame,
    title: str,
    path: str,
    *,
    center: Optional[float] = None,
    cmap: Optional[str] = None,
    max_dim: Optional[int] = None,
) -> None:
    fig_size_info = _prepare_heatmap_data(matrix_df, max_dim)
    if fig_size_info is None:
        return
    matrix, row_labels, col_labels = fig_size_info

    width_scale = min(12.0, max(4.0, len(col_labels) / 20.0))
    height_scale = min(12.0, max(4.0, len(row_labels) / 20.0))

    fig, ax = plt.subplots(figsize=(width_scale, height_scale))
    trimmed_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    im = render_matrix_heatmap(
        ax,
        trimmed_df,
        title,
        center=center,
        cmap=cmap,
        max_dim=None,
        precomputed=(matrix, row_labels, col_labels),
    )
    if im is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################

#%%

args = parse_args()
print(args.output_path)
os.makedirs(args.output_path, exist_ok=True)
print(f"{os.path.exists(args.output_path)=}")

if args.latent_distance_pkl is not None and os.path.exists(args.latent_distance_pkl):
    with open(args.latent_distance_pkl, "rb") as fh:
        latent_distance_matrices = pickle.load(fh)
else:
    latent_distance_matrices = None

df = load_metrics_table(args.input_csv)
df = augment_analysis(df, args)

plots_dir = args.output_path
heatmap_dirs = {
    "r2_heatmaps": ensure_dir(os.path.join(plots_dir, "r2_heatmaps")),
    "latent_cov": ensure_dir(os.path.join(plots_dir, "latent_cov")),
    "latent_corr": ensure_dir(os.path.join(plots_dir, "latent_corr")),
    "component_cov": ensure_dir(os.path.join(plots_dir, "component_cov")),
    "component_corr": ensure_dir(os.path.join(plots_dir, "component_corr")),
    "component_distance_grid": ensure_dir(os.path.join(plots_dir, "component_distance_grid")),
    "latent_distance": ensure_dir(os.path.join(plots_dir, "latent_distance")),
}

def _sort_key(cfg: Dict[str, Any]) -> tuple:
    top = cfg.get("top_k_k")
    lam = cfg.get("vanilla_lambda")
    top_val = float("inf") if pd.isna(top) else float(top)
    lam_val = float("inf") if pd.isna(lam) else float(lam)
    return (cfg.get("sae_type", ""), top_val, lam_val)

sae_configs = df[["sae_type", "top_k_k", "vanilla_lambda"]].drop_duplicates().to_dict("records")
sae_configs.sort(key=_sort_key)
num_configs = len(sae_configs)
if num_configs == 0:
    raise RuntimeError("No SAE configurations found in the metrics table.")


#%%


#%%

# Filter by activation rate threshold
if "activation_rate" in df.columns:
    initial_count = len(df)
    df = df[df["activation_rate"] >= args.act_rate_threshold].copy()
    filtered_count = len(df)
    print(f"Filtered dataframe from {initial_count} to {filtered_count} rows using activation_rate >= {args.act_rate_threshold}")
else:
    print("Warning: 'activation_rate' column not found in dataframe, skipping filtering")

for col in df.columns:
    print(col)

#%%

r2_cols = [
    "r2_tom_quantum",
    "r2_tom_quantum_1",
    "r2_mess3",
    "r2_mess3_1",
    "r2_mess3_2",
]


def _config_mask(frame: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    mask = frame["sae_type"] == cfg.get("sae_type")
    top_val = cfg.get("top_k_k")
    lam_val = cfg.get("vanilla_lambda")
    if pd.isna(top_val):
        mask &= frame["top_k_k"].isna()
    else:
        mask &= frame["top_k_k"] == top_val
    if pd.isna(lam_val):
        mask &= frame["vanilla_lambda"].isna()
    else:
        mask &= frame["vanilla_lambda"] == lam_val
    return mask


def _format_config_label(cfg: Dict[str, Any]) -> str:
    sae_type = cfg.get("sae_type", "unknown")
    top = cfg.get("top_k_k")
    lam = cfg.get("vanilla_lambda")
    if sae_type == "top_k" and not pd.isna(top):
        return f"top_k k={int(top)}"
    if sae_type == "vanilla" and not pd.isna(lam):
        return f"vanilla λ={lam:.4g}"
    return sae_type


hist_cols = r2_cols
num_hist_cols = len(hist_cols)

fig, axs = plt.subplots(
    num_configs,
    num_hist_cols,
    figsize=(num_hist_cols * 3.0, max(2, num_configs) * 2.2),
    squeeze=False,
)

for row_idx, cfg in enumerate(sae_configs):
    subset = df[_config_mask(df, cfg)]
    label = _format_config_label(cfg)
    for col_idx, col in enumerate(hist_cols):
        ax = axs[row_idx, col_idx]
        if col in subset.columns and not subset[col].dropna().empty:
            ax.hist(subset[col].dropna(), bins=30, alpha=0.7)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{col}\nNo Data", ha="center", va="center", fontsize=10)
            continue

        if row_idx == 0:
            ax.set_title(f"{col}")
        else:
            ax.set_title("")

        if col_idx == 0:
            ax.set_ylabel(label)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if row_idx == num_configs - 1:
            ax.set_xlabel("R²")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

plt.tight_layout()
hist_path = os.path.join(args.output_path, "r2_histograms.png")
plt.savefig(hist_path)
plt.show()
plt.close()
print(f"Saved R^2 histograms to {hist_path}")

#%%
# Plot ECDF (Empirical Cumulative Distribution Function) for each R^2 column

fig_ecdf, axs_ecdf = plt.subplots(
    num_configs,
    num_hist_cols,
    figsize=(num_hist_cols * 3.0, max(2, num_configs) * 2.2),
    squeeze=False,
)

for row_idx, cfg in enumerate(sae_configs):
    subset = df[_config_mask(df, cfg)]
    label = _format_config_label(cfg)
    for col_idx, col in enumerate(hist_cols):
        ax = axs_ecdf[row_idx, col_idx]
        if col in subset.columns:
            values = subset[col].dropna().values
            if len(values) > 0:
                sorted_vals = np.sort(values)
                yvals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                ax.plot(sorted_vals, yvals, drawstyle="steps-post")
                ax.set_ylim([0, 1.01])
                ax.set_xlim([np.min(sorted_vals), np.max(sorted_vals)])
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{col}\nNo Data", ha="center", va="center", fontsize=10)
                continue
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{col}\nNot Found", ha="center", va="center", fontsize=10)
            continue

        if row_idx == 0:
            ax.set_title(f"{col}")
        else:
            ax.set_title("")

        if col_idx == 0:
            ax.set_ylabel(label)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if row_idx == num_configs - 1:
            ax.set_xlabel("R²")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

plt.tight_layout()
ecdf_path = os.path.join(args.output_path, "r2_ecdf_plots.png")
plt.savefig(ecdf_path)
plt.show()
plt.close()
print(f"Saved R^2 ECDF plots to {ecdf_path}")



#%%

percentiles = [1, 5] + list(range(0, 101, 10)) + [95, 99]
percentiles = sorted(set(percentiles))

col_names = []
all_quantiles = []

for col in r2_cols:
    if col in df.columns:
        values = df[col].dropna()
        quantiles = np.percentile(values, percentiles)
        all_quantiles.append(quantiles)
        col_names.append(col[:12])
    else:
        all_quantiles.append([np.nan] * len(percentiles))
        col_names.append(col[:12])

# Create DataFrame: rows=percentiles, columns=short col names (variable names)
percentile_df = pd.DataFrame(
    data=np.stack(all_quantiles, axis=-1),
    index=[f"{p}%" for p in percentiles],
    columns=col_names
)
print("\nPercentiles for R^2 columns (variables as columns, percentiles as rows):")
print(percentile_df.round(5))


#%%

import re

# Find all columns that match the lasso_<decimal> pattern and extract unique decimals
lasso_pattern = re.compile(r"lasso_([.0-9]+)")
lasso_decimals = set()
for col in df.columns:
    m = lasso_pattern.match(col)
    if m:
        lasso_decimals.add(m.group(1))

# Sort decimals numerically
sorted_decimals = sorted(lasso_decimals, key=lambda x: float(x))

# hit_cols is always ["r2_indicator_sum"] plus lasso_hits_<decimal> for each unique decimal found
hit_cols = ["r2_indicator_sum"] + [f"lasso_hits_{d}" for d in sorted_decimals]

num_hit_cols = len(hit_cols)

fig, axs = plt.subplots(
    num_configs,
    num_hit_cols,
    figsize=(num_hit_cols * 2.8, max(2, num_configs) * 2.2),
    squeeze=False,
)

for row_idx, cfg in enumerate(sae_configs):
    subset = df[_config_mask(df, cfg)]
    label = _format_config_label(cfg)
    for col_idx, col in enumerate(hit_cols):
        ax = axs[row_idx, col_idx]
        if col in subset.columns and not subset[col].dropna().empty:
            vals = subset[col].dropna().astype(int)
            value_counts = np.bincount(vals, minlength=6)
            ax.bar(range(len(value_counts)), value_counts)
            ax.set_xticks(range(len(value_counts)))
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{col}\nNo Data", ha="center", va="center", fontsize=10)
            continue

        if row_idx == 0:
            ax.set_title(col)
        else:
            ax.set_title("")

        if col_idx == 0:
            ax.set_ylabel(label)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if row_idx == num_configs - 1:
            ax.set_xlabel("Count")
            ax.set_xticklabels([str(i) for i in range(len(value_counts))])
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

plt.tight_layout()
barplot_path = os.path.join(args.output_path, "hit_counts_barchart_grid.png")
plt.savefig(barplot_path)
plt.show()
plt.close()
print(f"Saved barchart grid for hit counts to {barplot_path}")

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

activation_rates = df["activation_rate"].dropna()
# Compute the histogram data just once so both plots have the same binning.
counts, bins, patches = axs[0].hist(
    activation_rates, bins=30, color="#DD8452", edgecolor="black", density=True
)
# Left plot: full histogram
axs[0].set_xlabel("Activation Rate")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Histogram of Latent Activation Rate")

# Right plot: histogram with way more bins, but the same xlim restriction
num_fine_bins = 500  # "way way more"
axs[1].hist(
    activation_rates, bins=num_fine_bins, color="#DD8452", edgecolor="black", density=True
)
axs[1].set_xlabel("Activation Rate")
axs[1].set_ylabel("Frequency")
axs[1].set_title("Histogram (xlim=0-0.05, bins=500)")
axs[1].set_xlim(0, 0.05)

hist_path = os.path.join(args.output_path, "activation_rate_histogram.png")
plt.tight_layout()
plt.savefig(hist_path, dpi=150)
plt.show()
plt.close()
print(f"Saved histogram of activation rate to {hist_path}")



percentiles = [1, 5] + list(range(10, 100, 10)) + [95, 99]

def get_config_colname(cfg: dict) -> str:
    """Produce a short, informative column name for the config."""
    sae_type = cfg.get("sae_type", "unknown")
    if sae_type == "top_k":
        return f"top_k_{int(cfg.get('top_k_k', -1))}"
    elif sae_type == "vanilla":
        lam = cfg.get("vanilla_lambda", None)
        if pd.isna(lam):
            lam = 0.0
        # Remove trailing zeros and decimal if possible for prettiness
        lam_str = f"{lam:.4g}".rstrip("0").rstrip(".")
        return f"vanilla_l{lam_str}"
    else:
        return sae_type

# List of column names in order corresponding to sae_configs
config_colnames = [get_config_colname(cfg) for cfg in sae_configs]

# Prepare a dictionary where keys are percentiles and values are dicts of {colname: value}
percentile_table = {p: {} for p in percentiles}

for config, colname in zip(sae_configs, config_colnames):
    mask = _config_mask(df, config)
    rates = df.loc[mask, "activation_rate"].dropna()
    if len(rates) == 0:
        percentile_values = [np.nan] * len(percentiles)
    else:
        percentile_values = np.percentile(rates, percentiles)
    for p, v in zip(percentiles, percentile_values):
        percentile_table[p][colname] = v

# Convert to DataFrame: rows are percentiles, columns are configs (named as above)
activation_percentiles_df = pd.DataFrame.from_dict(percentile_table, orient="index")
activation_percentiles_df.index.name = "Percentile"
activation_percentiles_df = activation_percentiles_df[config_colnames]  # preserve order

print("Activation rate percentiles by SAE config (table, percentiles as rows):")
print(activation_percentiles_df)

csv_path = os.path.join(args.output_path, "activation_rate_percentiles_per_config.csv")
activation_percentiles_df.to_csv(csv_path)
print(f"Saved activation rate percentiles by SAE config to {csv_path}")

#%%
print(sae_configs)

#%%
# Heatmaps of R² values organized by site

R2_NOISE_THRESHOLD = 0.005

# Get all R² columns (raw values, not indicators or sums)
r2_value_cols = sorted([col for col in df.columns if col.startswith("r2_") and not col.endswith("_indicator") and col != "r2_indicator_sum"])

if r2_value_cols and "site" in df.columns:
    sites = sorted(df["site"].unique())

    for site in sites:
        safe_site = _safe_label(site)
        site_df = df[df["site"] == site]

        site_configs: List[Dict[str, Any]] = []
        for cfg in sae_configs:
            if len(site_df[_config_mask(site_df, cfg)]) > 0:
                site_configs.append(cfg)

        if not site_configs:
            print(f"No data for site {site}, skipping")
            continue

        num_site_configs = len(site_configs)
        comp_cov_dfs: List[pd.DataFrame] = []
        comp_corr_dfs: List[pd.DataFrame] = []
        comp_labels: List[str] = []

        fig, axs = plt.subplots(
            1,
            num_site_configs,
            figsize=(num_site_configs * 3.5, 12),
            squeeze=False,
        )

        for col_idx, cfg in enumerate(site_configs):
            ax = axs[0, col_idx]
            subset = site_df[_config_mask(site_df, cfg)]
            label = _format_config_label(cfg)

            if len(subset) == 0:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{label}\nNo Data", ha="center", va="center", fontsize=10)
                continue

            subset_sorted = subset.sort_values("latent_index")
            heatmap_data = subset_sorted[r2_value_cols].values

            r2_values_df = subset_sorted[r2_value_cols].astype(float).copy()
            component_labels = [col.replace("r2_", "") for col in r2_value_cols]
            r2_values_df.columns = component_labels
            r2_values_df.index = subset_sorted["latent_index"].astype(int)

            safe_label = _safe_label(f"{site}_{label}")

            component_cov_df = r2_values_df.cov(min_periods=1)
            component_corr_df = r2_values_df.corr(min_periods=1)
            if not component_cov_df.empty:
                comp_cov_dfs.append(component_cov_df)
                comp_corr_dfs.append(component_corr_df)
                comp_labels.append(label)
                comp_cov_path = os.path.join(heatmap_dirs["component_cov"], f"{safe_label}.png")
                comp_corr_path = os.path.join(heatmap_dirs["component_corr"], f"{safe_label}.png")
                save_matrix_heatmap(component_cov_df, f"Component Covariance\n{site} / {label}", comp_cov_path, center=0.0, cmap="YlGnBu")
                save_matrix_heatmap(component_corr_df, f"Component Correlation\n{site} / {label}", comp_corr_path, center=0.0)

            if r2_values_df.shape[0] >= 2:
                latent_values_df = r2_values_df.T
                latent_cov_df = latent_values_df.cov(min_periods=1)
                latent_corr_df = latent_values_df.corr(min_periods=1)
                latent_cov_path = os.path.join(heatmap_dirs["latent_cov"], f"{safe_label}.png")
                latent_corr_path = os.path.join(heatmap_dirs["latent_corr"], f"{safe_label}.png")
                save_matrix_heatmap(latent_cov_df, f"Latent Covariance\n{site} / {label}", latent_cov_path, center=0.0, cmap="YlGnBu")
                save_matrix_heatmap(latent_corr_df, f"Latent Correlation\n{site} / {label}", latent_corr_path, center=0.0)

            heatmap_df = pd.DataFrame(heatmap_data, columns=component_labels, index=r2_values_df.index)
            prepared = _prepare_heatmap_data(heatmap_df, max_dim=None)
            if prepared is None:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{label}\nNo Data", ha="center", va="center", fontsize=10)
                continue
            im = render_matrix_heatmap(ax, heatmap_df, label, precomputed=prepared)

            matrix_proc, _, _ = prepared
            if matrix_proc.shape[0] >= 1:
                max_vals = np.max(matrix_proc, axis=1)
                noise_rows = np.nonzero(max_vals < R2_NOISE_THRESHOLD)[0]
                for row_idx in noise_rows:
                    ax.axhspan(row_idx - 0.5, row_idx + 0.5, facecolor="black", alpha=0.06, linewidth=0)
            if matrix_proc.shape[0] >= 2:
                argmax_cols = np.argmax(matrix_proc, axis=1)
                change_rows = np.nonzero(np.diff(argmax_cols))[0]
                for row_idx in change_rows:
                    ax.axhline(row_idx + 0.5, color="black", linewidth=0.4, alpha=0.7)
            if im is not None and col_idx == num_site_configs - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="R²")

            if col_idx == 0:
                ax.set_ylabel("Latent index")

        plt.suptitle(f"R² Heatmaps for {site}", fontsize=14, y=0.995)
        plt.tight_layout()
        heatmap_path = os.path.join(heatmap_dirs["r2_heatmaps"], f"{safe_site}.png")
        plt.savefig(heatmap_path, dpi=150)
        plt.show()
        plt.close()
        print(f"Saved R² value heatmaps for {site} to {heatmap_path}")

        if comp_cov_dfs:
            fig_cc, axes_cc = plt.subplots(
                2,
                len(comp_cov_dfs),
                figsize=(max(4.0, len(comp_cov_dfs) * 3.2), 6.5),
                squeeze=False,
            )
            for idx, label in enumerate(comp_labels):
                ax_cov = axes_cc[0, idx]
                im_cov = render_matrix_heatmap(ax_cov, comp_cov_dfs[idx], f"{label}\nCovariance", center=0.0, cmap="YlGnBu")
                if im_cov is not None:
                    fig_cc.colorbar(im_cov, ax=ax_cov, fraction=0.046, pad=0.04)
                if idx == 0:
                    ax_cov.set_ylabel("Covariance", fontsize=9)

                ax_corr = axes_cc[1, idx]
                im_corr = render_matrix_heatmap(ax_corr, comp_corr_dfs[idx], f"{label}\nCorrelation", center=0.0)
                if im_corr is not None:
                    fig_cc.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04)
                if idx == 0:
                    ax_corr.set_ylabel("Correlation", fontsize=9)

            fig_cc.suptitle(f"Component Distances • {site}", fontsize=14)
            fig_cc.tight_layout(rect=[0, 0, 1, 0.95])
            grid_path = os.path.join(heatmap_dirs["component_distance_grid"], f"{safe_site}.png")
            fig_cc.savefig(grid_path, dpi=150)
            plt.close(fig_cc)
            print(f"Saved component distance grid for {site} to {grid_path}")
else:
    print("Warning: No R² columns or site column found, skipping heatmap generation")

#%%

if latent_distance_matrices:
    latent_distance_base = heatmap_dirs["latent_distance"]

    for key, payload in latent_distance_matrices.items():
        if not payload:
            continue

        site, sae_type, param = key
        if sae_type == "top_k":
            cfg_dict = {"sae_type": "top_k", "top_k_k": param, "vanilla_lambda": math.nan}
        elif sae_type == "vanilla":
            cfg_dict = {"sae_type": "vanilla", "top_k_k": math.nan, "vanilla_lambda": param}
        else:
            cfg_dict = {"sae_type": sae_type, "top_k_k": math.nan, "vanilla_lambda": param}

        config_label = _format_config_label(cfg_dict)
        site_label = f"{site} / {config_label}"
        safe_config = _safe_label(f"{site}_{config_label}")
        config_dir = ensure_dir(os.path.join(latent_distance_base, safe_config))

        indices = payload.get("latent_indices")
        if isinstance(indices, (list, tuple)):
            idx_array = np.asarray(indices)
        else:
            idx_array = indices if isinstance(indices, np.ndarray) else None

        def plot_metric_dict(relative_dir: str, metric_dict: Dict[Any, Any], *, display_label: str, center: Optional[float] = None, cmap: Optional[str] = None) -> None:
            if not metric_dict:
                return
            target_dir = ensure_dir(os.path.join(config_dir, relative_dir))
            for metric_name, matrix in metric_dict.items():
                df = _matrix_df_from_array(matrix, idx_array)
                if df is None or df.empty:
                    continue
                metric_str = str(metric_name)
                filename = f"{_safe_label(metric_str)}.png"
                title = f"{display_label} ({metric_str})\n{site_label}"
                save_matrix_heatmap(df, title, os.path.join(target_dir, filename), center=center, cmap=cmap)

        plot_metric_dict(
            "r2_distances",
            payload.get("r2_distances", {}),
            display_label="R² Distance",
        )
        plot_metric_dict(
            "sensitivity_distances",
            payload.get("sensitivity_distances", {}),
            display_label="Sensitivity Distance",
        )
        plot_metric_dict(
            "selection_frequency_mean",
            payload.get("selection_frequency_mean", {}),
            display_label="Selection Frequency Distance (Mean)",
        )
        plot_metric_dict(
            "selection_coefficient_mean",
            payload.get("selection_coefficient_mean", {}),
            display_label="Selection Coefficient Distance (Mean)",
        )

        freq_per_lambda = payload.get("selection_frequency_per_lambda", {})
        for lam, dist_dict in sorted(freq_per_lambda.items(), key=lambda item: float(item[0]) if isinstance(item[0], (int, float)) else item[0]):
            lam_str = str(lam)
            safe_lam = _safe_label(f"lambda_{lam_str}")
            plot_metric_dict(
                os.path.join("selection_frequency_per_lambda", safe_lam),
                dist_dict,
                display_label=f"Selection Frequency Distance λ={lam_str}",
            )

        coef_per_lambda = payload.get("selection_coefficient_per_lambda", {})
        for lam, dist_dict in sorted(coef_per_lambda.items(), key=lambda item: float(item[0]) if isinstance(item[0], (int, float)) else item[0]):
            lam_str = str(lam)
            safe_lam = _safe_label(f"lambda_{lam_str}")
            plot_metric_dict(
                os.path.join("selection_coefficient_per_lambda", safe_lam),
                dist_dict,
                display_label=f"Selection Coefficient Distance λ={lam_str}",
            )

        activation_cov_df = _matrix_df_from_array(payload.get("activation_covariance"), idx_array)
        if activation_cov_df is not None:
            save_matrix_heatmap(
                activation_cov_df,
                f"Activation Covariance\n{site_label}",
                os.path.join(config_dir, "activation_covariance.png"),
                center=0.0,
                cmap="coolwarm",
            )

        activation_corr_df = _matrix_df_from_array(payload.get("activation_correlation"), idx_array)
        if activation_corr_df is not None:
            save_matrix_heatmap(
                activation_corr_df,
                f"Activation Correlation\n{site_label}",
                os.path.join(config_dir, "activation_correlation.png"),
                center=0.0,
            )
else:
    print("Warning: latent_distance_matrices not available; skipping latent distance visualizations")

#%%
