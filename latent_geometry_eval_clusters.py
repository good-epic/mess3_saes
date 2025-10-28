#!/usr/bin/env python3
"""
Evaluate "cheating" latent clusters formed by argmax R² assignments.

This script reuses the sampling pipeline from `latent_geometry_create_metrics.py`
to (re)compute per-latent metrics, then aggregates latents into clusters based on
their maximum single-latent R². Latents whose best R² falls below a configurable
cutoff are placed into a noise cluster. For each cluster we report:

* Basic statistics about max-R² and margins
* Cluster-level belief prediction quality (R² per component) obtained by
  summing latent contributions via the precomputed sensitivity matrix

The intent is to provide a ground-truth-assisted baseline that we can later try
to reproduce using only unsupervised metrics.
"""

import os

# Force JAX/Flax workloads onto CPU by default (matches other analysis scripts)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import latent_geometry_create_metrics as lgcm
from clustering.geometry_fitting import GeometryFittingConfig, GeometryFitter
from clustering.geometries import create_geometry_catalog


DEFAULT_R2_CUTOFF = 0.02
DEFAULT_RIDGE_ALPHA = 1e-3
DEFAULT_GW_EPSILON = 0.1
DEFAULT_SINKHORN_EPSILON = 0.1
DEFAULT_SINKHORN_MAX_ITER = 1000
DEFAULT_GW_MAX_ITER = 100
DEFAULT_GW_TOL = 1e-9
DEFAULT_GEOMETRY_TARGET_SAMPLES = 1000


def _safe_label(label: str) -> str:
    """Return a filesystem-friendly label derived from an arbitrary string."""
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", label)
    sanitized = sanitized.strip("_")
    return sanitized or "config"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments, extending the base metrics parser with new options."""
    base_args = lgcm.parse_args(argv)

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--r2_cutoff",
        type=float,
        default=DEFAULT_R2_CUTOFF,
        help="Minimum max-R² required to assign a latent to a component cluster.",
    )
    extra_parser.add_argument(
        "--cluster_output_dir",
        type=str,
        default=None,
        help="Directory to write cluster summaries (defaults to <output_path>/cheat_clusters).",
    )
    extra_parser.add_argument(
        "--save_assignments_csv",
        action="store_true",
        help="Write per-configuration CSV files with latent cluster assignments.",
    )
    extra_parser.add_argument(
        "--eval_top_k",
        type=float,
        nargs="*",
        default=None,
        help="Subset of top-k values to evaluate (defaults to all generated).",
    )
    extra_parser.add_argument(
        "--eval_lambda",
        type=float,
        nargs="*",
        default=None,
        help="Subset of lambda values to evaluate (defaults to all generated).",
    )
    extra_parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=DEFAULT_RIDGE_ALPHA,
        help="Ridge regression alpha used for cluster-level fits.",
    )
    extra_parser.add_argument(
        "--compute_geometry",
        action="store_true",
        help="If set, evaluate Gromov-Wasserstein geometry fits for each cluster.",
    )
    extra_parser.add_argument(
        "--geo_simplex_k_min",
        type=int,
        default=1,
        help="Minimum simplex dimension to consider during GW fitting.",
    )
    extra_parser.add_argument(
        "--geo_simplex_k_max",
        type=int,
        default=8,
        help="Maximum simplex dimension to consider during GW fitting.",
    )
    extra_parser.add_argument(
        "--geo_include_circle",
        action="store_true",
        default=True,
        help="Include circle geometry in GW evaluation.",
    )
    extra_parser.add_argument(
        "--geo_no_circle",
        action="store_true",
        help="Disable circle geometry (overrides --geo_include_circle).",
    )
    extra_parser.add_argument(
        "--geo_include_hypersphere",
        action="store_true",
        help="Include hypersphere geometries in GW evaluation.",
    )
    extra_parser.add_argument(
        "--geo_hypersphere_dims",
        type=int,
        nargs="*",
        default=None,
        help="Hypersphere dimensions to include (e.g. --geo_hypersphere_dims 2 3).",
    )
    extra_parser.add_argument(
        "--geo_cost_fn",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Cost function for GW evaluation.",
    )
    extra_parser.add_argument(
        "--geo_gw_epsilon",
        type=float,
        default=DEFAULT_GW_EPSILON,
        help="Entropic regularization epsilon for GW solver.",
    )
    extra_parser.add_argument(
        "--geo_gw_solver",
        type=str,
        choices=["PPA", "PGD"],
        default="PPA",
        help="GW solver to use.",
    )
    extra_parser.add_argument(
        "--geo_gw_max_iter",
        type=int,
        default=DEFAULT_GW_MAX_ITER,
        help="Maximum iterations for GW solver.",
    )
    extra_parser.add_argument(
        "--geo_gw_tol",
        type=float,
        default=DEFAULT_GW_TOL,
        help="Convergence tolerance for GW solver.",
    )
    extra_parser.add_argument(
        "--geo_sinkhorn_epsilon",
        type=float,
        default=DEFAULT_SINKHORN_EPSILON,
        help="Entropic regularization for Sinkhorn distance calculations.",
    )
    extra_parser.add_argument(
        "--geo_sinkhorn_max_iter",
        type=int,
        default=DEFAULT_SINKHORN_MAX_ITER,
        help="Maximum Sinkhorn iterations when fitting geometries.",
    )
    extra_parser.add_argument(
        "--geo_target_samples",
        type=int,
        default=DEFAULT_GEOMETRY_TARGET_SAMPLES,
        help="Number of uniform samples drawn from each candidate geometry.",
    )
    extra_parser.add_argument(
        "--geo_normalize_vectors",
        action="store_true",
        default=True,
        help="Normalize decoder directions before computing costs (default on).",
    )
    extra_parser.add_argument(
        "--geo_no_normalize_vectors",
        action="store_true",
        help="Disable normalization of decoder directions before GW fitting.",
    )

    extra_args, _ = extra_parser.parse_known_args(argv)
    for key, value in vars(extra_args).items():
        setattr(base_args, key, value)

    if base_args.cluster_output_dir is None:
        base_args.cluster_output_dir = os.path.join(base_args.output_path, "cheat_clusters")

    if getattr(base_args, "geo_hypersphere_dims", None) is None:
        base_args.geo_hypersphere_dims = [2]
    base_args.geo_include_circle = bool(base_args.geo_include_circle and not base_args.geo_no_circle)
    base_args.geo_normalize_vectors = bool(base_args.geo_normalize_vectors and not base_args.geo_no_normalize_vectors)
    if hasattr(base_args, "geo_no_circle"):
        delattr(base_args, "geo_no_circle")
    if hasattr(base_args, "geo_no_normalize_vectors"):
        delattr(base_args, "geo_no_normalize_vectors")

    return base_args


def _float_or_none(value: float) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return float(value)


def _matches_param(target: float, candidates: Optional[Sequence[float]]) -> bool:
    if candidates is None:
        return True
    return any(abs(target - cand) <= 1e-8 for cand in candidates)


def _build_geometry_tools(args: argparse.Namespace):
    if not getattr(args, "compute_geometry", False):
        return None, None

    config = GeometryFittingConfig()
    config.simplex_k_range = (args.geo_simplex_k_min, args.geo_simplex_k_max)
    config.include_circle = args.geo_include_circle
    config.include_hypersphere = args.geo_include_hypersphere
    config.hypersphere_dims = list(args.geo_hypersphere_dims or [])
    config.gw_epsilon = args.geo_gw_epsilon
    config.gw_solver = args.geo_gw_solver
    config.gw_max_iter = args.geo_gw_max_iter
    config.gw_tol = args.geo_gw_tol
    config.sinkhorn_epsilon = args.geo_sinkhorn_epsilon
    config.sinkhorn_max_iter = args.geo_sinkhorn_max_iter
    config.n_target_samples = args.geo_target_samples
    config.cost_fn = args.geo_cost_fn
    config.normalize_vectors = args.geo_normalize_vectors

    fitter = GeometryFitter(config)
    geometries = create_geometry_catalog(
        simplex_k_range=config.simplex_k_range,
        include_circle=config.include_circle,
        include_hypersphere=config.include_hypersphere,
        hypersphere_dims=config.hypersphere_dims,
    )
    if not geometries:
        raise ValueError("No candidate geometries available for GW evaluation.")
    return fitter, geometries


def _fit_cluster_component_r2(
    feature_matrix: np.ndarray,
    belief_matrix: np.ndarray,
    component_slices: Mapping[str, Tuple[int, int]],
    display_map: Mapping[str, str],
    latent_indices: np.ndarray,
    alpha: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit per-component ridge regressions for a cluster and report R² scores.
    """
    cluster_scores: Dict[str, Dict[str, Any]] = {}

    if latent_indices.size == 0:
        for comp_name in component_slices.keys():
            cluster_scores[display_map.get(comp_name, comp_name)] = {}
        return cluster_scores

    X = feature_matrix[:, latent_indices]

    # Guard against degenerate design matrices
    if X.shape[1] == 0 or not np.any(np.isfinite(X)):
        for comp_name in component_slices.keys():
            cluster_scores[display_map.get(comp_name, comp_name)] = {}
        return cluster_scores

    active_mask = np.any(X != 0.0, axis=1)
    if not np.any(active_mask):
        return cluster_scores

    cluster_features_active = X[active_mask]
    if cluster_features_active.shape[0] < 2:
        return cluster_scores

    def _prepare_targets(label: str, matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        if label.startswith("tom_quantum"):
            if matrix.shape[1] <= 1:
                return np.empty((matrix.shape[0], 0), dtype=matrix.dtype), 1
            return matrix[:, 1:], 1
        return matrix, 0

    for comp_name, (start, end) in component_slices.items():
        if end <= start:
            cluster_scores[display_map.get(comp_name, comp_name)] = {}
            continue
        y_true = belief_matrix[:, start:end]

        label = display_map.get(comp_name, comp_name)
        targets_prepped, dropped_dims = _prepare_targets(label, y_true)
        if targets_prepped.shape[1] == 0:
            continue
        targets_active = targets_prepped[active_mask]
        if targets_active.shape[0] < 2:
            continue

        try:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(cluster_features_active, targets_active)
            preds = model.predict(cluster_features_active)
        except Exception:
            continue

        target_mean = targets_active.mean(axis=0, keepdims=True)
        ss_res = np.sum((targets_active - preds) ** 2, axis=0)
        ss_tot = np.sum((targets_active - target_mean) ** 2, axis=0)
        denom = np.where(ss_tot == 0.0, 1.0, ss_tot)
        r2 = 1.0 - ss_res / denom

        cluster_scores[label] = {
            "per_dimension": r2.astype(float).tolist(),
            "mean_r2": float(np.mean(r2)),
            "n_active_samples": int(active_mask.sum()),
            "dropped_constant_dims": int(dropped_dims),
        }
    return cluster_scores


def _describe_vector(values: np.ndarray) -> Dict[str, Optional[float]]:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": _float_or_none(np.mean(finite_vals)),
        "median": _float_or_none(np.median(finite_vals)),
        "min": _float_or_none(np.min(finite_vals)),
        "max": _float_or_none(np.max(finite_vals)),
    }


def _compute_geometry_summary(
    fitter: Optional[GeometryFitter],
    geometries: Optional[Sequence],
    decoder_dirs: Optional[np.ndarray],
    latent_indices: np.ndarray,
) -> Optional[Dict[str, Any]]:
    if fitter is None or geometries is None or decoder_dirs is None or latent_indices.size < 2:
        return None

    points = decoder_dirs[latent_indices]
    if points.shape[0] < 2:
        return None

    fits = fitter.fit_cluster_to_geometries(points, geometries, verbose=False)
    if not fits:
        return {"status": "failed", "reason": "no_valid_fits"}

    sorted_geoms = sorted(fits.items(), key=lambda item: item[1].optimal_distance)
    best_name, best_fit = sorted_geoms[0]
    summary = {
        "best_geometry": best_name,
        "best_optimal_distance": float(best_fit.optimal_distance),
        "all_fits": {
            name: fit.to_summary_dict(include_raw_arrays=False) for name, fit in fits.items()
        },
    }
    return summary


def evaluate_configuration(
    args: argparse.Namespace,
    subset: pd.DataFrame,
    metrics_entry: Dict[str, Any],
    fitter: Optional[GeometryFitter],
    geometries: Optional[List],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    r2_columns = [
        col
        for col in subset.columns
        if col.startswith("r2_") and not col.endswith("_indicator") and col != "r2_indicator_sum"
    ]
    if not r2_columns:
        raise ValueError("No R² columns found in metrics dataframe subset.")

    component_labels = [col[len("r2_") :] for col in r2_columns]
    display_map = metrics_entry.get("component_display_names")
    if not display_map:
        component_order = metrics_entry.get("component_order")
        if component_order is None:
            component_order = component_labels
        display_map = {name: name for name in component_order}
    label_to_component = {disp: comp for comp, disp in display_map.items()}

    r2_values = subset[r2_columns].to_numpy(float)
    valid_mask = np.isfinite(r2_values)
    has_valid = valid_mask.any(axis=1)
    safe_values = np.where(valid_mask, r2_values, -np.inf)

    argmax_indices = safe_values.argmax(axis=1)
    argmax_indices[~has_valid] = -1
    max_r2 = np.where(has_valid, r2_values[np.arange(len(subset)), argmax_indices], np.nan)

    valid_counts = valid_mask.sum(axis=1)
    if safe_values.shape[1] > 1:
        second_best = np.partition(safe_values, -2, axis=1)[:, -2]
        second_best[valid_counts < 2] = np.nan
        second_best[np.isneginf(second_best)] = np.nan
    else:
        second_best = np.full(len(subset), np.nan)
    margin = max_r2 - second_best

    argmax_labels: List[Optional[str]] = []
    for idx in argmax_indices:
        if idx < 0:
            argmax_labels.append(None)
        else:
            argmax_labels.append(component_labels[idx])

    subset = subset.copy()
    subset["max_r2"] = max_r2
    subset["second_r2"] = second_best
    subset["margin"] = margin
    subset["argmax_label"] = argmax_labels

    noise_mask = (~has_valid) | (np.where(has_valid, max_r2, -np.inf) < args.r2_cutoff)
    assigned_clusters = []
    assigned_components = []
    for is_noise, label in zip(noise_mask, argmax_labels):
        if is_noise or label is None:
            assigned_clusters.append("noise")
            assigned_components.append(None)
        else:
            assigned_clusters.append(label)
            assigned_components.append(label_to_component.get(label, label))

    subset["assigned_cluster"] = assigned_clusters
    subset["assigned_component_name"] = assigned_components

    if "belief_matrix" not in metrics_entry:
        raise KeyError("Metric bundle missing 'belief_matrix'; rerun latent_geometry_create_metrics with updated version.")

    feature_matrix = metrics_entry["feature_matrix"]
    belief_matrix = metrics_entry["belief_matrix"]
    component_slices = metrics_entry["component_slices"]
    decoder_dirs = metrics_entry.get("decoder_dirs")

    cluster_summaries: List[Dict[str, Any]] = []
    for cluster_label, cluster_df in subset.groupby("assigned_cluster"):
        latent_indices = cluster_df["latent_index"].astype(int).to_numpy()
        cluster_scores = _fit_cluster_component_r2(
            feature_matrix,
            belief_matrix,
            component_slices,
            display_map,
            latent_indices,
            alpha=args.ridge_alpha,
        )
        geometry_summary = _compute_geometry_summary(
            fitter,
            geometries,
            decoder_dirs,
            latent_indices,
        )

        summary = {
            "cluster_label": cluster_label,
            "latent_count": int(len(cluster_df)),
            "mean_max_r2": _float_or_none(np.nanmean(cluster_df["max_r2"].to_numpy())),
            "median_max_r2": _float_or_none(np.nanmedian(cluster_df["max_r2"].to_numpy())),
            "mean_margin": _float_or_none(np.nanmean(cluster_df["margin"].to_numpy())),
            "median_margin": _float_or_none(np.nanmedian(cluster_df["margin"].to_numpy())),
            "max_r2_stats": _describe_vector(cluster_df["max_r2"].to_numpy()),
            "margin_stats": _describe_vector(cluster_df["margin"].to_numpy()),
            "cluster_component_r2": cluster_scores,
        }
        if geometry_summary is not None:
            summary["geometry"] = geometry_summary
        cluster_summaries.append(summary)

    config_summary = {
        "site": subset["site"].iloc[0],
        "sae_type": subset["sae_type"].iloc[0],
        "top_k_k": _float_or_none(subset["top_k_k"].iloc[0]),
        "vanilla_lambda": _float_or_none(subset["vanilla_lambda"].iloc[0]),
        "r2_cutoff": float(args.r2_cutoff),
        "ridge_alpha": float(args.ridge_alpha),
        "num_latents": int(len(subset)),
        "geometry_evaluated": bool(fitter is not None and geometries is not None),
        "clusters": cluster_summaries,
    }

    return config_summary, subset


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    Path(args.cluster_output_dir).mkdir(parents=True, exist_ok=True)

    fitter, geometries = _build_geometry_tools(args)

    print("Recomputing latent metrics and cached activations...")
    df, _l0_df, metric_store = lgcm.run_analysis(args)

    summaries: Dict[str, Any] = {}
    for key, metrics_entry in metric_store.items():
        site, sae_type, param_value = key

        if sae_type == "top_k" and not _matches_param(param_value, args.eval_top_k):
            continue
        if sae_type == "vanilla" and not _matches_param(param_value, args.eval_lambda):
            continue

        subset = df[
            (df["site"] == site)
            & (df["sae_type"] == sae_type)
            & (
                (df["top_k_k"] == param_value)
                if sae_type == "top_k"
                else (df["vanilla_lambda"] == param_value)
            )
        ]
        if subset.empty:
            continue

        config_summary, annotated_df = evaluate_configuration(args, subset, metrics_entry, fitter, geometries)
        safe_name = _safe_label(f"{site}_{sae_type}_{param_value}")

        summary_path = Path(args.cluster_output_dir) / f"{safe_name}.json"
        with summary_path.open("w") as fh:
            json.dump(config_summary, fh, indent=2, sort_keys=True, allow_nan=False)

        if args.save_assignments_csv:
            csv_cols = [
                "site",
                "sae_type",
                "top_k_k",
                "vanilla_lambda",
                "latent_index",
                "assigned_cluster",
                "assigned_component_name",
                "argmax_label",
                "max_r2",
                "second_r2",
                "margin",
                "activation_rate",
            ]
            csv_path = Path(args.cluster_output_dir) / f"{safe_name}_assignments.csv"
            annotated_df.loc[:, [col for col in csv_cols if col in annotated_df.columns]].to_csv(csv_path, index=False)

        summaries[safe_name] = config_summary
        print(
            f"Processed {site} / {sae_type} param={param_value}: "
            f"{len(config_summary['clusters'])} clusters written to {summary_path}"
        )

    overall_path = Path(args.cluster_output_dir) / "summary.json"
    with overall_path.open("w") as fh:
        json.dump(summaries, fh, indent=2, sort_keys=True, allow_nan=False)
    print(f"Saved overall summary to {overall_path}")


if __name__ == "__main__":
    main()
