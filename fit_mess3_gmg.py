#%%
# === Imports and Configuration === #
###################################
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import argparse
import json
import math
from datetime import datetime
from typing import Any, Dict, List

import torch
import jax
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np

from sae_variants.banded_cov_sae import BandedCovarianceSAE

from clustering import (
    ClusteringConfig,
    SiteClusteringPipeline,
    evaluate_clustering_metrics,
)
from multipartite_utils import (
    MultipartiteSampler,
    _resolve_device,
    _load_process_stack,
    _load_transformer,
    _select_sites,
)
from mess3_gmg_analysis_utils import (
    extract_topk_l2,
    extract_vanilla_l2,
    extract_banded_l2,
    compute_average_l2,
    compute_average_l2,
    find_elbow_k,
    load_metrics_summary,
    plot_activity_histogram,
    plot_activity_histograms_by_site,
    plot_activity_histograms_site_clusters,
    plot_l2_bar_chart,
)


# Mapping from human-readable site names to HookedTransformer cache keys
SITE_HOOK_MAP = {
    "embeddings": "hook_embed",
    "layer_0": "blocks.0.hook_resid_post",
    "layer_1": "blocks.1.hook_resid_post",
    "layer_2": "blocks.2.hook_resid_post",
    "layer_3": "blocks.3.hook_resid_post",
}


PRESET_PROCESS_CONFIGS = {
    "single_mess3": [
        {"type": "mess3", "params": {"x": 0.1, "a": 0.7}},
    ],
    "3xmess3_2xtquant_001": [
        {
            "type": "mess3",
            "instances": [
                {"x": 0.10, "a": 0.50},
                {"x": 0.25, "a": 0.80},
                {"x": 0.40, "a": 0.20},
            ],
        },
        {
            "type": "tom_quantum",
            "instances": [
                {"alpha": 0.9, "beta": float(1.3)},
                {"alpha": 1.0, "beta": float(np.sqrt(51))},
            ],
        },
    ],
    "3xmess3_2xtquant_002": [
        {
            "type": "mess3",
            "instances": [
                {"x": 0.10, "a": 0.50},
                {"x": 0.25, "a": 0.80},
                {"x": 0.40, "a": 0.20},
            ],
        },
        {
            "type": "tom_quantum",
            "instances": [
                {"alpha": 0.9, "beta": float(1.3)},
                {"alpha": 1.0, "beta": float(np.sqrt(51))},
            ],
        },
    ],
}


#%%
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multipartite SAEs for decoder/PCA visualizations")

    # Paths and devices
    parser.add_argument("--sae_folder", type=str, default="outputs/saes/multipartite_002", help="Folder with SAE checkpoints")
    parser.add_argument("--metrics_summary", type=str, default=None, help="metrics_summary.json path; defaults to <sae_folder>/metrics_summary.json")
    parser.add_argument("--output_dir", type=str, default="outputs/reports/multipartite_002", help="Root directory for analysis outputs")
    parser.add_argument("--model_ckpt", type=str, default="outputs/checkpoints/multipartite_002/checkpoint_step_125000.pt", help="Transformer checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log_to_mlflow", action="store_true", help="Log metrics and artifacts to MLflow")
    parser.add_argument("--mlflow_experiment", type=str, default=None, help="MLflow experiment path (e.g., /Shared/my-experiment)")
    parser.add_argument("--mlflow_run_name_base", type=str, default=None, help="Base run name; timestamp is appended automatically")

    # Process configuration
    parser.add_argument("--process_config", type=str, default="process_configs.json", help="Path to JSON describing a stack of generative processes or a mapping of named configurations")
    parser.add_argument("--process_config_name", type=str, default="3xmess3_2xtquant_002", help="Key within --process_config when the file stores multiple named configurations")
    parser.add_argument("--process_preset", type=str, default=None, help="Named preset for generative process configuration")

    # Transformer configuration (if checkpoint lacks config)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--d_vocab", type=int, default=None)
    parser.add_argument("--act_fn", type=str, default="relu")

    # SAE and clustering controls
    parser.add_argument("--sites", type=str, nargs="+", default=None, help="Subset of site names to include (e.g. embeddings layer_0)")
    parser.add_argument("--sae_type", type=str, default="top_k", choices=["top_k", "vanilla", "banded"], help="SAE architecture type")
    parser.add_argument("--force_k", type=int, default=None, help="Override elbow selection with manual k (for top_k SAEs)")
    parser.add_argument("--force_lambda", type=float, default=None, help="Override elbow selection with manual lambda (for vanilla SAEs)")
    parser.add_argument("--force_banded_params", type=str, default=None, help="Override selection with manual params for banded SAEs (e.g. 'ls_0p005__la_0p005')")
    parser.add_argument("--clustering_method", type=str, default="k_subspaces", choices=["spectral", "k_subspaces", "ensc"], help="Clustering method for decoder directions")
    parser.add_argument("--sim_metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "jaccard", "dice", "overlap", "phi", "mutual_info"],
                        help="Similarity metric for spectral clustering. Geometry-based: cosine, euclidean. Co-occurrence-based: jaccard, dice, overlap, phi, mutual_info")
    parser.add_argument("--max_clusters", type=int, default=12, help="Upper bound for eigengap clustering (spectral method only)")
    parser.add_argument("--min_clusters", type=int, default=2, help="Lower bound for eigengap clustering (spectral method only)")
    parser.add_argument("--plot_eigengap", action="store_true", help="Plot eigengap spectrum diagnostics (spectral method only)")
    parser.add_argument("--center_decoder_rows", action="store_true", help="Center and renormalize decoder rows before computing similarities (spectral method only)")

    # Co-occurrence metric configuration (for jaccard, dice, overlap, phi, mutual_info)
    parser.add_argument("--cooc_n_batches", type=int, default=1000,
                        help="Number of batches for co-occurrence collection. Total tokens = n_batches * batch_size * seq_len")
    parser.add_argument("--cooc_batch_size", type=int, default=32,
                        help="Sequences per batch for co-occurrence collection")
    parser.add_argument("--cooc_seq_len", type=int, default=256,
                        help="Tokens per sequence for co-occurrence collection")
    parser.add_argument("--cooc_activation_threshold", type=float, default=1e-6,
                        help="Feature counts as firing if |activation| > threshold")
    parser.add_argument("--cooc_skip_special_tokens", action="store_true", default=True,
                        help="Skip BOS token (position 0) when collecting co-occurrence")
    parser.add_argument("--cooc_cache_path", type=str, default=None,
                        help="Path to save/load co-occurrence stats (avoids recomputation when trying different metrics)")

    # Subspace clustering controls
    parser.add_argument("--subspace_rank", type=int, default=None, help="Rank of each subspace for k_subspaces/ensc methods (if None, auto-detected per cluster)")
    parser.add_argument("--subspace_n_clusters", type=int, default=None, help="Number of clusters for subspace methods (if None, auto-detected)")
    parser.add_argument("--use_grid_search", action="store_true", help="Use grid search instead of automatic parameter selection for k_subspaces")
    parser.add_argument("--subspace_k_values", type=int, nargs="+", default=[4, 5, 6, 7, 8], help="K values to try in grid search for k_subspaces (only used with --use_grid_search)")
    parser.add_argument("--subspace_r_values", type=int, nargs="+", default=[2, 3, 4, 5], help="Subspace rank values to try in grid search for k_subspaces (only used with --use_grid_search)")
    parser.add_argument("--subspace_variance_threshold", type=float, default=0.95, help="Cumulative variance threshold for auto rank detection (k_subspaces/ensc, only used when subspace_rank=None)")
    parser.add_argument("--subspace_gap_threshold", type=float, default=2.0, help="Singular value gap ratio threshold for auto rank detection (k_subspaces/ensc, only used when subspace_rank=None)")
    parser.add_argument("--ensc_lambda1", type=float, default=0.01, help="L1 regularization weight for ENSC")
    parser.add_argument("--ensc_lambda2", type=float, default=0.001, help="L2 regularization weight for ENSC")
    parser.add_argument("--cosine_dedup_threshold", type=float, default=0.995, help="Cosine similarity threshold for removing near-duplicate decoder directions")

    # Belief-aligned seeding controls
    parser.add_argument("--seed_with_beliefs", action="store_true", help="Use belief-state supervision to seed latent clusters")
    parser.add_argument("--belief_ridge_alpha", type=float, default=1e-3, help="Ridge regularization for residual-to-belief readout fit")
    parser.add_argument("--seed_lasso_alpha", type=float, default=None, help="Override alpha for Lasso when selecting latent seeds (defaults to cross-validated)")
    parser.add_argument("--seed_lasso_cv", type=int, default=5, help="Cross-validation folds for Lasso seed selection")
    parser.add_argument("--seed_lasso_max_iter", type=int, default=5000, help="Maximum iterations for Lasso seed selection")
    parser.add_argument("--seed_coef_threshold", type=float, default=1e-4, help="Minimum aggregate coefficient magnitude required to keep a latent in the seed set")
    parser.add_argument("--seed_max_latents", type=int, default=None, help="Maximum number of latents to seed per component (None keeps all above threshold)")
    parser.add_argument("--seed_lock_mode", type=str, default="fixed", choices=["fixed", "release_after_init"], help="How to handle seeded latents during k-subspaces: 'fixed' keeps them in their seeded cluster throughout; 'release_after_init' only uses them to initialise bases, then lets them move")
    parser.add_argument("--protect_seed_duplicates", action="store_true", help="Keep duplicate decoder directions if they appear in belief-seeded clusters")

    # Sampling controls for PCA projections
    parser.add_argument("--sample_sequences", type=int, default=1024, help="Number of sequences to sample")
    parser.add_argument("--sample_seq_len", type=int, default=None, help="Sequence length for sampling; defaults to model n_ctx")
    parser.add_argument("--max_activations", type=int, default=50000, help="Maximum activations per site for PCA analysis")
    parser.add_argument("--cluster_activation_threshold", type=float, default=1e-6, help="Minimum latent activation threshold")
    parser.add_argument("--min_cluster_samples", type=int, default=512, help="Minimum samples per cluster to fit PCA")
    parser.add_argument("--plot_max_points", type=int, default=4000, help="Maximum scatter points in PCA plots")
    parser.add_argument("--latent_activity_threshold", type=float, default=0.01, help="Minimum fraction of samples where a latent must be active to keep it")
    parser.add_argument("--latent_activation_eps", type=float, default=1e-6, help="Magnitude threshold when computing latent activity rates")
    parser.add_argument("--activation_batches", type=int, default=8, help="Number of independent batches to sample for latent activity statistics")
    parser.add_argument("--pca_components", type=int, default=6, help="Number of PCA components to compute before selecting the top three for plotting")
    parser.add_argument("--skip_pca_plots", action="store_true", help="Skip PCA fitting and plotting for cluster reconstructions")

    # Cluster EPDF controls
    parser.add_argument("--build_cluster_epdfs", action="store_true", help="Generate EPDF visualizations for each cluster")
    parser.add_argument("--epdf_sites", type=str, nargs="+", default=None, help="Subset of sites to generate cluster EPDFs for (e.g., layer_2); if None, processes all sites")
    parser.add_argument("--epdf_plot_mode", type=str, default="both", choices=["both", "all_only", "per_latent_only"], help="Which EPDF plots to generate")
    parser.add_argument("--epdf_bandwidth", default="scott", help="KDE bandwidth method ('scott', 'silverman', or float value)")
    parser.add_argument("--epdf_grid_size", type=int, default=100, help="Grid density for KDE evaluation over belief geometry")
    parser.add_argument("--epdf_min_samples", type=int, default=20, help="Minimum active samples required before fitting EPDFs for a latent")

    # Geometry-guided refinement controls
    parser.add_argument("--refine_with_geometries", action="store_true", help="Refine clustering using belief geometry fitting (GMG)")
    parser.add_argument("--geo_soft_method", type=str, default="top_m", choices=["top_m", "threshold", "both"], help="Method for extracting soft cluster assignments")
    parser.add_argument("--geo_top_m", type=int, default=3, help="Keep latents in top M clusters (for top_m method)")
    parser.add_argument("--geo_soft_threshold", type=float, default=0.1, help="Minimum soft weight to keep (for threshold method)")
    parser.add_argument("--geo_simplex_k_min", type=int, default=1, help="Minimum K for simplex geometries to test")
    parser.add_argument("--geo_simplex_k_max", type=int, default=8, help="Maximum K for simplex geometries to test")
    parser.add_argument("--geo_include_circle", action="store_true", default=True, help="Test circle (Bloch sphere slice) geometry")
    parser.add_argument("--geo_include_hypersphere", action="store_true", help="Test hypersphere geometries")
    parser.add_argument("--geo_cost_fn", type=str, default="cosine", choices=["cosine", "euclidean"], help="Cost function for computing pairwise distances")
    parser.add_argument("--geo_gw_epsilon", type=float, default=0.1, help="Entropic regularization for Gromov-Wasserstein")
    parser.add_argument("--geo_sinkhorn_epsilon", type=float, default=0.1, help="Entropic regularization for Sinkhorn")
    parser.add_argument("--geo_sinkhorn_max_iter", type=int, default=1000, help="Maximum iterations for Sinkhorn algorithm")
    parser.add_argument("--geo_threshold_mode", type=str, default="normalized", choices=["normalized", "raw"],
                        help="Threshold mode: 'normalized' applies threshold to [0,1] normalized scores per cluster, 'raw' applies threshold to raw values across all clusters")
    parser.add_argument("--geo_per_point_threshold", type=float, default=0.5,
                        help="Per-point threshold for filtering. If mode='normalized', this is applied to [0,1] normalized scores (default 0.5). If mode='raw', this is applied to raw GW contribution values.")
    parser.add_argument("--geo_optimal_distortion_threshold", type=float, default=1.0, help="Optimal GW distortion threshold above which cluster fit is poor")
    parser.add_argument("--geo_filter_metrics", type=str, nargs="+", default=["gw_full"], choices=["gw_full"],
                        help="Per-point distortion metric to use for filtering (currently only gw_full supported)")

    args, _ = parser.parse_known_args()

    if args.process_config and args.process_preset:
        parser.error("Specify at most one of --process_config or --process_preset")

    return args


# === TEMP: Manually override args from run_fit_mess3_gmg.sh for debugging ===
# args.sae_folder = "outputs/saes/multipartite_003e"
# args.metrics_summary = "outputs/saes/multipartite_003e/metrics_summary.json"
# args.output_dir = "outputs/reports/multipartite_003e"
# args.model_ckpt = "outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt"
# args.device = "cuda"
# args.seed = 43
# args.process_config = "process_configs.json"
# args.process_config_name = "3xmess3_2xtquant_003"
# args.d_model = 128
# args.n_heads = 4
# args.n_layers = 3
# args.n_ctx = 16
# args.d_head = 32
# args.act_fn = "relu"
# args.sim_metric = "cosine"
# args.clustering_method = "k_subspaces"
# args.max_clusters = 12
# args.plot_eigengap = True
# args.ensc_lambda1 = 0.01
# args.ensc_lambda2 = 0.001
# args.cosine_dedup_threshold = 0.99
# args.seed_lasso_cv = 8
# args.seed_coef_threshold = 1e-4
# args.seed_max_latents = 40
# args.sample_sequences = 1024
# args.max_activations = 50000
# args.cluster_activation_threshold = 1e-6
# args.center_decoder_rows = True
# args.latent_activity_threshold = 0.01
# args.latent_activation_eps = 1e-6
# args.activation_batches = 8
# args.build_cluster_epdfs = True
# args.epdf_sites = ["layer_0", "layer_1", "layer_2"]
# args.epdf_plot_mode = "both"
# args.epdf_bandwidth = "scott"
# args.epdf_grid_size = 200
# args.subspace_n_clusters = 8
# args.refine_with_geometries = True
# args.geo_include_circle = True
# args.geo_filter_metrics = ["sinkhorn", "gw_full", "marginal", "local"]
# args.geo_sinkhorn_max_iter = 5000
# args.geo_sinkhorn_epsilon = 0.2





#%%
# ==== Setup ==== #
###################

args = _parse_args()
device = _resolve_device(args.device)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

mlflow_enabled = bool(args.log_to_mlflow)
mlflow_module = None
mlflow_run_id = None
mlflow_run_name = None

if mlflow_enabled:
    if not args.mlflow_experiment:
        raise ValueError("--mlflow_experiment is required when --log_to_mlflow is enabled")
    if not args.mlflow_run_name_base:
        raise ValueError("--mlflow_run_name_base is required when --log_to_mlflow is enabled")

    from mlflow_utils import start_or_continue_run, log_script_params  # noqa: E402
    import mlflow as mlflow_module  # noqa: E402

    mlflow_run_name = f"{args.mlflow_run_name_base}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        mlflow_run_id = start_or_continue_run(
            script_name="fit_mess3_gmg",
            experiment_name=args.mlflow_experiment,
            run_name=mlflow_run_name,
        )
    except Exception as mlflow_exc:
        print(f"fit_mess3_gmg: MLflow disabled for this run ({mlflow_exc})")
        mlflow_enabled = False
        mlflow_run_id = None

    if mlflow_run_id is not None:
        log_script_params("fit_mess3_gmg", args)
        mlflow_module.log_param("fit_mess3_gmg_run_name", mlflow_run_name)
        mlflow_module.log_param("fit_mess3_gmg_run_id", mlflow_run_id)
        tracking_host = os.getenv("DATABRICKS_HOST")
        if tracking_host:
            mlflow_module.log_param("fit_mess3_gmg_tracking_uri", tracking_host)

        import atexit  # noqa: E402

        def _end_run_at_exit() -> None:
            if mlflow_module.active_run():
                mlflow_module.end_run()

        atexit.register(_end_run_at_exit)

        # PCA plots are not useful for logging runs; skip them automatically when MLflow logging is active.
        args.skip_pca_plots = True
        print("MLflow logging enabled; skipping PCA plot generation.")

    def _extract_numeric(data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a nested dictionary containing only numeric leaves."""
        numeric_only: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                nested = _extract_numeric(value)
                if nested:
                    numeric_only[key] = nested
            elif isinstance(value, (int, float, np.floating, np.integer)):
                numeric_only[key] = float(value)
        return numeric_only

    def _log_numeric(prefix: str, payload: Dict[str, Any]) -> None:
        for key, value in payload.items():
            metric_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                _log_numeric(metric_key, value)
            else:
                if value is None or (isinstance(value, float) and not math.isfinite(value)):
                    continue
                mlflow_module.log_metric(metric_key, float(value))

else:
    mlflow_enabled = False
    mlflow_run_id = None

mlflow_active_run = mlflow_enabled and mlflow_run_id is not None

process_cfg_raw, components, data_source = _load_process_stack(args, PRESET_PROCESS_CONFIGS)
if isinstance(data_source, MultipartiteSampler):
    vocab_size = data_source.vocab_size
    process_kind = "multipartite"
    component_summary = [
        {
            "name": comp.name,
            "vocab_size": int(comp.vocab_size),
            "state_dim": int(comp.state_dim),
        }
        for comp in data_source.components
    ]
    print(
        f"Multipartite process with components {[c['name'] for c in component_summary]}"
        f" → vocab={vocab_size}, belief_dim={data_source.belief_dim}"
    )
else:
    vocab_size = data_source.vocab_size
    process_kind = "single"
    component_summary = [
        {
            "name": getattr(data_source, "name", "process"),
            "vocab_size": int(vocab_size),
            "state_dim": int(getattr(data_source, "num_states", 0)),
        }
    ]
    print(f"Process: vocab_size={vocab_size}, states={component_summary[0]['state_dim']}")

model, cfg = _load_transformer(args, device, vocab_size)
seq_len = args.sample_seq_len if args.sample_seq_len is not None else cfg.n_ctx

metrics_path = args.metrics_summary or os.path.join(args.sae_folder, "metrics_summary.json")
metrics_summary = load_metrics_summary(metrics_path)
sites = _select_sites(metrics_summary, args.sites, SITE_HOOK_MAP)
if not sites:
    raise ValueError("No valid sites available for analysis")


#%%
# ==== Choose K/Lambda and Setup Output ==== #
###############################################
# Extract L2 metrics based on SAE type
if args.sae_type == "top_k":
    l2_by_site = extract_topk_l2(metrics_summary, site_filter=sites)
    if not l2_by_site:
        raise ValueError("No top-k SAE metrics found for the selected sites")
    param_label = "k"
elif args.sae_type == "vanilla":
    l2_by_site = extract_vanilla_l2(metrics_summary, site_filter=sites)
    if not l2_by_site:
        raise ValueError("No vanilla SAE metrics found for the selected sites")
    param_label = "lambda"
elif args.sae_type == "banded":
    l2_by_site = extract_banded_l2(metrics_summary, site_filter=sites)
    if not l2_by_site:
        raise ValueError("No banded SAE metrics found for the selected sites")
    param_label = "params"
else:
    raise ValueError(f"Unknown SAE type: {args.sae_type}")

average_l2 = compute_average_l2(l2_by_site)
if not average_l2:
    raise ValueError(f"Unable to compute average L2 across {param_label} values")

# Select parameter value (k, lambda, or params) per site
selected_k_by_site: dict[str, Any] = {}
if args.sae_type == "top_k" and args.force_k is not None:
    print(f"Using forced k={args.force_k} for all sites")
    selected_k_by_site = {site: int(args.force_k) for site in sites}
elif args.sae_type == "vanilla" and args.force_lambda is not None:
    print(f"Using forced lambda={args.force_lambda} for all sites")
    selected_k_by_site = {site: float(args.force_lambda) for site in sites}
elif args.sae_type == "banded" and args.force_banded_params is not None:
    print(f"Using forced params={args.force_banded_params} for all sites")
    # Parse force string "ls_0p005__la_0p005" -> (0.005, 0.005)
    try:
        parts = args.force_banded_params.split("__")
        ls_val = float(parts[0].replace("ls_", "").replace("p", "."))
        la_val = float(parts[1].replace("la_", "").replace("p", "."))
        forced_tuple = (ls_val, la_val)
        selected_k_by_site = {site: forced_tuple for site in sites}
    except Exception as e:
        raise ValueError(f"Failed to parse force_banded_params '{args.force_banded_params}': {e}")
else:
    for site in sites:
        site_metrics = l2_by_site.get(site, {})
        if not site_metrics:
            raise ValueError(f"No {args.sae_type} metrics available for site '{site}'")
        
        if args.sae_type == "banded":
            # Simple heuristic: choose params with minimum L2
            # (Elbow method is harder to define for 2D param space without a clear ordering)
            best_param = min(site_metrics.items(), key=lambda x: x[1])[0]
            selected_k_by_site[site] = best_param
            print(f"{site}: selected {param_label}={best_param} (min L2)")
        else:
            sorted_params = sorted(site_metrics.keys())
            losses = [site_metrics[p] for p in sorted_params]
            # Convert to int for elbow_k (works with both int and float)
            int_params = [int(p) if isinstance(p, int) else p for p in sorted_params]
            selected_param = find_elbow_k(int_params, losses, prefer_high_k=True)
            if args.sae_type == "top_k":
                selected_k_by_site[site] = int(selected_param)
            else:
                selected_k_by_site[site] = float(sorted_params[int_params.index(selected_param)])
            print(f"{site}: selected {param_label}={selected_k_by_site[site]} via elbow heuristic")

# Determine run label
if args.sae_type == "banded":
    # Just use "banded" prefix, maybe with params if unique
    unique_vals = list(set(selected_k_by_site.values()))
    if len(unique_vals) == 1:
        p = unique_vals[0]
        # Format back to string "ls_0p005__la_0p005"
        p_str = f"ls_{str(p[0]).replace('.', 'p')}__la_{str(p[1]).replace('.', 'p')}"
        run_dir_label = f"banded_{p_str}"
    else:
        run_dir_label = "banded_mixed"
else:
    unique_param_values = set(selected_k_by_site.values())
    if len(unique_param_values) == 1:
        param_val = next(iter(unique_param_values))
        if args.sae_type == "top_k":
            run_dir_label = f"k{int(param_val)}"
        else:
            run_dir_label = f"lambda{param_val}"
    else:
        run_dir_label = f"{param_label}var"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(args.output_dir, f"{run_dir_label}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Plotting L2 bar chart might be tricky for banded (2D), but we can try flattening keys
# For now, skip L2 plot for banded or adapt it? 
# The existing plot_l2_bar_chart expects int keys (or maybe float).
# Let's skip it for banded for now to avoid breaking.
if args.sae_type != "banded":
    l2_plot_path = os.path.join(run_dir, "l2_summary.png")
    # Ensure keys are compatible
    # plot_l2_bar_chart expects Mapping[str, Mapping[int, float]] but actually handles generic keys if we check the code?
    # The code casts keys to string for labels, so it might work if we pass it.
    # But let's be safe.
    try:
        plot_l2_bar_chart(l2_by_site, l2_plot_path, title=f"L2 Loss ({args.sae_type})")
    except Exception as e:
        print(f"Skipping L2 plot: {e}")

with open(os.path.join(run_dir, "l2_summary.json"), "w", encoding="utf-8") as f:
    # Convert params to compatible format for JSON
    per_site_dict = {}
    for site, metrics in l2_by_site.items():
        if args.sae_type == "top_k":
            per_site_dict[site] = {int(k): float(v) for k, v in metrics.items()}
        elif args.sae_type == "vanilla":
            per_site_dict[site] = {float(k): float(v) for k, v in metrics.items()}
        else:
            # Banded: keys are tuples
            per_site_dict[site] = {str(k): float(v) for k, v in metrics.items()}

    average_dict = {}
    for param, val in average_l2.items():
        if args.sae_type == "top_k":
            average_dict[int(param)] = float(val)
        elif args.sae_type == "vanilla":
            average_dict[float(param)] = float(val)
        else:
            average_dict[str(param)] = float(val)

    # Handle selected_param_by_site values
    sel_param_dict = {}
    for site, p in selected_k_by_site.items():
        if isinstance(p, tuple):
            sel_param_dict[site] = list(p)
        else:
            sel_param_dict[site] = float(p)

    json.dump(
        {
            "sae_type": args.sae_type,
            "per_site": per_site_dict,
            "average": average_dict,
            "selected_param_by_site": sel_param_dict,
            "process_kind": process_kind,
            "components": component_summary,
        },
        f,
        indent=2,
    )



#%%
# ==== Sample Tokens and Prepare Beliefs ==== #
################################################
batch_size = args.sample_sequences
sample_len = seq_len

# Sample tokens and get corresponding beliefs for EPDF generation
if isinstance(data_source, MultipartiteSampler):
    key = jax.random.PRNGKey(args.seed)
    key, belief_states, product_tokens, component_observations = data_source.sample(
        key, batch_size, sample_len
    )
    tokens = torch.from_numpy(np.array(product_tokens)).long().to(device)

    # Store beliefs for later EPDF generation
    beliefs_np = np.array(belief_states)  # (batch, seq, total_belief_dim)

    # Split beliefs by component
    global_component_belief_arrays = []
    cursor = 0
    for dim in data_source.component_state_dims:
        global_component_belief_arrays.append(beliefs_np[..., cursor: cursor + dim])
        cursor += dim
    component_order = [str(comp.name) for comp in data_source.components]
    component_belief_flat_cache: Dict[str, np.ndarray] = {}
    for comp_name, comp_array in zip(component_order, global_component_belief_arrays):
        component_belief_flat_cache[comp_name] = comp_array.reshape(-1, comp_array.shape[-1])

    # Build component metadata
    component_meta_map: Dict[str, Dict[str, Any]] = {}
    for idx, comp in enumerate(data_source.components):
        comp_type = getattr(comp, "process_type", comp.name.split("_")[0])
        comp_name = str(comp.name)
        vocab_size_comp = int(comp.vocab_size) if hasattr(comp, "vocab_size") else component_summary[idx]["vocab_size"]
        belief_dim = global_component_belief_arrays[idx].shape[-1]
        component_meta_map[comp_name] = {
            "name": comp_name,
            "type": comp_type,
            "vocab_size": vocab_size_comp,
            "belief_dim": belief_dim,
        }
else:
    from multipartite_utils import _sample_tokens
    tokens = _sample_tokens(
        data_source,
        batch_size,
        sample_len,
        seq_len,
        args.seed,
        device,
    )
    global_component_belief_arrays = None
    component_order = []
    component_belief_flat_cache = {}
    component_meta_map = None

with torch.no_grad():
    _, cache = model.run_with_cache(tokens, return_type=None)


#%%
# ==== Run Clustering Pipeline for Each Site ==== #
####################################################
pipeline = SiteClusteringPipeline(
    sae_folder=args.sae_folder,
    site_hook_map=SITE_HOOK_MAP,
)

overall_active_rates: list[np.ndarray] = []
per_site_active_rates: dict[str, np.ndarray] = {}
per_site_cluster_rates: dict[str, dict[int, np.ndarray]] = {}
process_info = {
    "process_kind": process_kind,
    "components": component_summary,
    "process_config": process_cfg_raw,
}

for site_idx, site in enumerate(sites):
    site_selected_k = selected_k_by_site.get(site)
    if site_selected_k is None:
        print(f"Skipping {site}: no selected k available")
        continue

    site_dir = os.path.join(run_dir, site)

    # Create clustering config
    config = ClusteringConfig.from_args(args, site, site_selected_k)

    # Run pipeline
    result = pipeline.run(
        config=config,
        model=model,
        cache=cache,
        data_source=data_source,
        site_dir=site_dir,
        component_beliefs_flat=component_belief_flat_cache,
        component_order=component_order,
        component_metadata=component_meta_map,
        site_idx=site_idx,
        device=device,
    )

    # Collect activity rates for histograms
    site_active_rates = result.activity_rates[result.active_indices]
    per_site_active_rates[site] = site_active_rates
    if site_active_rates.size > 0:
        overall_active_rates.append(site_active_rates)

    # Collect cluster rates
    for cid, stats in result.cluster_stats.items():
        if "activity_rates" in stats:
            latent_ids = stats.get("latent_indices", [])
            rates_array = np.array([result.activity_rates[idx] for idx in latent_ids], dtype=float)
            if rates_array.size > 0:
                if site not in per_site_cluster_rates:
                    per_site_cluster_rates[site] = {}
                per_site_cluster_rates[site][int(cid)] = rates_array

    # Evaluate metrics (GMG, R²)
    # Construct SAE path based on type
    # Construct SAE path based on type
    if args.sae_type == "top_k":
        sae_filename = f"{site}_top_k_k{int(site_selected_k)}.pt"
    elif args.sae_type == "vanilla":
        sae_filename = f"{site}_vanilla_lambda_{site_selected_k}.pt"
    elif args.sae_type == "banded":
        # site_selected_k is a tuple (ls, la)
        ls, la = site_selected_k
        ls_str = str(ls).replace(".", "p")
        la_str = str(la).replace(".", "p")
        sae_filename = f"{site}_banded_ls_{ls_str}__la_{la_str}.pt"
    else:
        raise ValueError(f"Unknown SAE type: {args.sae_type}")
    sae_path = os.path.join(args.sae_folder, sae_filename)

    decoder_dirs = None
    if os.path.exists(sae_path):
        ckpt = torch.load(sae_path, map_location=device, weights_only=False)
        decoder_dirs = ckpt["state_dict"]["W_dec"].cpu().numpy()

    metrics = evaluate_clustering_metrics(result, decoder_dirs=decoder_dirs)
    result.add_metrics(metrics)

    # Geometry-guided refinement (if enabled)
    if config.geometry_fitting_config.enabled:
        from clustering import GeometryRefinementPipeline

        print(f"{site}: running geometry-guided refinement...")

        # Get soft assignments from clustering strategy result
        # These were computed during the clustering step
        soft_assignments = None
        if hasattr(result, '_strategy_result') and result._strategy_result is not None:
            soft_assignments = result._strategy_result.soft_weights

        # If soft weights not stored, we need to re-extract them
        # For now, check if they exist in the internal result storage
        # The pipeline.run() should store the strategy result

        if soft_assignments is None:
            print(f"{site}: warning - no soft assignments found, skipping geometry refinement")
        else:
            # Get active decoder directions
            active_decoder = decoder_dirs[result.active_indices] if decoder_dirs is not None else None

            if active_decoder is None:
                print(f"{site}: warning - decoder directions not available, skipping geometry refinement")
            else:
                refinement_pipeline = GeometryRefinementPipeline(config.geometry_fitting_config)

                try:
                    refined_result = refinement_pipeline.refine_clusters(
                        decoder_dirs=active_decoder,
                        soft_assignments=soft_assignments,
                        verbose=True
                    )
                    result.add_geometry_refinement(refined_result)
                    print(f"{site}: geometry refinement complete")

                    # Compute refined R² and assignment after geometry refinement
                    if component_order and component_belief_flat_cache:
                        from clustering.analysis import ClusterAnalyzer

                        # Need to reconstruct the data that was used in the pipeline
                        # Get the feature activations and other data from result
                        if hasattr(result, '_epdf_data') and result._epdf_data:
                            acts_flat = result._epdf_data['acts_flat']
                            decoder_dirs_full = result._epdf_data['decoder_dirs']

                            # Encode features
                            from mess3_gmg_analysis_utils import sae_encode_features
                            ckpt = torch.load(sae_path, map_location=device, weights_only=False)
                            if args.sae_type == "top_k":
                                from BatchTopK.sae import TopKSAE
                                topk_cfg = dict(ckpt["cfg"])
                                topk_cfg["device"] = "cuda" if device.startswith("cuda") else "cpu"
                                ckpt["cfg"]["device"] = topk_cfg["device"]
                                sae = TopKSAE(topk_cfg).to(device)
                            elif args.sae_type == "vanilla":
                                from BatchTopK.sae import VanillaSAE
                                vanilla_cfg = dict(ckpt["cfg"])
                                vanilla_cfg["device"] = "cuda" if device.startswith("cuda") else "cpu"
                                ckpt["cfg"]["device"] = vanilla_cfg["device"]
                                sae = VanillaSAE(vanilla_cfg).to(device)
                            elif args.sae_type == "banded":
                                banded_cfg = dict(ckpt["cfg"])
                                banded_cfg["device"] = "cuda" if device.startswith("cuda") else "cpu"
                                ckpt["cfg"]["device"] = banded_cfg["device"]
                                sae = BandedCovarianceSAE(banded_cfg).to(device)
                            else:
                                raise ValueError(f"Unknown SAE type: {args.sae_type}")
                            sae.load_state_dict(ckpt["state_dict"])
                            sae.eval()

                            with torch.no_grad():
                                feature_acts, _, _ = sae_encode_features(sae, acts_flat)
                            feature_np = feature_acts.detach().cpu().numpy()

                            analyzer = ClusterAnalyzer(config.analysis_config)

                            # Filter to active latents for refined R²
                            # refined_assignments has shape (n_active_latents, n_clusters)
                            active_feature_np = feature_np[:, result.active_indices]
                            active_decoder_dirs = decoder_dirs_full[result.active_indices]
                            active_cluster_labels = result.cluster_labels[result.active_indices]

                            # Compute refined R²
                            result.belief_r2_summary_refined = analyzer.compute_belief_r2(
                                acts_flat,
                                active_feature_np,
                                active_decoder_dirs,
                                active_cluster_labels,
                                result.n_clusters,
                                component_belief_flat_cache,
                                component_order,
                                config.belief_seeding.ridge_alpha,
                                site,
                                soft_assignments=refined_result.refined_assignments,
                                assignment_name="refined",
                            )

                            # Compute optimal assignment for refined R²
                            if result.belief_r2_summary_refined:
                                result.component_assignment_refined = analyzer.compute_optimal_component_assignment(
                                    result.belief_r2_summary_refined,
                                    component_order,
                                    result.n_clusters,
                                )
                                print(f"{site}: refined R² and optimal assignment computed")

                except Exception as e:
                    print(f"{site}: geometry refinement failed: {e}")
                    import traceback
                    traceback.print_exc()

    if mlflow_enabled:
        if result.metrics:
            metrics_numeric = _extract_numeric(result.metrics)
            if metrics_numeric:
                _log_numeric(f"{site}/metrics", metrics_numeric)

        if result.component_assignment_soft:
            soft_payload = {
                key: value
                for key, value in result.component_assignment_soft.items()
                if key not in {"assignments", "noise_clusters"}
            }
            numeric_soft = _extract_numeric(soft_payload)
            if numeric_soft:
                _log_numeric(f"{site}/component_assignment_soft", numeric_soft)
            if "noise_clusters" in result.component_assignment_soft:
                mlflow_module.log_param(
                    f"{site}_component_assignment_soft_noise_clusters",
                    json.dumps(result.component_assignment_soft["noise_clusters"]),
                )
            if "assignments" in result.component_assignment_soft:
                mlflow_module.log_param(
                    f"{site}_component_assignment_soft_assignments",
                    json.dumps(result.component_assignment_soft["assignments"]),
                )

        if result.component_assignment_refined:
            refined_payload = {
                key: value
                for key, value in result.component_assignment_refined.items()
                if key not in {"assignments", "noise_clusters"}
            }
            numeric_refined = _extract_numeric(refined_payload)
            if numeric_refined:
                _log_numeric(f"{site}/component_assignment_refined", numeric_refined)
            if "noise_clusters" in result.component_assignment_refined:
                mlflow_module.log_param(
                    f"{site}_component_assignment_refined_noise_clusters",
                    json.dumps(result.component_assignment_refined["noise_clusters"]),
                )
            if "assignments" in result.component_assignment_refined:
                mlflow_module.log_param(
                    f"{site}_component_assignment_refined_assignments",
                    json.dumps(result.component_assignment_refined["assignments"]),
                )

        if result.coherence_metrics_soft:
            soft_coherence = _extract_numeric(result.coherence_metrics_soft)
            if soft_coherence:
                _log_numeric(f"{site}/coherence_metrics_soft", soft_coherence)

        if result.subspace_diagnostics:
            within_energy = result.subspace_diagnostics.get("within_projection_energy")
            if isinstance(within_energy, dict):
                within_numeric = _extract_numeric(within_energy)
                if within_numeric:
                    _log_numeric(f"{site}/within_projection_energy", within_numeric)
            between_energy = result.subspace_diagnostics.get("between_projection_energy")
            if isinstance(between_energy, (int, float, np.floating, np.integer)):
                if math.isfinite(float(between_energy)):
                    mlflow_module.log_metric(f"{site}/between_projection_energy", float(between_energy))

    # Save result
    result.save_to_directory(site_dir, process_info, average_l2)

    if mlflow_enabled:
        cluster_summary_path = os.path.join(site_dir, "cluster_summary.json")
        if os.path.exists(cluster_summary_path):
            mlflow_module.log_artifact(cluster_summary_path, artifact_path=site)

    print(f"{site}: saved to {site_dir}")
    if metrics:
        print(f"{site} metrics: {metrics}")


#%%
# ==== Generate Overall Activity Histograms ==== #
##################################################
if overall_active_rates:
    concatenated = np.concatenate([arr for arr in overall_active_rates if arr.size > 0])
    if concatenated.size > 0:
        overall_hist_path = os.path.join(run_dir, "latent_activity_hist_overall.png")
        plot_activity_histogram(
            concatenated,
            overall_hist_path,
            title="Latent activation rates (all sites)",
        )

if per_site_active_rates:
    per_site_hist_path = os.path.join(run_dir, "latent_activity_hist_per_site.png")
    plot_activity_histograms_by_site(
        per_site_active_rates,
        per_site_hist_path,
        suptitle="Latent activation rates by site",
    )

if per_site_cluster_rates and any(per_site_cluster_rates[site] for site in per_site_cluster_rates):
    cluster_hist_path = os.path.join(run_dir, "latent_activity_hist_per_site_cluster.png")
    plot_activity_histograms_site_clusters(
        per_site_cluster_rates,
        cluster_hist_path,
        suptitle="Latent activation rates by site and cluster",
    )

print(f"Analysis complete. Outputs written to {run_dir}")

if mlflow_active_run and mlflow_module is not None and mlflow_module.active_run():
    mlflow_module.end_run()
