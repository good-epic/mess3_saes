#%%
# === Imports and Configuration === #
###################################
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import argparse
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple


import torch
import jax
import jax.numpy as jnp
import numpy as np

from transformer_lens import HookedTransformer, HookedTransformerConfig
from simplexity.generative_processes.torch_generator import generate_data_batch

from BatchTopK.sae import TopKSAE

from training_and_analysis_utils import build_similarity_matrix, spectral_clustering_with_eigengap
from mess3_gmg_analysis_utils import (
    collect_cluster_reconstructions,
    compute_average_l2,
    extract_topk_l2,
    find_elbow_k,
    fit_pca_for_clusters,
    fit_residual_to_belief_map,
    load_metrics_summary,
    lasso_select_latents,
    plot_activity_histogram,
    plot_activity_histograms_by_site,
    plot_activity_histograms_site_clusters,
    plot_cluster_pca,
    plot_l2_bar_chart,
    project_decoder_directions_to_pca,
    sae_encode_features,
    write_cluster_metadata,
)
from subspace_clustering_utils import (
    normalize_and_deduplicate,
    k_subspaces_clustering,
    ensc_clustering,
    grid_search_k_subspaces,
    add_diagnostics_to_result,
    _compute_belief_seed_sets,
)
from multipartite_utils import (
    MultipartiteSampler,
    build_components_from_config,
    _resolve_device,
    _load_process_stack,
    _load_transformer,
    _select_sites,
    _sample_tokens,
    collect_latent_activity_data,
)
from epdf_utils import (
    build_epdfs_from_sae_and_beliefs,
    plot_epdfs_to_directory,
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
    parser.add_argument("--force_k", type=int, default=None, help="Override elbow selection with manual k")
    parser.add_argument("--clustering_method", type=str, default="k_subspaces", choices=["spectral", "k_subspaces", "ensc"], help="Clustering method for decoder directions")
    parser.add_argument("--sim_metric", type=str, default="cosine", choices=["cosine", "euclidean", "phi"], help="Similarity metric for decoder clustering (spectral method only)")
    parser.add_argument("--max_clusters", type=int, default=12, help="Upper bound for eigengap clustering (spectral method only)")
    parser.add_argument("--plot_eigengap", action="store_true", help="Plot eigengap spectrum diagnostics (spectral method only)")
    parser.add_argument("--center_decoder_rows", action="store_true", help="Center and renormalize decoder rows before computing similarities (spectral method only)")

    # Subspace clustering controls
    parser.add_argument("--subspace_rank", type=int, default=None, help="Rank of each subspace for k_subspaces/ensc methods (if None, auto-detected per cluster)")
    parser.add_argument("--subspace_n_clusters", type=int, default=None, help="Number of clusters for subspace methods (if None, auto-detected)")
    parser.add_argument("--use_grid_search", action="store_true", help="Use grid search instead of automatic parameter selection for k_subspaces")
    parser.add_argument("--subspace_k_values", type=int, nargs="+", default=[4, 5, 6, 7, 8], help="K values to try in grid search for k_subspaces (only used with --use_grid_search)")
    parser.add_argument("--subspace_r_values", type=int, nargs="+", default=[2, 3, 4, 5], help="Subspace rank values to try in grid search for k_subspaces (only used with --use_grid_search)")
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

    args, _ = parser.parse_known_args()

    if args.process_config and args.process_preset:
        parser.error("Specify at most one of --process_config or --process_preset")

    return args



#%%
# ==== Setup ==== #
###################

#def main() -> None:
args = _parse_args()
device = _resolve_device(args.device)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

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
# ==== Choose K and Do Clustering ==== #
########################################
l2_by_site = extract_topk_l2(metrics_summary, site_filter=sites)
if not l2_by_site:
    raise ValueError("No top-k SAE metrics found for the selected sites")

average_l2 = compute_average_l2(l2_by_site)
if not average_l2:
    raise ValueError("Unable to compute average L2 across k values")

selected_k_by_site: dict[str, int] = {}
if args.force_k is not None:
    print(f"Using forced k={args.force_k} for all sites")
    selected_k_by_site = {site: int(args.force_k) for site in sites}
else:
    for site in sites:
        site_metrics = l2_by_site.get(site, {})
        if not site_metrics:
            raise ValueError(f"No top-k metrics available for site '{site}'")
        sorted_k = sorted(site_metrics.keys())
        losses = [site_metrics[k] for k in sorted_k]
        site_k = find_elbow_k(sorted_k, losses, prefer_high_k=True)
        selected_k_by_site[site] = int(site_k)
        print(f"{site}: selected k={site_k} via elbow heuristic")

unique_k_values = set(selected_k_by_site.values())
if len(unique_k_values) == 1:
    run_dir_label = f"k{next(iter(unique_k_values))}"
else:
    run_dir_label = "kvar"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(args.output_dir, f"{run_dir_label}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

l2_plot_path = os.path.join(run_dir, "l2_summary.png")
plot_l2_bar_chart(l2_by_site, l2_plot_path)
with open(os.path.join(run_dir, "l2_summary.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "per_site": {site: {int(k): float(v) for k, v in metrics.items()} for site, metrics in l2_by_site.items()},
            "average": {int(k): float(v) for k, v in average_l2.items()},
            "selected_k": int(next(iter(unique_k_values))) if len(unique_k_values) == 1 else None,
            "selected_k_by_site": {site: int(k) for site, k in selected_k_by_site.items()},
            "process_kind": process_kind,
            "components": component_summary,
        },
        f,
        indent=2,
    )

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
else:
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

with torch.no_grad():
    _, cache = model.run_with_cache(tokens, return_type=None)

overall_active_rates: list[np.ndarray] = []
per_site_active_rates: dict[str, np.ndarray] = {}
per_site_cluster_rates: dict[str, dict[int, np.ndarray]] = {}

for site_idx, site in enumerate(sites):
    site_selected_k = selected_k_by_site.get(site)
    if site_selected_k is None:
        print(f"Skipping {site}: no selected k available")
        continue

    sae_path = os.path.join(args.sae_folder, f"{site}_top_k_k{site_selected_k}.pt")
    if not os.path.exists(sae_path):
        print(f"Skipping {site}: checkpoint not found at {sae_path}")
        continue

    ckpt = torch.load(sae_path, map_location=device, weights_only=False)
    sae = TopKSAE(ckpt["cfg"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])  # type: ignore
    sae.eval()

    site_dir = os.path.join(run_dir, site)
    os.makedirs(site_dir, exist_ok=True)

    decoder_dirs = sae.W_dec.detach().cpu().numpy()

    hook_name = SITE_HOOK_MAP[site]
    activity_stats = collect_latent_activity_data(
        model,
        sae,
        data_source,
        hook_name,
        batch_size=args.sample_sequences,
        sample_len=sample_len,
        target_len=seq_len,
        n_batches=args.activation_batches,
        seed=args.seed + site_idx * (args.activation_batches + 1),
        device=device,
        activation_eps=args.latent_activation_eps,
        collect_matrix=(args.sim_metric == "phi"),
    )
    activity_rates = activity_stats["activity_rates"]
    mean_abs_activation = activity_stats["mean_abs_activation"]
    latent_activity_matrix = activity_stats["latent_matrix"]
    total_activity_samples = int(activity_stats["total_samples"])

    acts = cache[hook_name].detach()
    acts_flat = acts.reshape(-1, acts.shape[-1]).to(device)
    subsample_idx = None
    if args.max_activations and acts_flat.shape[0] > args.max_activations:
        idx = torch.randperm(acts_flat.shape[0], device=acts_flat.device)[: args.max_activations]
        acts_flat = acts_flat[idx].contiguous()
        subsample_idx = idx.cpu().numpy()  # Save for belief subsampling
    else:
        acts_flat = acts_flat.contiguous()

    with torch.no_grad():
        feature_acts, x_mean, x_std = sae_encode_features(sae, acts_flat)
    feature_acts = feature_acts.detach()
    active_mask = activity_rates >= args.latent_activity_threshold
    active_indices = np.nonzero(active_mask)[0]
    inactive_indices = np.where(~active_mask)[0]
    per_site_cluster_rates[site] = {}
    site_active_rates = activity_rates[active_indices]
    per_site_active_rates[site] = site_active_rates
    if site_active_rates.size > 0:
        overall_active_rates.append(site_active_rates)
    print(
        f"{site}: {active_indices.size}/{decoder_dirs.shape[0]} latents pass activity threshold "
        f"{args.latent_activity_threshold:.2%}"
    )
    cluster_labels_full = np.full(decoder_dirs.shape[0], -1, dtype=int)

    if active_indices.size == 0:
        metadata_path = os.path.join(site_dir, "cluster_summary.json")
        extra_fields_no_active = {
            "site": site,
            "spectral_clusters": 0,
            "n_activations_used": int(acts_flat.shape[0]),
            "cluster_activation_threshold": float(args.cluster_activation_threshold),
            "min_cluster_samples": int(args.min_cluster_samples),
            "clusters_with_pca": [],
            "process_kind": process_kind,
            "components": component_summary,
            "process_config": process_cfg_raw,
            "latent_activity_rates": {int(i): float(val) for i, val in enumerate(activity_rates)},
            "mean_abs_activation": {int(i): float(val) for i, val in enumerate(mean_abs_activation)},
            "active_latent_indices": [],
            "inactive_latent_indices": [int(i) for i in inactive_indices.tolist()],
            "latent_activity_threshold": float(args.latent_activity_threshold),
            "latent_activation_eps": float(args.latent_activation_eps),
            "activation_samples": total_activity_samples,
            "decoder_rows_centered": bool(args.center_decoder_rows),
            "clustering_method": args.clustering_method,
            "belief_seeding_used": False,
        }
        write_cluster_metadata(
            metadata_path,
            {},
            site_selected_k,
            average_l2,
            cluster_labels=cluster_labels_full.tolist(),
            extra_fields=extra_fields_no_active,
        )
        del feature_acts
        continue

    decoder_active = decoder_dirs[active_indices]

    belief_seed_clusters_active: Dict[int, List[int]] = {}
    belief_seed_clusters_global: Dict[int, List[int]] = {}
    belief_seed_metadata: Dict[str, Any] | None = None
    belief_slice_map: Dict[str, Tuple[int, int]] | None = None
    belief_seeding_used = False

    if args.seed_with_beliefs and component_belief_flat_cache:
        print(f"{site}: doing belief-aligned seeding")
        component_beliefs_flat = {}
        for comp_name in component_order:
            comp_flat = component_belief_flat_cache[comp_name]
            if subsample_idx is not None:
                comp_flat = comp_flat[subsample_idx]
            component_beliefs_flat[comp_name] = comp_flat

        active_idx_tensor = torch.from_numpy(active_indices).to(feature_acts.device)
        feature_active = feature_acts[:, active_idx_tensor]
        try:
            seeds_active_map, seed_metadata, residual_map, residual_intercept, slice_map = _compute_belief_seed_sets(
                acts_flat,
                feature_active,
                decoder_active,
                component_beliefs_flat,
                ridge_alpha=args.belief_ridge_alpha,
                lasso_alpha=args.seed_lasso_alpha,
                lasso_cv=args.seed_lasso_cv,
                lasso_max_iter=args.seed_lasso_max_iter,
                coef_threshold=args.seed_coef_threshold,
                max_latents=args.seed_max_latents,
                random_state=args.seed + site_idx,
            )
        except Exception as exc:
            print(f"{site}: belief-aligned seeding failed ({exc})")
            seeds_active_map = {}
            seed_metadata = {}
            slice_map = {}
        if seeds_active_map:
            latent_best_component: Dict[int, Tuple[str, float]] = {}
            for comp_name, meta in seed_metadata.items():
                for latent_idx, score in meta.get("scores", {}).items():
                    latent_idx = int(latent_idx)
                    score_val = float(score)
                    best = latent_best_component.get(latent_idx)
                    if best is None or score_val > best[1]:
                        latent_best_component[latent_idx] = (comp_name, score_val)

            resolved_seed_map: Dict[str, List[int]] = {}
            for comp_name in component_order:
                candidates = seeds_active_map.get(comp_name, [])
                resolved: List[int] = []
                for idx in candidates:
                    best = latent_best_component.get(idx)
                    if best is None or best[0] != comp_name:
                        continue
                    if idx not in resolved:
                        resolved.append(idx)
                if resolved:
                    resolved_seed_map[comp_name] = resolved

            if resolved_seed_map:
                belief_seeding_used = True
                cluster_component_map: Dict[int, str] = {}
                belief_seed_metadata = {
                    "per_component": seed_metadata,
                    "resolved": {
                        comp: [int(active_indices[idx]) for idx in indices]
                        for comp, indices in resolved_seed_map.items()
                    },
                    "ridge_alpha": float(args.belief_ridge_alpha),
                    "lasso_alpha_override": None if args.seed_lasso_alpha is None else float(args.seed_lasso_alpha),
                }
                belief_slice_map = slice_map

                cluster_counter = 0
                for comp_name in component_order:
                    indices = resolved_seed_map.get(comp_name, [])
                    if not indices:
                        continue
                    belief_seed_clusters_active[cluster_counter] = indices
                    belief_seed_clusters_global[cluster_counter] = [int(active_indices[idx]) for idx in indices]
                    cluster_component_map[cluster_counter] = comp_name
                    cluster_counter += 1
                belief_seed_metadata["cluster_components"] = {
                    int(cid): name for cid, name in cluster_component_map.items()
                }
            else:
                print(f"{site}: belief-aligned seeding produced no confident latents")

    protected_positions = None
    if args.protect_seed_duplicates and belief_seed_clusters_active:
        protected_positions = sorted({int(idx) for indices in belief_seed_clusters_active.values() for idx in indices})

    # Variables to track deduplication (only used for subspace methods)
    decoder_normalized = None
    normalized_to_full_idx = None

    # Handle trivial case: single active latent
    if decoder_active.shape[0] == 1:
        cluster_labels_active = np.array([0], dtype=int)
        spectral_k = 1
        subspace_result = None
    else:
        # Clustering dispatch based on method
        if args.clustering_method == "spectral":
            if belief_seeding_used:
                print(f"{site}: belief-aligned seeds are not applied for spectral clustering")
                belief_seeding_used = False
            # Original spectral clustering path
            if args.center_decoder_rows and decoder_active.size > 0:
                row_mean = decoder_active.mean(axis=0, keepdims=True)
                decoder_active = decoder_active - row_mean
                norms = np.linalg.norm(decoder_active, axis=1, keepdims=True)
                norms = np.where(norms == 0.0, 1.0, norms)
                decoder_active = decoder_active / norms

            if args.sim_metric == "phi":
                if latent_activity_matrix is None:
                    raise ValueError("Phi similarity requires latent activation matrix; enable collect_matrix")
                latent_active = latent_activity_matrix[:, active_indices]
                sim_matrix = build_similarity_matrix(
                    decoder_active,
                    method="phi",
                    latent_acts=latent_active,
                )
                sim_matrix = (sim_matrix + 1.0) / 2.0
                np.fill_diagonal(sim_matrix, 1.0)
                latent_activity_matrix = None
            else:
                sim_matrix = build_similarity_matrix(decoder_active, method=args.sim_metric)

            eig_plot = None
            if args.plot_eigengap:
                eig_plot = os.path.join(site_dir, "eigengap.png")
            cluster_labels_active, spectral_k = spectral_clustering_with_eigengap(
                sim_matrix,
                max_clusters=min(args.max_clusters, decoder_active.shape[0]),
                random_state=args.seed,
                plot=args.plot_eigengap,
                plot_path=eig_plot,
            )
            cluster_labels_active = np.asarray(cluster_labels_active, dtype=int)
            subspace_result = None

        elif args.clustering_method in ["k_subspaces", "ensc"]:
            # Preprocess: normalize and deduplicate
            decoder_normalized, kept_indices = normalize_and_deduplicate(
                decoder_active,
                cosine_threshold=args.cosine_dedup_threshold,
                protected_indices=protected_positions if protected_positions else None,
            )

            # Map kept_indices back to active_indices space
            kept_active_indices = active_indices[kept_indices]

            # Create mapping from normalized indices to full decoder indices
            # This is the ONLY valid way to map back to original latent IDs after deduplication
            normalized_to_full_idx = active_indices[kept_indices]

            initial_clusters_normalized: Dict[int, List[int]] = {}
            if belief_seeding_used and belief_seed_clusters_active:
                index_map = {int(active_idx): int(norm_idx) for norm_idx, active_idx in enumerate(kept_indices)}
                updated_global: Dict[int, List[int]] = {}
                for cluster_id, active_list in belief_seed_clusters_active.items():
                    normalized_list = [index_map[int(idx)] for idx in active_list if int(idx) in index_map]
                    if not normalized_list:
                        continue
                    initial_clusters_normalized[cluster_id] = normalized_list
                    updated_global[cluster_id] = [int(active_indices[idx]) for idx in active_list if int(idx) in index_map]
                if not initial_clusters_normalized:
                    print(f"{site}: belief-aligned seeds were removed during deduplication")
                    belief_seeding_used = False
                else:
                    belief_seed_clusters_global = updated_global
                    if belief_seed_metadata is not None:
                        belief_seed_metadata["post_dedup"] = {
                            int(cid): latents for cid, latents in updated_global.items()
                        }

            print(f"{site}: after deduplication, kept {len(decoder_normalized)}/{len(decoder_active)} active latents")

            if len(decoder_normalized) == 0:
                cluster_labels_active = np.array([], dtype=int)
                spectral_k = 0
                subspace_result = None
            elif len(decoder_normalized) == 1:
                cluster_labels_active = np.array([0], dtype=int)
                spectral_k = 1
                subspace_result = None
            else:
                # Run subspace clustering
                if args.clustering_method == "k_subspaces":
                    if args.use_grid_search and belief_seeding_used:
                        raise ValueError("Belief-aligned seeding is not currently compatible with --use_grid_search")
                    effective_n_clusters = args.subspace_n_clusters
                    override_reason = None
                    if belief_seeding_used and initial_clusters_normalized:
                        num_seed_clusters = len(initial_clusters_normalized)
                        seeded_latent_total = sum(len(v) for v in initial_clusters_normalized.values())
                        total_components = len(component_order)
                        if total_components and num_seed_clusters == total_components:
                            effective_n_clusters = num_seed_clusters
                            override_reason = "all components seeded"
                        elif (
                            seeded_latent_total >= len(decoder_normalized)
                            and total_components
                            and num_seed_clusters < total_components
                        ):
                            # Future idea: force empty component clusters by seeding their top ridge latent.
                            effective_n_clusters = num_seed_clusters
                            override_reason = "all latents covered by seeded components"
                        if override_reason is not None:
                            print(
                                f"{site}: overriding k_subspaces cluster count to {effective_n_clusters} ({override_reason})"
                            )
                    if args.use_grid_search:
                        # Grid search over K and r
                        subspace_result, grid_results = grid_search_k_subspaces(
                            decoder_normalized,
                            k_values=args.subspace_k_values,
                            r_values=args.subspace_r_values,
                            random_state=args.seed,
                            bic_penalty_weight=0.1,
                        )
                        print(
                            f"{site}: grid search selected K={subspace_result.n_clusters}, "
                            f"r={subspace_result.subspace_rank} "
                            f"(total error={subspace_result.total_reconstruction_error:.4f})"
                        )
                    else:
                        # Automatic or manual parameter selection
                        subspace_result = k_subspaces_clustering(
                            decoder_normalized,
                            n_clusters=effective_n_clusters,  # Can be None for auto
                            subspace_rank=args.subspace_rank,     # Can be None for auto
                            max_iters=20,
                            random_state=args.seed,
                            initial_clusters=initial_clusters_normalized if belief_seeding_used else None,
                            lock_mode=args.seed_lock_mode,
                        )
                        if args.subspace_n_clusters is None or args.subspace_rank is None:
                            auto_info = []
                            if args.subspace_n_clusters is None:
                                auto_info.append(f"K={subspace_result.n_clusters} (auto)")
                            else:
                                auto_info.append(f"K={subspace_result.n_clusters}")
                            if args.subspace_rank is None and subspace_result.cluster_ranks:
                                ranks_str = ",".join(str(subspace_result.cluster_ranks[i]) for i in sorted(subspace_result.cluster_ranks.keys()))
                                auto_info.append(f"r=[{ranks_str}] (auto per-cluster)")
                            elif args.subspace_rank is not None:
                                auto_info.append(f"r={subspace_result.subspace_rank}")
                            print(f"{site}: {', '.join(auto_info)}, error={subspace_result.total_reconstruction_error:.4f}")

                elif args.clustering_method == "ensc":
                    if belief_seeding_used:
                        print(f"{site}: belief-aligned seeds are not applied for ENSC clustering")
                        belief_seeding_used = False
                    # ENSC with automatic or manual parameter selection
                    subspace_result = ensc_clustering(
                        decoder_normalized,
                        n_clusters=args.subspace_n_clusters,  # Can be None for auto (eigengap)
                        subspace_rank=args.subspace_rank,     # Can be None for auto
                        lambda_1=args.ensc_lambda1,
                        lambda_2=args.ensc_lambda2,
                        random_state=args.seed,
                    )
                    if args.subspace_n_clusters is None or args.subspace_rank is None:
                        auto_info = []
                        if args.subspace_n_clusters is None:
                            auto_info.append(f"K={subspace_result.n_clusters} (auto eigengap)")
                        else:
                            auto_info.append(f"K={subspace_result.n_clusters}")
                        if args.subspace_rank is None and subspace_result.cluster_ranks:
                            ranks_str = ",".join(str(subspace_result.cluster_ranks[i]) for i in sorted(subspace_result.cluster_ranks.keys()))
                            auto_info.append(f"r=[{ranks_str}] (auto per-cluster)")
                        elif args.subspace_rank is not None:
                            auto_info.append(f"r={subspace_result.subspace_rank}")
                        print(f"{site}: {', '.join(auto_info)}, error={subspace_result.total_reconstruction_error:.4f}")

                # Add diagnostics
                subspace_result = add_diagnostics_to_result(subspace_result, decoder_normalized)

                # Map labels back to full active set (including duplicates)
                cluster_labels_active = np.full(len(decoder_active), -1, dtype=int)
                cluster_labels_active[kept_indices] = subspace_result.cluster_labels

                # Assign duplicates to nearest kept point's cluster
                for dup_idx in range(len(decoder_active)):
                    if cluster_labels_active[dup_idx] == -1:
                        # Find most similar kept vector
                        dup_vec = decoder_active[dup_idx]
                        similarities = np.abs(np.dot(decoder_normalized, dup_vec))
                        nearest_kept = np.argmax(similarities)
                        cluster_labels_active[dup_idx] = subspace_result.cluster_labels[nearest_kept]

                spectral_k = subspace_result.n_clusters

                # Print diagnostics
                if subspace_result.principal_angles is not None:
                    min_angles_deg = {
                        pair: float(np.rad2deg(angles.min()))
                        for pair, angles in subspace_result.principal_angles.items()
                    }
                    overall_min = min(min_angles_deg.values()) if min_angles_deg else 0.0
                    print(f"{site}: minimum principal angle = {overall_min:.1f}°")

                if subspace_result.within_projection_energy is not None and subspace_result.between_projection_energy is not None:
                    mean_within = np.mean(list(subspace_result.within_projection_energy.values()))
                    ratio = mean_within / subspace_result.between_projection_energy if subspace_result.between_projection_energy > 0 else float('inf')
                    print(f"{site}: within/between energy ratio = {ratio:.2f}")
        else:
            raise ValueError(f"Unknown clustering method: {args.clustering_method}")

    cluster_labels_full[active_indices] = cluster_labels_active

    if belief_seeding_used and belief_seed_clusters_global:
        for cid in sorted(belief_seed_clusters_global.keys()):
            seeded_latents = belief_seed_clusters_global[cid]
            label_counts = Counter()
            for latent_idx in seeded_latents:
                if 0 <= latent_idx < len(cluster_labels_full):
                    label_counts[cluster_labels_full[latent_idx]] += 1
            assigned = label_counts.get(cid, 0)
            total_seeded = len(seeded_latents)
            reassigned = {
                int(k): int(v)
                for k, v in label_counts.items()
                if k != cid and v > 0
            }
            if reassigned:
                print(
                    f"{site}: belief-seeded cluster {cid} retained {assigned}/{total_seeded} preselected latents (reassigned: {reassigned})"
                )
            else:
                print(
                    f"{site}: belief-seeded cluster {cid} retained {assigned}/{total_seeded} preselected latents"
                )

    print(
        f"{site}: clustered {decoder_active.shape[0]} active latents into {spectral_k} groups"
    )

    # Build subspace diagnostics for metadata
    subspace_diagnostics = {}
    if subspace_result is not None:
        subspace_diagnostics["clustering_method"] = args.clustering_method
        subspace_diagnostics["subspace_rank"] = int(subspace_result.subspace_rank) if subspace_result.subspace_rank is not None else None
        subspace_diagnostics["subspace_reconstruction_error"] = float(subspace_result.total_reconstruction_error)

        # Add cluster-specific ranks if available
        if subspace_result.cluster_ranks is not None:
            subspace_diagnostics["cluster_ranks"] = {
                int(k): int(v) for k, v in subspace_result.cluster_ranks.items()
            }

        # Record whether automatic selection was used
        subspace_diagnostics["n_clusters_auto"] = args.subspace_n_clusters is None and not args.use_grid_search
        subspace_diagnostics["subspace_rank_auto"] = args.subspace_rank is None and not args.use_grid_search
        subspace_diagnostics["used_grid_search"] = args.use_grid_search

        if subspace_result.principal_angles is not None:
            # Convert principal angles to degrees for readability
            principal_angles_deg = {}
            min_principal_angles = {}
            for (ci, cj), angles in subspace_result.principal_angles.items():
                key = f"({ci},{cj})"
                principal_angles_deg[key] = [float(np.rad2deg(a)) for a in angles]
                min_principal_angles[key] = float(np.rad2deg(angles.min()))

            subspace_diagnostics["principal_angles_deg"] = principal_angles_deg
            subspace_diagnostics["min_principal_angles_deg"] = min_principal_angles
            subspace_diagnostics["overall_min_principal_angle_deg"] = float(min(min_principal_angles.values())) if min_principal_angles else None

        if subspace_result.within_projection_energy is not None:
            subspace_diagnostics["within_projection_energy"] = {
                int(k): float(v) for k, v in subspace_result.within_projection_energy.items()
            }

        if subspace_result.between_projection_energy is not None:
            subspace_diagnostics["between_projection_energy"] = float(subspace_result.between_projection_energy)

            if subspace_result.within_projection_energy is not None:
                mean_within = np.mean(list(subspace_result.within_projection_energy.values()))
                ratio = mean_within / subspace_result.between_projection_energy if subspace_result.between_projection_energy > 0 else float('inf')
                subspace_diagnostics["energy_contrast_ratio"] = float(ratio)
    else:
        subspace_diagnostics["clustering_method"] = args.clustering_method

    encoded_cache = (feature_acts, x_mean, x_std)
    with torch.no_grad():
        cluster_recons, cluster_stats = collect_cluster_reconstructions(
            acts_flat,
            sae,
            cluster_labels_full.tolist(),
            min_activation=args.cluster_activation_threshold,
            encoded_cache=encoded_cache,
        )
    feature_np_for_r2 = feature_acts.detach().cpu().numpy()
    del feature_acts
    del encoded_cache
    del x_mean, x_std

    cluster_r2_summary: Dict[int, Dict[str, Any]] | None = None
    if component_belief_flat_cache:
        try:
            component_beliefs_for_scoring: Dict[str, np.ndarray] = {}
            for comp_name in component_order:
                comp_flat = component_belief_flat_cache[comp_name]
                if subsample_idx is not None:
                    comp_flat = comp_flat[subsample_idx]
                component_beliefs_for_scoring[comp_name] = comp_flat

            acts_np = acts_flat.detach().cpu().numpy()
            beliefs_concat = np.concatenate(list(component_beliefs_for_scoring.values()), axis=1)
            readout_coef, readout_intercept = fit_residual_to_belief_map(
                acts_np,
                beliefs_concat,
                alpha=args.belief_ridge_alpha,
            )
            if decoder_dirs.shape[1] != readout_coef.shape[0]:
                raise ValueError(
                    f"Decoder width {decoder_dirs.shape[1]} must equal residual_to_belief rows {readout_coef.shape[0]}"
                )

            latent_sens_full = decoder_dirs @ readout_coef

            component_slices: Dict[str, Tuple[int, int]] = {}
            offset = 0
            for comp_name, comp_matrix in component_beliefs_for_scoring.items():
                comp_dim = comp_matrix.shape[1]
                component_slices[comp_name] = (offset, offset + comp_dim)
                offset += comp_dim

            cluster_r2_summary = {}
            for cluster_id in range(int(spectral_k)):
                latent_ids = np.where(cluster_labels_full == cluster_id)[0]
                if latent_ids.size == 0:
                    continue
                cluster_entry: Dict[str, Any] = {}
                cluster_features = feature_np_for_r2[:, latent_ids]
                for comp_name in component_order:
                    start, end = component_slices[comp_name]
                    comp_targets = component_beliefs_for_scoring[comp_name]
                    comp_targets_centered = comp_targets - readout_intercept[start:end]
                    cluster_sens = latent_sens_full[latent_ids][:, start:end]
                    preds = cluster_features @ cluster_sens
                    residual = comp_targets_centered - preds
                    ss_res = np.sum(residual ** 2, axis=0)
                    centered = comp_targets_centered - comp_targets_centered.mean(axis=0, keepdims=True)
                    ss_tot = np.sum(centered ** 2, axis=0)
                    denom = np.where(ss_tot == 0.0, 1.0, ss_tot)
                    r2 = 1.0 - ss_res / denom
                    cluster_entry[comp_name] = {
                        "per_dimension": r2.tolist(),
                        "mean_r2": float(np.mean(r2)),
                    }
                if cluster_entry:
                    cluster_r2_summary[int(cluster_id)] = cluster_entry
        except Exception as exc:
            print(f"{site}: failed to compute belief R^2 diagnostics ({exc})")
            cluster_r2_summary = None

    if not cluster_stats:
        print(f"{site}: no cluster activations above threshold {args.cluster_activation_threshold}; skipping PCA plots")
        metadata_path = os.path.join(site_dir, "cluster_summary.json")
        extra_fields_no_cluster_stats = {
            "site": site,
            "spectral_clusters": int(spectral_k),
            "n_activations_used": int(acts_flat.shape[0]),
            "cluster_activation_threshold": float(args.cluster_activation_threshold),
            "min_cluster_samples": int(args.min_cluster_samples),
            "clusters_with_pca": [],
            "process_kind": process_kind,
            "components": component_summary,
            "process_config": process_cfg_raw,
            "latent_activity_rates": {int(i): float(val) for i, val in enumerate(activity_rates)},
            "mean_abs_activation": {int(i): float(val) for i, val in enumerate(mean_abs_activation)},
            "active_latent_indices": [int(i) for i in active_indices.tolist()],
            "inactive_latent_indices": [int(i) for i in inactive_indices.tolist()],
            "latent_activity_threshold": float(args.latent_activity_threshold),
            "latent_activation_eps": float(args.latent_activation_eps),
            "activation_samples": total_activity_samples,
            "decoder_rows_centered": bool(args.center_decoder_rows),
            "belief_seeding_used": bool(belief_seeding_used),
        }
        if belief_seed_metadata is not None:
            belief_seed_metadata["applied"] = bool(belief_seeding_used)
            extra_fields_no_cluster_stats["belief_seed_metadata"] = belief_seed_metadata
        if cluster_r2_summary is not None:
            extra_fields_no_cluster_stats["belief_cluster_r2"] = cluster_r2_summary
        extra_fields_no_cluster_stats.update(subspace_diagnostics)
        write_cluster_metadata(
            metadata_path,
            cluster_stats,
            site_selected_k,
            average_l2,
            cluster_labels=cluster_labels_full.tolist(),
            extra_fields=extra_fields_no_cluster_stats,
        )
        continue

    for cid, stats in cluster_stats.items():
        latent_ids = stats.get("latent_indices", [])
        stats["activity_rates"] = {int(idx): float(activity_rates[idx]) for idx in latent_ids}
        stats["mean_abs_activation"] = {int(idx): float(mean_abs_activation[idx]) for idx in latent_ids}
        rates_array = np.array([activity_rates[idx] for idx in latent_ids], dtype=float)
        if rates_array.size > 0:
            per_site_cluster_rates[site][int(cid)] = rates_array

    clusters_with_pca = []
    if not args.skip_pca_plots:
        pca_results = fit_pca_for_clusters(
            cluster_recons,
            n_components=args.pca_components,
            min_samples=args.min_cluster_samples,
        )
        project_decoder_directions_to_pca(sae, pca_results, cluster_stats)

        for cid, result in pca_results.items():
            clusters_with_pca.append(int(cid))
            stats = cluster_stats.get(cid)
            if stats is not None:
                stats["explained_variance_ratio"] = [float(x) for x in result.pca.explained_variance_ratio_]
                stats["decoder_scale_factor"] = float(result.scale_factor)
                stats["decoder_projections_selected_pcs"] = {
                    int(latent_idx): [float(x) for x in vec.tolist()]
                    for latent_idx, vec in result.decoder_coords.items()
                }
            plot_path = os.path.join(site_dir, f"cluster_{cid}_pca.png")
            try:
                plot_cluster_pca(
                    site,
                    site_selected_k,
                    cid,
                    result,
                    plot_path,
                    max_points=args.plot_max_points,
                    random_state=args.seed,
                )
            except ValueError as exc:
                print(f"Skipping PCA plot for {site} cluster {cid}: {exc}")
    else:
        print(f"{site}: skipping PCA plots (--skip_pca_plots enabled)")

    # Generate cluster EPDFs if requested
    if args.build_cluster_epdfs and (args.epdf_sites is None or site in args.epdf_sites):
        print(f"{site}: generating cluster EPDFs")

        # Use the pre-sampled component beliefs that correspond to the tokens
        if global_component_belief_arrays is not None:
            # Flatten component beliefs to match acts_flat structure
            flattened_component_beliefs: dict[str, np.ndarray] = {}
            for idx, comp in enumerate(data_source.components):
                comp_name = str(comp.name)
                beliefs = global_component_belief_arrays[idx]  # (batch, seq, belief_dim)
                beliefs_flat = beliefs.reshape(-1, beliefs.shape[-1])

                # Apply same subsampling as acts_flat if it was subsampled
                if subsample_idx is not None:
                    beliefs_flat = beliefs_flat[subsample_idx]

                flattened_component_beliefs[comp_name] = beliefs_flat

            # Build component metadata mapping
            component_meta_map: dict[str, dict[str, Any]] = {}
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

            component_order = [str(comp.name) for comp in data_source.components]
        else:
            print(f"{site}: skipping cluster EPDFs (not a multipartite sampler)")
            component_meta_map = None

        if component_meta_map is not None:
            # Convert bandwidth to appropriate type
            try:
                epdf_bw = float(args.epdf_bandwidth)
            except ValueError:
                epdf_bw = args.epdf_bandwidth  # Keep as string (e.g., "scott")

            # Generate EPDFs for each cluster
            for cluster_id in range(spectral_k):
                cluster_mask = cluster_labels_full == cluster_id
                cluster_latent_indices = np.where(cluster_mask)[0].tolist()

                if not cluster_latent_indices:
                    continue

                print(f"{site}: cluster {cluster_id} - building EPDFs for {len(cluster_latent_indices)} latents")

                try:
                    epdfs = build_epdfs_from_sae_and_beliefs(
                        site_name=site,
                        sae_id=("top_k", f"k{site_selected_k}"),
                        sae=sae,
                        activations=acts_flat,
                        component_beliefs=flattened_component_beliefs,
                        component_metadata=component_meta_map,
                        latent_indices=cluster_latent_indices,
                        activation_threshold=args.cluster_activation_threshold,
                        min_active_samples=args.epdf_min_samples,
                        bw_method=epdf_bw,
                        progress=True,
                        progress_desc=f"{site} cluster {cluster_id}",
                    )

                    # Print diagnostic information about activation statistics
                    print(f"{site} cluster {cluster_id} activation statistics:")
                    for latent_idx, epdf in epdfs.items():
                        activation_frac = epdf.activation_fraction
                        n_samples_used = int(activation_frac * len(acts_flat))
                        print(f"  Latent {latent_idx}: {activation_frac:.1%} active ({n_samples_used} samples)")

                    # Compute cosine similarities within cluster
                    # Use deduplicated vectors if available (subspace methods), otherwise use full decoder
                    if len(cluster_latent_indices) > 1:
                        from sklearn.metrics.pairwise import cosine_similarity

                        if decoder_normalized is not None and normalized_to_full_idx is not None:
                            # Subspace clustering: use only non-deduplicated vectors
                            # Find which cluster_latent_indices are in normalized_to_full_idx (i.e., were kept)
                            kept_mask = np.isin(cluster_latent_indices, normalized_to_full_idx)
                            kept_in_cluster = np.array(cluster_latent_indices)[kept_mask]

                            if len(kept_in_cluster) > 1:
                                # Map full indices to normalized indices
                                full_to_normalized = {full_idx: norm_idx for norm_idx, full_idx in enumerate(normalized_to_full_idx)}
                                normalized_indices = [full_to_normalized[idx] for idx in kept_in_cluster]

                                # Compute similarities using decoder_normalized
                                cluster_decoders = decoder_normalized[normalized_indices]
                                cos_sim = cosine_similarity(cluster_decoders)

                                # Build list of all pairs with their similarities
                                all_pairs = []
                                for i in range(len(kept_in_cluster)):
                                    for j in range(i+1, len(kept_in_cluster)):
                                        full_idx_i = kept_in_cluster[i]
                                        full_idx_j = kept_in_cluster[j]
                                        all_pairs.append((full_idx_i, full_idx_j, cos_sim[i, j]))

                                # Sort by similarity descending and show top 5
                                all_pairs.sort(key=lambda x: x[2], reverse=True)
                                print(f"  Top 5 cosine similarities within cluster {cluster_id} (non-deduplicated vectors only):")
                                for idx_i, idx_j, sim in all_pairs[:5]:
                                    print(f"    Latent {idx_i} & {idx_j}: {sim:.4f}")

                                # Also warn about suspicious pairs > 0.99
                                suspicious_pairs = [(i, j, s) for i, j, s in all_pairs if s > 0.99]
                                if suspicious_pairs:
                                    print(f"  WARNING: Found {len(suspicious_pairs)} pairs with similarity > 0.99:")
                                    for idx_i, idx_j, sim in suspicious_pairs:
                                        print(f"    Latent {idx_i} & {idx_j}: {sim:.4f}")
                            else:
                                print(f"  Cluster {cluster_id} has only {len(kept_in_cluster)} non-deduplicated vector(s)")
                        else:
                            # Spectral clustering: use decoder_dirs directly (no deduplication)
                            cluster_decoders = decoder_dirs[cluster_latent_indices]
                            cos_sim = cosine_similarity(cluster_decoders)

                            # Build list of all pairs
                            all_pairs = []
                            for i in range(len(cluster_latent_indices)):
                                for j in range(i+1, len(cluster_latent_indices)):
                                    idx_i = cluster_latent_indices[i]
                                    idx_j = cluster_latent_indices[j]
                                    all_pairs.append((idx_i, idx_j, cos_sim[i, j]))

                            # Sort and show top 5
                            all_pairs.sort(key=lambda x: x[2], reverse=True)
                            print(f"  Top 5 cosine similarities within cluster {cluster_id}:")
                            for idx_i, idx_j, sim in all_pairs[:5]:
                                print(f"    Latent {idx_i} & {idx_j}: {sim:.4f}")

                            # Warn about suspicious pairs
                            suspicious_pairs = [(i, j, s) for i, j, s in all_pairs if s > 0.99]
                            if suspicious_pairs:
                                print(f"  WARNING: Found {len(suspicious_pairs)} pairs with similarity > 0.99:")
                                for idx_i, idx_j, sim in suspicious_pairs:
                                    print(f"    Latent {idx_i} & {idx_j}: {sim:.4f}")

                    # Plot to directory
                    epdf_cluster_dir = os.path.join(
                        site_dir, "epdfs", args.clustering_method, f"cluster_{cluster_id}"
                    )
                    plot_epdfs_to_directory(
                        epdfs,
                        epdf_cluster_dir,
                        component_order,
                        plot_mode=args.epdf_plot_mode,
                        grid_size=args.epdf_grid_size,
                        title_prefix=f"{site} cluster {cluster_id}: ",
                    )
                    print(f"{site}: cluster {cluster_id} EPDFs saved to {epdf_cluster_dir}")

                except Exception as exc:
                    print(f"{site}: failed to generate EPDFs for cluster {cluster_id}: {exc}")

    metadata_path = os.path.join(site_dir, "cluster_summary.json")
    extra_fields_final = {
        "site": site,
        "spectral_clusters": int(spectral_k),
        "n_activations_used": int(acts_flat.shape[0]),
        "cluster_activation_threshold": float(args.cluster_activation_threshold),
        "min_cluster_samples": int(args.min_cluster_samples),
        "clusters_with_pca": clusters_with_pca,
        "process_kind": process_kind,
        "components": component_summary,
        "process_config": process_cfg_raw,
        "latent_activity_rates": {int(i): float(val) for i, val in enumerate(activity_rates)},
        "mean_abs_activation": {int(i): float(val) for i, val in enumerate(mean_abs_activation)},
        "active_latent_indices": [int(i) for i in active_indices.tolist()],
        "inactive_latent_indices": [int(i) for i in inactive_indices.tolist()],
        "latent_activity_threshold": float(args.latent_activity_threshold),
        "latent_activation_eps": float(args.latent_activation_eps),
        "activation_samples": total_activity_samples,
        "decoder_rows_centered": bool(args.center_decoder_rows),
        "belief_seeding_used": bool(belief_seeding_used),
    }
    if belief_seed_metadata is not None:
        belief_seed_metadata["applied"] = bool(belief_seeding_used)
        belief_seed_metadata.setdefault("seed_clusters", {
            int(cid): latents for cid, latents in belief_seed_clusters_global.items()
        })
        if belief_slice_map is not None:
            belief_seed_metadata["belief_slices"] = {
                comp: [int(bounds[0]), int(bounds[1])] for comp, bounds in belief_slice_map.items()
            }
        extra_fields_final["belief_seed_metadata"] = belief_seed_metadata
    if cluster_r2_summary is not None:
        extra_fields_final["belief_cluster_r2"] = cluster_r2_summary
    extra_fields_final.update(subspace_diagnostics)
    write_cluster_metadata(
        metadata_path,
        cluster_stats,
        site_selected_k,
        average_l2,
        cluster_labels=cluster_labels_full.tolist(),
        extra_fields=extra_fields_final,
    )

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


# #%%
# if __name__ == "__main__":
#     main()
