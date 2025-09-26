#%%
# === Imports and Configuration === #
###################################
import argparse
import json
import os
from copy import deepcopy
from datetime import datetime

os.environ["JAX_PLATFORM_NAME"] = "cpu"

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
    load_metrics_summary,
    plot_activity_histogram,
    plot_activity_histograms_by_site,
    plot_activity_histograms_site_clusters,
    plot_cluster_pca,
    plot_l2_bar_chart,
    project_decoder_directions_to_pca,
    sae_encode_features,
    write_cluster_metadata,
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
    parser.add_argument("--sae_folder", type=str, default="outputs/saes/multipartite_1", help="Folder with SAE checkpoints")
    parser.add_argument("--metrics_summary", type=str, default=None, help="metrics_summary.json path; defaults to <sae_folder>/metrics_summary.json")
    parser.add_argument("--output_dir", type=str, default="outputs/reports/multipartite_1", help="Root directory for analysis outputs")
    parser.add_argument("--model_ckpt", type=str, default="outputs/checkpoints/multipartite_1/checkpoint_step_500000_final.pt", help="Transformer checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Process configuration
    parser.add_argument("--process_config", type=str, default=None, help="JSON file describing stacked generative processes")
    parser.add_argument("--process_preset", type=str, default="3xmess3_2xtquant_001", help="Named preset for generative process configuration")

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
    parser.add_argument("--sim_metric", type=str, default="phi", choices=["cosine", "euclidean", "phi"], help="Similarity metric for decoder clustering")
    parser.add_argument("--max_clusters", type=int, default=12, help="Upper bound for eigengap clustering")
    parser.add_argument("--plot_eigengap", action="store_true", help="Plot eigengap spectrum diagnostics")
    parser.add_argument("--center_decoder_rows", action="store_true", help="Center and renormalize decoder rows before computing similarities")

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
    parser.add_argument(
        "--pca_components",
        type=int,
        default=6,
        help="Number of PCA components to compute before selecting the top three for plotting",
    )

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
tokens = _sample_tokens(
    data_source,
    batch_size,
    sample_len,
    seq_len,
    args.seed,
    device,
)
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
    if args.max_activations and acts_flat.shape[0] > args.max_activations:
        idx = torch.randperm(acts_flat.shape[0], device=acts_flat.device)[: args.max_activations]
        acts_flat = acts_flat[idx].contiguous()
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
        write_cluster_metadata(
            metadata_path,
            {},
            site_selected_k,
            average_l2,
            cluster_labels=cluster_labels_full.tolist(),
            extra_fields={
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
            },
        )
        del feature_acts
        continue

    decoder_active = decoder_dirs[active_indices]
    if args.center_decoder_rows and decoder_active.size > 0:
        row_mean = decoder_active.mean(axis=0, keepdims=True)
        decoder_active = decoder_active - row_mean
        norms = np.linalg.norm(decoder_active, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        decoder_active = decoder_active / norms
    if decoder_active.shape[0] == 1:
        cluster_labels_active = np.array([0], dtype=int)
        spectral_k = 1
    else:
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
    cluster_labels_full[active_indices] = cluster_labels_active
    print(
        f"{site}: clustered {decoder_active.shape[0]} active latents into {spectral_k} groups"
    )

    encoded_cache = (feature_acts, x_mean, x_std)
    with torch.no_grad():
        cluster_recons, cluster_stats = collect_cluster_reconstructions(
            acts_flat,
            sae,
            cluster_labels_full.tolist(),
            min_activation=args.cluster_activation_threshold,
            encoded_cache=encoded_cache,
        )
    del feature_acts
    del encoded_cache
    del x_mean, x_std

    if not cluster_stats:
        print(f"{site}: no cluster activations above threshold {args.cluster_activation_threshold}; skipping PCA plots")
        metadata_path = os.path.join(site_dir, "cluster_summary.json")
        write_cluster_metadata(
            metadata_path,
            cluster_stats,
            site_selected_k,
            average_l2,
            cluster_labels=cluster_labels_full.tolist(),
            extra_fields={
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
            },
        )
        continue

    for cid, stats in cluster_stats.items():
        latent_ids = stats.get("latent_indices", [])
        stats["activity_rates"] = {int(idx): float(activity_rates[idx]) for idx in latent_ids}
        stats["mean_abs_activation"] = {int(idx): float(mean_abs_activation[idx]) for idx in latent_ids}
        rates_array = np.array([activity_rates[idx] for idx in latent_ids], dtype=float)
        if rates_array.size > 0:
            per_site_cluster_rates[site][int(cid)] = rates_array

    pca_results = fit_pca_for_clusters(
        cluster_recons,
        n_components=args.pca_components,
        min_samples=args.min_cluster_samples,
    )
    project_decoder_directions_to_pca(sae, pca_results, cluster_stats)

    clusters_with_pca = []
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

    metadata_path = os.path.join(site_dir, "cluster_summary.json")
    write_cluster_metadata(
        metadata_path,
        cluster_stats,
        site_selected_k,
        average_l2,
        cluster_labels=cluster_labels_full.tolist(),
        extra_fields={
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
            },
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
