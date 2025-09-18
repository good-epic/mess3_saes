#%%
# === Imports and Configuration === #
###################################
import argparse
import json
import os
from copy import deepcopy
from datetime import datetime

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig
from simplexity.generative_processes.torch_generator import generate_data_batch

from BatchTopK.sae import TopKSAE

from mess3_gmg import build_similarity_matrix, spectral_clustering_with_eigengap
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
from multipartite_generation import MultipartiteSampler, build_components_from_config


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
    "3xmess3_2xtquant": [
        {
            "type": "mess3",
            "instances": [
                {"x": 0.1, "a": 0.8},
                {"x": 0.25, "a": 0.2},
                {"x": 0.4, "a": 0.5},
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
    parser.add_argument("--model_ckpt", type=str, default="outputs/checkpoints/multipartite_1/checkpoint_step_50000_final.pt", help="Transformer checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Process configuration
    parser.add_argument("--process_config", type=str, default=None, help="JSON file describing stacked generative processes")
    parser.add_argument("--process_preset", type=str, default="3xmess3_2xtquant", help="Named preset for generative process configuration")

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

    args, _ = parser.parse_known_args()

    if args.process_config and args.process_preset:
        parser.error("Specify at most one of --process_config or --process_preset")

    return args


#%%
def _resolve_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


#%%
def _load_process_stack(args: argparse.Namespace) -> tuple[list[dict], list, object]:
    if args.process_config:
        with open(args.process_config, "r", encoding="utf-8") as f:
            process_cfg_raw = json.load(f)
    elif args.process_preset:
        if args.process_preset not in PRESET_PROCESS_CONFIGS:
            raise ValueError(f"Unknown process preset '{args.process_preset}'")
        process_cfg_raw = deepcopy(PRESET_PROCESS_CONFIGS[args.process_preset])
    else:
        process_cfg_raw = deepcopy(PRESET_PROCESS_CONFIGS["single_mess3"])

    components = build_components_from_config(process_cfg_raw)
    data_source: object
    if len(components) == 1:
        data_source = components[0].process
    else:
        data_source = MultipartiteSampler(components)
    return process_cfg_raw, components, data_source


#%%
def _load_transformer(args: argparse.Namespace, device: str, vocab_size: int):
    cfg = HookedTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_ctx=args.n_ctx,
        d_vocab=args.d_vocab if args.d_vocab is not None else vocab_size,
        act_fn=args.act_fn,
        device=device,
        d_head=args.d_head,
    )
    model = HookedTransformer(cfg).to(device)
    ckpt = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt.get("config"), dict):
        cfg_loaded = HookedTransformerConfig.from_dict(ckpt["config"])
        model = HookedTransformer(cfg_loaded).to(device)
        cfg = cfg_loaded

    state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state_dict is None:
        available = ", ".join(sorted(ckpt.keys()))
        raise KeyError(
            "Checkpoint does not contain 'state_dict' or 'model_state_dict'. "
            f"Available keys: {available}"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
    if missing or unexpected:
        print(
            "Warning: load_state_dict reported issues",
            {"missing": missing, "unexpected": unexpected},
        )
    model.eval()
    return model, cfg


#%%
def _select_sites(metrics_summary, requested_sites):
    available = [site for site in SITE_HOOK_MAP if site in metrics_summary]
    if requested_sites is None:
        return available
    valid = [s for s in requested_sites if s in SITE_HOOK_MAP]
    missing = sorted(set(requested_sites) - set(valid))
    if missing:
        print(f"Warning: ignoring unknown sites {missing}")
    return [s for s in valid if s in metrics_summary]


#%%
def _sample_tokens(
    data_source,
    batch_size: int,
    sample_len: int,
    target_len: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    key = jax.random.PRNGKey(seed)
    if isinstance(data_source, MultipartiteSampler):
        key, beliefs, tokens, _ = data_source.sample(key, batch_size, sample_len)
        _ = beliefs  # unused, but kept for clarity
        arr = np.array(tokens)
    else:
        gen_states = jnp.repeat(data_source.initial_state[None, :], batch_size, axis=0)
        _, inputs, _ = generate_data_batch(gen_states, data_source, batch_size, sample_len, key)
        arr = np.array(inputs)
    if target_len is not None and arr.shape[1] != target_len:
        if arr.shape[1] < target_len:
            raise ValueError(
                f"Sampled sequence length {arr.shape[1]} smaller than target length {target_len}"
            )
        arr = arr[:, :target_len]
    tokens = torch.from_numpy(arr).long().to(device)
    return tokens


def collect_latent_activity_data(
    model,
    sae,
    data_source,
    hook_name: str,
    *,
    batch_size: int,
    sample_len: int,
    target_len: int,
    n_batches: int,
    seed: int,
    device: str,
    activation_eps: float,
    collect_matrix: bool = False,
):
    if n_batches <= 0:
        raise ValueError("activation sampling requires n_batches >= 1")

    dict_size = sae.W_dec.shape[0]
    active_counts = torch.zeros(dict_size, dtype=torch.float64)
    mean_abs_sum = torch.zeros(dict_size, dtype=torch.float64)
    total_samples = 0
    binary_batches: list[torch.Tensor] = [] if collect_matrix else []

    for batch_idx in range(n_batches):
        tokens = _sample_tokens(
            data_source,
            batch_size,
            sample_len,
            target_len,
            seed + batch_idx + 1,
            device,
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None)
            acts = cache[hook_name].reshape(-1, cache[hook_name].shape[-1]).to(device)
            feature_acts, _, _ = sae_encode_features(sae, acts)
        mask = (feature_acts.abs() > activation_eps)
        active_counts += mask.sum(dim=0).to(torch.float64).cpu()
        mean_abs_sum += feature_acts.abs().sum(dim=0).to(torch.float64).cpu()
        total_samples += mask.shape[0]

        if collect_matrix:
            binary_batches.append(mask.cpu())

        del cache
        del acts
        del feature_acts
        del mask
        del tokens

    activity_rates = (active_counts / max(total_samples, 1)).numpy()
    mean_abs_activation = (mean_abs_sum / max(total_samples, 1)).numpy()
    latent_matrix = None
    if collect_matrix:
        latent_matrix = torch.cat(binary_batches, dim=0).numpy()
    return {
        "activity_rates": activity_rates,
        "mean_abs_activation": mean_abs_activation,
        "latent_matrix": latent_matrix,
        "total_samples": total_samples,
    }


#%%
# ==== Setup ==== #
###################

#def main() -> None:
args = _parse_args()
device = _resolve_device(args.device)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

process_cfg_raw, components, data_source = _load_process_stack(args)
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
sites = _select_sites(metrics_summary, args.sites)
if not sites:
    raise ValueError("No valid sites available for analysis")


#%%
# ==== Choose K ==== #
######################
l2_by_site = extract_topk_l2(metrics_summary, site_filter=sites)
if not l2_by_site:
    raise ValueError("No top-k SAE metrics found for the selected sites")

average_l2 = compute_average_l2(l2_by_site)
if not average_l2:
    raise ValueError("Unable to compute average L2 across k values")

if args.force_k is not None:
    selected_k = args.force_k
    print(f"Using forced k={selected_k}")
else:
    sorted_k = sorted(average_l2.keys())
    losses = [average_l2[k] for k in sorted_k]
    selected_k = find_elbow_k(sorted_k, losses, prefer_high_k=True)
    print(f"Selected k={selected_k} via elbow heuristic")

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
sae_folder_name = os.path.basename(os.path.normpath(args.sae_folder)) or "saes"
run_dir = os.path.join(args.output_dir, sae_folder_name, f"k{selected_k}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

l2_plot_path = os.path.join(run_dir, "l2_summary.png")
plot_l2_bar_chart(l2_by_site, l2_plot_path)
with open(os.path.join(run_dir, "l2_summary.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "per_site": {site: {int(k): float(v) for k, v in metrics.items()} for site, metrics in l2_by_site.items()},
            "average": {int(k): float(v) for k, v in average_l2.items()},
            "selected_k": int(selected_k),
            "process_kind": process_kind,
            "components": component_summary,
        },
        f,
        indent=2,
    )

batch_size = args.sample_sequences
if isinstance(data_source, MultipartiteSampler):
    sample_len = seq_len
else:
    sample_len = seq_len + 1
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
    sae_path = os.path.join(args.sae_folder, f"{site}_top_k_k{selected_k}.pt")
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
            selected_k,
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
            },
        )
        del feature_acts
        continue

    decoder_active = decoder_dirs[active_indices]
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
            selected_k,
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
        n_components=3,
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
            stats["decoder_projections_pc123"] = {
                int(latent_idx): [float(x) for x in vec.tolist()]
                for latent_idx, vec in result.decoder_coords.items()
            }
        plot_path = os.path.join(site_dir, f"cluster_{cid}_pca.png")
        plot_cluster_pca(
            site,
            selected_k,
            cid,
            result,
            plot_path,
            max_points=args.plot_max_points,
            random_state=args.seed,
        )

    metadata_path = os.path.join(site_dir, "cluster_summary.json")
    write_cluster_metadata(
        metadata_path,
        cluster_stats,
        selected_k,
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
