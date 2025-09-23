import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import torch
import torch.nn.functional as F
import numpy as np
import jax
import jax.numpy as jnp

from typing import Dict, List, Tuple, Sequence, Mapping, Any, Callable
import itertools
from itertools import combinations
import warnings
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from collections import Counter
import scipy.stats as st
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from simplexity.generative_processes.torch_generator import generate_data_batch
from BatchTopK.sae import VanillaSAE, TopKSAE
from multipartite_utils import MultipartiteSampler, build_components_from_config



def check_cuda_memory():
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")


def _build_sae_cfg(
    act_size: int,
    dict_mul: int = 4,
    k: int = 2,
    l1_coeff: float = 1e-3,
    device: str = "cpu",
    input_unit_norm: bool = True,
    n_batches_to_dead: int = 5,
    top_k_aux: int | None = None,
    aux_penalty: float = 1/32,
    bandwidth: float = 0.001,
):
    return {
        "seed": 0,
        "act_size": act_size,
        "dict_size": dict_mul * act_size,
        "device": device,
        "dtype": torch.float32,
        "input_unit_norm": input_unit_norm,
        "l1_coeff": l1_coeff,
        "n_batches_to_dead": n_batches_to_dead,
        # (Batch)TopK-specific
        "top_k": k,
        "top_k_aux": (max(4 * k, 8) if top_k_aux is None else top_k_aux),
        "aux_penalty": aux_penalty,
        # JumpReLU bandwidth exists in repo; unused here
        "bandwidth": bandwidth,
    }


def _generate_sequences(rng_key, batch_size: int, sequence_len: int, source) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """Return ``(new_rng_key, states, observations)`` for a generator source."""
    if isinstance(source, MultipartiteSampler):
        rng_key, states, observations, _ = source.sample(rng_key, batch_size, sequence_len)
        return rng_key, states, observations

    if hasattr(source, "sample") and callable(source.sample):
        rng_key, states, observations = source.sample(rng_key, batch_size, sequence_len)
        return rng_key, states, observations

    rng_key, subkey = jax.random.split(rng_key)
    batch_keys = jax.random.split(subkey, batch_size)
    initial_states = jnp.tile(source.initial_state, (batch_size, 1))
    states, observations = source.generate(initial_states, batch_keys, sequence_len, True)
    return rng_key, states, observations


def _tokens_from_observations(observations, device: str) -> torch.Tensor:
    if isinstance(observations, torch.Tensor):
        return observations.long().to(device)
    return torch.from_numpy(np.array(observations)).long().to(device)


def _acts_batch(hook_name: str, tokens: torch.Tensor, model) -> torch.Tensor:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type=None)
        acts = cache[hook_name]  # [batch, seq, d_model]
        acts = acts.reshape(-1, acts.shape[-1]).detach()
    return acts
# Multi-site training: generate one batch per step, run model once, use all hooks
def train_saes_for_sites(
    site_to_hook: Dict[str, str],
    *,
    model,
    data_source,
    cfg,
    device: str,
    rng_key,
    steps: int = 200,
    batch_size: int = 64,
    seq_len: int = None,
    k_values: list[int] | None = None,
    lambda_values_seq: list[float] | None = None,
    lambda_values_beliefs: list[float] | None = None,
    dict_mul: int = 4,
    input_unit_norm: bool = True,
    n_batches_to_dead: int = 5,
    top_k_aux: int | None = None,
    aux_penalty: float = 1/32,
    bandwidth: float = 0.001,
    belief_dim: int | None = None,
    belief_dict_size: int | None = None,
    sae_output_dir: str = "outputs/saes",
):
    if seq_len is None:
        seq_len = cfg.n_ctx - 1

    act_size = cfg.d_model
    seq_lams = list(lambda_values_seq) if lambda_values_seq is not None else []
    beliefs_lams = list(lambda_values_beliefs) if lambda_values_beliefs is not None else []
    topk_values = list(k_values) if k_values is not None else []

    has_topk = len(topk_values) > 0
    has_seq_vanilla = len(seq_lams) > 0
    has_beliefs = len(beliefs_lams) > 0

    if not (has_topk or has_seq_vanilla or has_beliefs):
        raise ValueError(
            "train_saes_for_sites requires at least one of k_values, lambda_values_seq, or "
            "lambda_values_beliefs to be provided."
        )

    if has_beliefs:
        if belief_dim is None:
            if hasattr(data_source, "belief_dim"):
                belief_dim = int(getattr(data_source, "belief_dim"))
            elif hasattr(data_source, "num_states"):
                belief_dim = int(getattr(data_source, "num_states"))
            else:
                raise ValueError("Unable to infer belief dimensionality; please provide belief_dim explicitly when training belief SAEs.")
        if belief_dict_size is None:
            belief_dict_size = max(8, belief_dim * 2)

    # Build SAEs and optimizers per site
    seq_topk_all: Dict[str, Dict[str, TopKSAE]] = {site: {} for site in site_to_hook}
    seq_vanilla_all: Dict[str, Dict[str, VanillaSAE]] = {site: {} for site in site_to_hook}
    true_coord_saes_all: Dict[str, Dict[str, VanillaSAE]] = {site: {} for site in site_to_hook}

    opt_seq_topk_all: Dict[str, Dict[str, torch.optim.Optimizer]] = {site: {} for site in site_to_hook}
    opt_seq_vanilla_all: Dict[str, Dict[str, torch.optim.Optimizer]] = {site: {} for site in site_to_hook}
    opt_true_all: Dict[str, Dict[str, torch.optim.Optimizer]] = {site: {} for site in site_to_hook}

    for site_name in site_to_hook.keys():
        if has_topk:
            for k in topk_values:
                cfg_topk = _build_sae_cfg(
                    act_size=act_size,
                    dict_mul=dict_mul,
                    k=k,
                    l1_coeff=0.0,
                    device=device,
                    input_unit_norm=input_unit_norm,
                    n_batches_to_dead=n_batches_to_dead,
                    top_k_aux=top_k_aux,
                    aux_penalty=aux_penalty,
                    bandwidth=bandwidth,
                )
                sae = TopKSAE(cfg_topk)
                name = f"k{k}"
                seq_topk_all[site_name][name] = sae
                opt_seq_topk_all[site_name][name] = torch.optim.Adam(sae.parameters(), lr=3e-4, betas=(0.9, 0.99))

        if has_seq_vanilla:
            for lam in seq_lams:
                cfg_s_v = _build_sae_cfg(
                    act_size=act_size,
                    dict_mul=dict_mul,
                    k=1,
                    l1_coeff=lam,
                    device=device,
                    input_unit_norm=input_unit_norm,
                    n_batches_to_dead=n_batches_to_dead,
                    top_k_aux=top_k_aux,
                    aux_penalty=aux_penalty,
                    bandwidth=bandwidth,
                )
                sae = VanillaSAE(cfg_s_v)
                name = f"lambda_{lam}"
                seq_vanilla_all[site_name][name] = sae
                opt_seq_vanilla_all[site_name][name] = torch.optim.Adam(sae.parameters(), lr=3e-4, betas=(0.9, 0.99))

        if has_beliefs:
            for lam in beliefs_lams:
                cfg_v = _build_sae_cfg(
                    act_size=belief_dim,
                    dict_mul=0,
                    k=1,
                    l1_coeff=lam,
                    device=device,
                    input_unit_norm=input_unit_norm,
                    n_batches_to_dead=n_batches_to_dead,
                    top_k_aux=top_k_aux,
                    aux_penalty=aux_penalty,
                    bandwidth=bandwidth,
                )
                cfg_v["dict_size"] = belief_dict_size
                sae = VanillaSAE(cfg_v)
                name = f"lambda_{lam}"
                true_coord_saes_all[site_name][name] = sae
                opt_true_all[site_name][name] = torch.optim.Adam(sae.parameters(), lr=3e-4, betas=(0.9, 0.99))

    # Metrics containers per site
    metrics_raw_all: Dict[str, Dict] = {}
    for site_name in site_to_hook.keys():
        metrics_raw_all[site_name] = {
            "sequence": {
                "top_k": {name: {"loss": [], "l2_loss": [], "l1_loss": [], "l0_norm": [], "l1_norm": [], "aux_loss": [], "num_dead_features": [], "active_latents": [], "active_counts": [], "active_sums": []} for name in seq_topk_all[site_name].keys()},
                "vanilla": {name: {"loss": [], "l2_loss": [], "l1_loss": [], "l0_norm": [], "l1_norm": [], "active_latents": [], "active_counts": [], "active_sums": []} for name in seq_vanilla_all[site_name].keys()},
            },
            "beliefs": {name: {"loss": [], "l2_loss": [], "l1_loss": [], "l0_norm": [], "l1_norm": [], "active_latents": [], "active_counts": [], "active_sums": []} for name in true_coord_saes_all[site_name].keys()},
        }
    
    miniters = 500 if steps > 500 else steps
    progress_bar = tqdm(range(steps), desc="SAEs (all sites)", miniters=miniters, disable=not sys.stderr.isatty())
    # Training loop
    for ii in progress_bar:
        rng_key, states, observations = _generate_sequences(
            rng_key,
            batch_size=batch_size,
            sequence_len=seq_len,
            source=data_source,
        )
        tokens = _tokens_from_observations(observations, device=device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None)

        beliefs_tensor = None
        if has_beliefs:
            beliefs = states
            if beliefs.ndim == 3:
                if beliefs.shape[1] == tokens.shape[1] + 1:
                    beliefs = beliefs[:, 1:, :]
                elif beliefs.shape[1] == tokens.shape[1]:
                    pass
            beliefs_tensor = torch.from_numpy(np.array(beliefs)).to(device=device, dtype=torch.float32)
            beliefs_tensor = beliefs_tensor.reshape(-1, beliefs_tensor.shape[-1])

        # Per-site updates for sequence SAEs
        for site_name, hook_name in site_to_hook.items():
            acts = cache[hook_name]
            acts_seq = acts.reshape(-1, acts.shape[-1]).detach()

            # TopK
            for name, sae in seq_topk_all[site_name].items():
                sae.train()
                out = sae(acts_seq)
                loss = out["loss"]
                opt = opt_seq_topk_all[site_name][name]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1e5)
                sae.make_decoder_weights_and_grad_unit_norm()
                opt.step()
                mr = metrics_raw_all[site_name]["sequence"]["top_k"][name]
                mr["loss"].append(float(loss.detach().item()))
                for key in ("l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss"):
                    if key in out:
                        mr[key].append(float(out[key].detach().item()))
                if "num_dead_features" in out:
                    v = out["num_dead_features"]
                    v = v.item() if hasattr(v, "item") else int(v)
                    mr["num_dead_features"].append(int(v))
                with torch.no_grad():
                    acts_f = out.get("feature_acts")
                    if acts_f is not None:
                        mask = (acts_f > 0).any(dim=0).detach().cpu().numpy().astype(bool)
                        counts = (acts_f > 0).sum(dim=0).detach().cpu().numpy()
                        sums = acts_f.clamp(min=0).sum(dim=0).detach().cpu().numpy()
                        mr["active_latents"].append(mask)
                        mr["active_counts"].append(counts)
                        mr["active_sums"].append(sums)

            # Sequence Vanilla
            for name, sae in seq_vanilla_all[site_name].items():
                sae.train()
                out = sae(acts_seq)
                loss = out["loss"]
                opt = opt_seq_vanilla_all[site_name][name]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1e5)
                sae.make_decoder_weights_and_grad_unit_norm()
                opt.step()
                mr = metrics_raw_all[site_name]["sequence"]["vanilla"][name]
                mr["loss"].append(float(loss.detach().item()))
                for key in ("l2_loss", "l1_loss", "l0_norm", "l1_norm"):
                    if key in out:
                        mr[key].append(float(out[key].detach().item()))
                with torch.no_grad():
                    acts_f = out.get("feature_acts")
                    if acts_f is not None:
                        mask = (acts_f > 0).any(dim=0).detach().cpu().numpy().astype(bool)
                        counts = (acts_f > 0).sum(dim=0).detach().cpu().numpy()
                        sums = acts_f.clamp(min=0).sum(dim=0).detach().cpu().numpy()
                        mr["active_latents"].append(mask)
                        mr["active_counts"].append(counts)
                        mr["active_sums"].append(sums)

        # Beliefs SAEs (per site)
        if has_beliefs:
            for site_name in site_to_hook.keys():
                for name, sae in true_coord_saes_all[site_name].items():
                    if beliefs_tensor is None:
                        raise RuntimeError("Belief SAEs requested but belief tensor was not computed.")
                    sae.train()
                    out = sae(beliefs_tensor)
                    loss = out["loss"]
                    opt = opt_true_all[site_name][name]
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1e5)
                    sae.make_decoder_weights_and_grad_unit_norm()
                    opt.step()
                    mr = metrics_raw_all[site_name]["beliefs"][name]
                    mr["loss"].append(float(loss.detach().item()))
                    for key in ("l2_loss", "l1_loss", "l0_norm", "l1_norm"):
                        if key in out:
                            mr[key].append(float(out[key].detach().item()))
                    with torch.no_grad():
                        acts_f = out.get("feature_acts")
                        if acts_f is not None:
                            mask = (acts_f > 0).any(dim=0).detach().cpu().numpy().astype(bool)
                            counts = (acts_f > 0).sum(dim=0).detach().cpu().numpy()
                            sums = acts_f.clamp(min=0).sum(dim=0).detach().cpu().numpy()
                            mr["active_latents"].append(mask)
                            mr["active_counts"].append(counts)
                            mr["active_sums"].append(sums)
        if (ii + 1) % miniters == 0:
            progress_bar.set_description(f"SAEs (all sites) (Loss: {loss.item():.4f})", refresh=False)


    # Evaluate reconstruction errors on finalized SAEs
    reconstruction_error_accumulators: Dict[str, Dict] = {
        site_name: {
            "sequence": {
                "top_k": {name: {"sum": 0.0, "count": 0} for name in seq_topk_all[site_name]},
                "vanilla": {name: {"sum": 0.0, "count": 0} for name in seq_vanilla_all[site_name]},
            },
            "beliefs": {name: {"sum": 0.0, "count": 0} for name in true_coord_saes_all[site_name]},
        }
        for site_name in site_to_hook.keys()
    }
    baseline_error_accumulators: Dict[str, Dict[str, Dict[str, float]]] = {
        site_name: {
            "sequence": {"sum": 0.0, "count": 0.0},
            "beliefs": {"sum": 0.0, "count": 0.0},
        }
        for site_name in site_to_hook.keys()
    }

    def _compute_baseline_mse(data: torch.Tensor) -> float:
        if data.numel() == 0:
            return 0.0
        data_float = data.detach().to(torch.float32)
        if input_unit_norm:
            mean_per_sample = data_float.mean(dim=-1, keepdim=True)
            std_per_sample = data_float.std(dim=-1, unbiased=False, keepdim=True)
            data_float = (data_float - mean_per_sample) / (std_per_sample + 1e-5)
        dataset_mean = data_float.mean(dim=0, keepdim=True)
        residual = data_float - dataset_mean
        return float(residual.pow(2).mean().item())
    eval_batches = max(1, min(steps, 10))
    with torch.no_grad():
        for _ in range(eval_batches):
            rng_key, states_eval, observations_eval = _generate_sequences(
                rng_key,
                batch_size=batch_size,
                sequence_len=seq_len,
                source=data_source,
            )
            tokens_eval = _tokens_from_observations(observations_eval, device=device)
            _, cache_eval = model.run_with_cache(tokens_eval, return_type=None)

            beliefs_eval_tensor = None
            belief_element_count = 0
            if has_beliefs:
                beliefs_eval = states_eval
                if beliefs_eval.ndim == 3:
                    if beliefs_eval.shape[1] == tokens_eval.shape[1] + 1:
                        beliefs_eval = beliefs_eval[:, 1:, :]
                    elif beliefs_eval.shape[1] == tokens_eval.shape[1]:
                        pass
                beliefs_np = np.array(beliefs_eval)
                beliefs_eval_tensor = torch.from_numpy(beliefs_np).to(device=device, dtype=torch.float32)
                belief_dim_eval = beliefs_eval_tensor.shape[-1]
                beliefs_eval_tensor = beliefs_eval_tensor.reshape(-1, belief_dim_eval)
                belief_element_count = beliefs_eval_tensor.shape[0] * beliefs_eval_tensor.shape[1]
            belief_baseline_mse = None
            if has_beliefs and beliefs_eval_tensor is not None and belief_element_count > 0:
                belief_baseline_mse = _compute_baseline_mse(beliefs_eval_tensor)

            for site_name, hook_name in site_to_hook.items():
                acts_eval = cache_eval[hook_name]
                acts_eval = acts_eval.reshape(-1, acts_eval.shape[-1]).detach()
                sample_element_count = acts_eval.shape[0] * acts_eval.shape[1]

                if sample_element_count > 0:
                    baseline_mse_val = _compute_baseline_mse(acts_eval)
                    baseline_acc = baseline_error_accumulators[site_name]["sequence"]
                    baseline_acc["sum"] += baseline_mse_val * sample_element_count
                    baseline_acc["count"] += sample_element_count

                for name, sae in seq_topk_all[site_name].items():
                    sae.eval()
                    out = sae(acts_eval)
                    l2_loss = out.get("l2_loss") or out.get("loss")
                    if l2_loss is None or sample_element_count == 0:
                        continue
                    recon_error = float(l2_loss.detach().item())
                    acc = reconstruction_error_accumulators[site_name]["sequence"]["top_k"][name]
                    acc["sum"] += recon_error * sample_element_count
                    acc["count"] += sample_element_count

                for name, sae in seq_vanilla_all[site_name].items():
                    sae.eval()
                    out = sae(acts_eval)
                    l2_loss = out.get("l2_loss") or out.get("loss")
                    if l2_loss is None or sample_element_count == 0:
                        continue
                    recon_error = float(l2_loss.detach().item())
                    acc = reconstruction_error_accumulators[site_name]["sequence"]["vanilla"][name]
                    acc["sum"] += recon_error * sample_element_count
                    acc["count"] += sample_element_count

                if has_beliefs and beliefs_eval_tensor is not None and belief_element_count > 0:
                    if belief_baseline_mse is not None:
                        baseline_beliefs_acc = baseline_error_accumulators[site_name]["beliefs"]
                        baseline_beliefs_acc["sum"] += belief_baseline_mse * belief_element_count
                        baseline_beliefs_acc["count"] += belief_element_count
                    for name, sae in true_coord_saes_all[site_name].items():
                        sae.eval()
                        out = sae(beliefs_eval_tensor)
                        l2_loss = out.get("l2_loss") or out.get("loss")
                        if l2_loss is None:
                            continue
                        recon_error = float(l2_loss.detach().item())
                        acc = reconstruction_error_accumulators[site_name]["beliefs"][name]
                        acc["sum"] += recon_error * belief_element_count
                        acc["count"] += belief_element_count

    reconstruction_errors_all: Dict[str, Dict] = {
        site_name: {"sequence": {"top_k": {}, "vanilla": {}}, "beliefs": {}}
        for site_name in site_to_hook.keys()
    }
    percent_variance_explained_all: Dict[str, Dict] = {
        site_name: {"sequence": {"top_k": {}, "vanilla": {}}, "beliefs": {}}
        for site_name in site_to_hook.keys()
    }
    baseline_mse_all: Dict[str, Dict[str, float | None]] = {
        site_name: {"sequence": None, "beliefs": None}
        for site_name in site_to_hook.keys()
    }

    for site_name, site_baseline in baseline_error_accumulators.items():
        seq_baseline = site_baseline["sequence"]
        if seq_baseline["count"] > 0:
            baseline_mse_all[site_name]["sequence"] = float(seq_baseline["sum"] / seq_baseline["count"])
        beliefs_baseline = site_baseline["beliefs"]
        if beliefs_baseline["count"] > 0:
            baseline_mse_all[site_name]["beliefs"] = float(beliefs_baseline["sum"] / beliefs_baseline["count"])

    for site_name, site_acc in reconstruction_error_accumulators.items():
        seq_baseline_val = baseline_mse_all[site_name]["sequence"]
        for name, stats in site_acc["sequence"]["top_k"].items():
            if stats["count"] > 0:
                recon_error = float(stats["sum"] / stats["count"])
                reconstruction_errors_all[site_name]["sequence"]["top_k"][name] = recon_error
                if seq_baseline_val is not None and seq_baseline_val > 0:
                    percent_variance_explained_all[site_name]["sequence"]["top_k"][name] = float(1.0 - recon_error / seq_baseline_val)
                else:
                    percent_variance_explained_all[site_name]["sequence"]["top_k"][name] = None
            else:
                reconstruction_errors_all[site_name]["sequence"]["top_k"][name] = None
                percent_variance_explained_all[site_name]["sequence"]["top_k"][name] = None
        for name, stats in site_acc["sequence"]["vanilla"].items():
            if stats["count"] > 0:
                recon_error = float(stats["sum"] / stats["count"])
                reconstruction_errors_all[site_name]["sequence"]["vanilla"][name] = recon_error
                if seq_baseline_val is not None and seq_baseline_val > 0:
                    percent_variance_explained_all[site_name]["sequence"]["vanilla"][name] = float(1.0 - recon_error / seq_baseline_val)
                else:
                    percent_variance_explained_all[site_name]["sequence"]["vanilla"][name] = None
            else:
                reconstruction_errors_all[site_name]["sequence"]["vanilla"][name] = None
                percent_variance_explained_all[site_name]["sequence"]["vanilla"][name] = None

        beliefs_baseline_val = baseline_mse_all[site_name]["beliefs"]
        for name, stats in site_acc["beliefs"].items():
            if stats["count"] > 0:
                recon_error = float(stats["sum"] / stats["count"])
                reconstruction_errors_all[site_name]["beliefs"][name] = recon_error
                if beliefs_baseline_val is not None and beliefs_baseline_val > 0:
                    percent_variance_explained_all[site_name]["beliefs"][name] = float(1.0 - recon_error / beliefs_baseline_val)
                else:
                    percent_variance_explained_all[site_name]["beliefs"][name] = None
            else:
                reconstruction_errors_all[site_name]["beliefs"][name] = None
                percent_variance_explained_all[site_name]["beliefs"][name] = None

    # Summarization helpers
    def summarize_series(series: list, steps: int):
        if not series:
            return None, []
        start = max(0, int(steps * 3 / 4))
        last_quarter = series[start:]
        avg_last_quarter = float(np.mean(last_quarter)) if len(last_quarter) > 0 else None
        last50 = series[-50:] if len(series) >= 50 else series[:]
        return avg_last_quarter, last50

    def summarize_active_stats(active_counts: list, active_sums: list, steps: int):
        if not active_counts or not active_sums:
            return {}
        start = max(0, int(steps * 3 / 4))
        counts_win = active_counts[start:]
        sums_win = active_sums[start:]
        try:
            counts_total = np.stack(counts_win, axis=0).sum(axis=0)
            sums_total = np.stack(sums_win, axis=0).sum(axis=0)
            result = {}
            for idx, cnt in enumerate(counts_total.tolist()):
                if cnt > 0:
                    result[int(idx)] = (int(cnt), float(sums_total[idx]))
            return result
        except Exception:
            return {}

    # Build metrics_summary per site and save SAEs
    metrics_summary_all: Dict[str, Dict] = {}
    os.makedirs(sae_output_dir, exist_ok=True)
    for site_name in site_to_hook.keys():
        metrics_raw = metrics_raw_all[site_name]
        metrics_summary: Dict[str, Dict] = {"sequence": {"top_k": {}, "vanilla": {}}, "beliefs": {}}

        for name, series_dict in metrics_raw["sequence"]["top_k"].items():
            avg_l2, _ = summarize_series(series_dict.get("l2_loss", []), steps)
            avg_l1, _ = summarize_series(series_dict.get("l1_loss", []), steps)
            avg_aux, _ = summarize_series(series_dict.get("aux_loss", []), steps)
            avg_l0, _ = summarize_series(series_dict.get("l0_norm", []), steps)
            avg_dead, _ = summarize_series(series_dict.get("num_dead_features", []), steps)
            _, last50 = summarize_series(series_dict.get("loss", []), steps)
            final_loss = series_dict.get("loss", [None])[-1] if series_dict.get("loss") else None
            active_dict = summarize_active_stats(series_dict.get("active_counts", []), series_dict.get("active_sums", []), steps)
            metrics_summary["sequence"]["top_k"][name] = {
                "avg_last_quarter": {"l2": avg_l2, "l1": avg_l1, "aux": avg_aux, "l0": avg_l0, "dead": avg_dead},
                "last50_loss": last50,
                "final_loss": final_loss,
                "active_latents_last_quarter": active_dict,
            }

        for name, series_dict in metrics_raw["sequence"]["vanilla"].items():
            avg_l2, _ = summarize_series(series_dict.get("l2_loss", []), steps)
            avg_l1, _ = summarize_series(series_dict.get("l1_loss", []), steps)
            avg_l0, _ = summarize_series(series_dict.get("l0_norm", []), steps)
            _, last50 = summarize_series(series_dict.get("loss", []), steps)
            final_loss = series_dict.get("loss", [None])[-1] if series_dict.get("loss") else None
            active_dict = summarize_active_stats(series_dict.get("active_counts", []), series_dict.get("active_sums", []), steps)
            metrics_summary["sequence"]["vanilla"][name] = {
                "avg_last_quarter": {"l2": avg_l2, "l1": avg_l1, "l0": avg_l0},
                "last50_loss": last50,
                "final_loss": final_loss,
                "active_latents_last_quarter": active_dict,
            }

        for name, series_dict in metrics_raw["beliefs"].items():
            avg_l2, _ = summarize_series(series_dict.get("l2_loss", []), steps)
            avg_l1, _ = summarize_series(series_dict.get("l1_loss", []), steps)
            avg_l0, _ = summarize_series(series_dict.get("l0_norm", []), steps)
            _, last50 = summarize_series(series_dict.get("loss", []), steps)
            final_loss = series_dict.get("loss", [None])[-1] if series_dict.get("loss") else None
            active_dict = summarize_active_stats(series_dict.get("active_counts", []), series_dict.get("active_sums", []), steps)
            metrics_summary["beliefs"][name] = {
                "avg_last_quarter": {"l2": avg_l2, "l1": avg_l1, "l0": avg_l0},
                "last50_loss": last50,
                "final_loss": final_loss,
                "active_latents_last_quarter": active_dict,
            }

        # Save SAEs for this site
        for name, sae in seq_topk_all[site_name].items():
            torch.save({"state_dict": sae.state_dict(), "cfg": sae.cfg}, os.path.join(sae_output_dir, f"{site_name}_top_k_{name}.pt"))
        for name, sae in seq_vanilla_all[site_name].items():
            torch.save({"state_dict": sae.state_dict(), "cfg": sae.cfg}, os.path.join(sae_output_dir, f"{site_name}_vanilla_{name}.pt"))
        for name, sae in true_coord_saes_all[site_name].items():
            torch.save({"state_dict": sae.state_dict(), "cfg": sae.cfg}, os.path.join(sae_output_dir, f"{site_name}_beliefs_{name}.pt"))

        metrics_summary_all[site_name] = metrics_summary

    reconstruction_errors_path = os.path.join(sae_output_dir, "reconstruction_errors.json")
    reconstruction_report = {
        "reconstruction_mse": reconstruction_errors_all,
        "percent_variance_explained": percent_variance_explained_all,
        "baseline_mse": baseline_mse_all,
    }
    with open(reconstruction_errors_path, "w") as f:
        json.dump(reconstruction_report, f, indent=2)

    # Return grouped by site, mirroring prior structures
    sequence_saes_all = {site: {"top_k": seq_topk_all[site], "vanilla": seq_vanilla_all[site]} for site in site_to_hook.keys()}
    return sequence_saes_all, true_coord_saes_all, metrics_summary_all



def project_decoder_directions(
    sae_entries: list,
    pcas: dict,
    latent_indices: dict,
    dims: int = 2,
    normalize: bool = True,
) -> dict:
    """
    Project SAE decoder directions for specified latents into a PCA subspace per site.

    Args:
      sae_entries: list of SAE descriptors; each item is a dict with keys:
        - 'site': site_name, e.g. 'layer_0', 'embeddings'
        - 'sae': instantiated SAE module with attribute W_dec (torch.nn.Parameter)
        - 'sae_type': e.g. 'top_k' or 'vanilla'
        - 'param': string identifier, e.g. 'k2' or 'lambda_0.1'
      pcas: dict mapping site_name -> fitted PCA object (sklearn.decomposition.PCA)
      latent_indices: dict mapping (site, sae_type, param) -> iterable of latent indices to project
      dims: 2 or 3, number of principal components to project onto
      normalize: if True, L2-normalize decoder rows before projection (direction only)

    Returns:
      projections: dict[site][sae_type][param][latent_idx] = np.ndarray of shape (dims,)
    """
    assert dims in (2, 3), "dims must be 2 or 3"

    projections: dict = {}

    for entry in sae_entries:
        site = entry.get('site')
        sae = entry.get('sae')
        sae_type = entry.get('sae_type')
        param = entry.get('param')

        if site not in pcas:
            raise KeyError(f"No PCA provided for site '{site}'")
        pca = pcas[site]
        if getattr(pca, 'components_', None) is None:
            raise ValueError(f"PCA for site '{site}' is not fitted (missing components_)\n")
        if pca.components_.shape[0] < dims:
            raise ValueError(f"PCA for site '{site}' has only {pca.components_.shape[0]} components, but dims={dims}")

        # Get latent indices to project
        idx_key = (site, sae_type, param)
        idxs = latent_indices.get(idx_key, None)
        if idxs is None:
            # If not provided, skip this SAE
            continue

        # Access decoder weights
        if not hasattr(sae, 'W_dec'):
            raise AttributeError(f"SAE for {idx_key} has no attribute 'W_dec'")
        W_dec = sae.W_dec.detach().cpu().numpy()  # shape: [dict_size, d_model]

        # Prepare PCA components (dims x d_model)
        comps = pca.components_[:dims, :]  # (dims, d_model)

        # Initialize nested dicts
        projections.setdefault(site, {}).setdefault(sae_type, {}).setdefault(param, {})

        for li in idxs:
            li_int = int(li)
            if li_int < 0 or li_int >= W_dec.shape[0]:
                continue
            v = W_dec[li_int]  # (d_model,)
            if normalize:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            # Project: coords = components * v
            coords = comps @ v  # (dims,)
            projections[site][sae_type][param][li_int] = coords

    return projections


def plot_decoder_projections(
    projections: dict,
    *,
    dims: int = 2,
    selection: list | None = None,
    title: str | None = None,
    use_cones_3d: bool = True,
    scale: float = 1.0,
    activity_rates: dict | None = None,
) -> go.Figure:
    """
    Plot decoder direction projections as arrows from the origin.

    projections: dict returned by project_decoder_directions:
      projections[site][sae_type][param][latent_idx] = np.ndarray (dims,)

    dims: 2 or 3 for 2D/3D plot
    selection: optional list of (site, sae_type, param, latent_idx) to plot;
               if None, plot all in projections
    title: optional plot title
    use_cones_3d: when dims==3, draw Cone arrows (anchored at tail) if True, otherwise line segments
    scale: multiply each vector by this factor for visualization scaling
    activity_rates: optional dict mapping (site, sae_type, param, latent_idx) -> activity fraction in [0,1]

    Returns: plotly.graph_objects.Figure
    """
    assert dims in (2, 3), "dims must be 2 or 3"

    # Collect vectors
    items = []  # list of (site, sae_type, param, latent_idx, vec)
    if selection is not None:
        for site, sae_type, param, li in selection:
            try:
                v = projections[site][sae_type][param][int(li)]
                if v.shape[0] >= dims:
                    items.append((site, sae_type, param, int(li), v[:dims] * scale))
            except Exception:
                continue
    else:
        for site, d1 in projections.items():
            for sae_type, d2 in d1.items():
                for param, d3 in d2.items():
                    for li, v in d3.items():
                        if v.shape[0] >= dims:
                            items.append((site, sae_type, param, int(li), v[:dims] * scale))

    fig = go.Figure()

    # color palette
    colors = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#FFFF33", "#A65628", "#F781BF", "#999999", "#66C2A5",
        "#E6AB02", "#A6CEE3", "#B2DF8A", "#FB9A99", "#CAB2D6",
        "#FFD92F", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    ]

    # Group by (site, sae_type, param) for legend clarity
    for idx, (site, sae_type, param, li, vec) in enumerate(items):
        color = colors[idx % len(colors)]
        # Build compact legend name: just latent number, optionally with activity rate
        rate_suffix = ""
        if activity_rates is not None:
            key = (site, sae_type, param, li)
            rate = activity_rates.get(key)
            if rate is None:
                # try string latent key variant if upstream stored strings
                rate = activity_rates.get((site, sae_type, param, int(li)))
            if rate is not None:
                try:
                    rate_suffix = f" ({float(rate):.1%})"
                except Exception:
                    pass
        name = f"latent {li}{rate_suffix}"

        if dims == 2:
            x0, y0 = 0.0, 0.0
            x1, y1 = float(vec[0]), float(vec[1])
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color, opacity=1.0),
                name=name,
            ))
        else:
            x0, y0, z0 = 0.0, 0.0, 0.0
            x1, y1, z1 = float(vec[0]), float(vec[1]), float(vec[2] if vec.shape[0] >= 3 else 0.0)
            if use_cones_3d:
                fig.add_trace(go.Cone(
                    x=[x0], y=[y0], z=[z0],
                    u=[x1], v=[y1], w=[z1],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.12,
                    anchor="tail",
                    name=name,
                ))
                # opaque legend marker
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers",
                    marker=dict(size=6, color=color, opacity=1.0),
                    name=name,
                    showlegend=True,
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode="lines+markers",
                    line=dict(color=color, width=4),
                    marker=dict(size=4, color=color, opacity=1.0),
                    name=name,
                ))

    if dims == 2:
        fig.update_layout(
            title=title or "Decoder directions (PCA space, 2D)",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(),
            height=700, width=800,
            legend=dict(font=dict(color="black")),
        )
    else:
        fig.update_layout(
            title=title or "Decoder directions (PCA space, 3D)",
            scene=dict(
                xaxis=dict(title="PC1"),
                yaxis=dict(title="PC2"),
                zaxis=dict(title="PC3"),
                aspectmode="cube",
            ),
            height=700, width=900,
            legend=dict(font=dict(color="black")),
        )

    return fig



def generate_mp_emissions(
    key,
    n_tom_quantum,
    n_mess3,
    tom_stationaries,
    mess3_stationaries,
    batch_size,
    seq_len,
    tom_quantum_processes,
    mess3_processes,
    tom_quantum_vocab_size,
    mess3_vocab_size,
    product_vocab_size,
    device,
):
    """
    Generate a new batch of factored inputs for tom_quantum and mess3 processes,
    and combine them into product space tokens.

    Args:
        key: JAX PRNGKey
        n_tom_quantum: int, number of tom_quantum processes
        n_mess3: int, number of mess3 processes
        tom_stationaries: list of stationary distributions for tom_quantum
        mess3_stationaries: list of stationary distributions for mess3
        batch_size: int, batch size
        seq_len: int, sequence length
        tom_quantum_processes: list of tom_quantum process objects
        mess3_processes: list of mess3 process objects
        tom_quantum_vocab_size: int, vocab size for tom_quantum
        mess3_vocab_size: int, vocab size for mess3
        product_vocab_size: int, total product vocab size
        device: torch device

    Returns:
        key: new JAX PRNGKey after split
        tom_inputs_list: list of np.ndarray, one per tom_quantum process
        mess3_inputs_list: list of np.ndarray, one per mess3 process
        tokens: torch.LongTensor, combined product tokens on device
    """
    key, *subkeys = jax.random.split(key, 1 + n_tom_quantum + n_mess3)

    tom_inputs_list = []
    for i in range(n_tom_quantum):
        tom_states = jnp.repeat(tom_stationaries[i][None, :], batch_size, axis=0)
        _, tom_inputs, _ = generate_data_batch(
            tom_states, tom_quantum_processes[i], batch_size, seq_len, subkeys[i]
        )
        if isinstance(tom_inputs, torch.Tensor):
            tom_inputs_list.append(tom_inputs.cpu().numpy())
        else:
            tom_inputs_list.append(np.array(tom_inputs))

    mess3_inputs_list = []
    for i in range(n_mess3):
        mess3_states = jnp.repeat(mess3_stationaries[i][None, :], batch_size, axis=0)
        _, mess3_inputs, _ = generate_data_batch(
            mess3_states, mess3_processes[i], batch_size, seq_len, subkeys[n_tom_quantum + i]
        )
        if isinstance(mess3_inputs, torch.Tensor):
            mess3_inputs_list.append(mess3_inputs.cpu().numpy())
        else:
            mess3_inputs_list.append(np.array(mess3_inputs))

    # Combine into product space: token = sum(input_i * base_i)
    # The base for each process will be the product of the vocab sizes of the processes that come after it in the combination.
    # For example, if we have tom1, tom2, mess3_1, mess3_2 with vocab sizes T1, T2, M1, M2
    # token = tom1 * (T2 * M1 * M2) + tom2 * (M1 * M2) + mess3_1 * M2 + mess3_2
    # Assuming all tom_quantum have the same vocab size (tom_quantum_vocab_size) and all mess3 have the same (mess3_vocab_size)
    # token = tom1 * (tom_quantum_vocab_size**(n_tom_quantum-1) * mess3_vocab_size**n_mess3) + ... + mess3_2

    # Initialize product_tokens based on whether there are any inputs
    if n_tom_quantum > 0:
        product_tokens = np.zeros_like(tom_inputs_list[0])
    elif n_mess3 > 0:
        product_tokens = np.zeros_like(mess3_inputs_list[0])
    else:
        raise ValueError("Both n_tom_quantum and n_mess3 are zero. Cannot generate data.")

    # Add tom_quantum contributions
    for i in range(n_tom_quantum):
        base_for_tom = (tom_quantum_vocab_size ** (n_tom_quantum - 1 - i)) * (mess3_vocab_size ** n_mess3)
        product_tokens += tom_inputs_list[i] * base_for_tom

    # Add mess3 contributions
    for i in range(n_mess3):
        base_for_mess3 = (mess3_vocab_size ** (n_mess3 - 1 - i))
        product_tokens += mess3_inputs_list[i] * base_for_mess3

    tokens = torch.from_numpy(product_tokens).long().to(device)

    return key, tom_inputs_list, mess3_inputs_list, tokens


def plot_pca_subplots(
    pca_coords,
    mess3_point_colors,
    tom_point_colors,
    pc_indices,
    mess3_label_to_color,
    tom_label_to_color,
    marker_size=3,
    legend_marker_size=8,
    opacity=0.6,
    height=1200,
    width=1200,
    show=True,
    title_text=None,
    output_dir="outputs/reports",
    save=None,
    n_points_to_plot=2000
):
    """
    Plots interactive PCA subplots for Mess3 and Tom Quantum labels.

    Args:
        pca_coords: np.ndarray, shape (n_points, n_pcs)
        mess3_point_colors: list of lists of color strings, one per Mess3 label set
        tom_point_colors: list of lists of color strings, one per Tom Quantum label set
        pc_indices: list or tuple of length 2 or 3, 0-indexed PC indices to plot
        mess3_label_to_color: dict mapping mess3 labels -> color
        tom_label_to_color: dict mapping tom labels -> color
        marker_size: int, size of data point markers
        legend_marker_size: int, size of legend marker circles
        n_points_to_plot: int, number of points to randomly sample for plotting
    """
    import numpy as np

    assert len(pc_indices) in (2, 3), "pc_indices must be length 2 or 3"
    is_3d = len(pc_indices) == 3

    n_total_points = pca_coords.shape[0]
    n_points = min(n_points_to_plot, n_total_points)
    rng = np.random.default_rng()
    sample_indices = rng.choice(n_total_points, size=n_points, replace=False)

    # Subset all relevant arrays/lists to the sampled indices
    pca_coords_sampled = pca_coords[sample_indices, :]
    mess3_point_colors_sampled = [np.array(color_list)[sample_indices] for color_list in mess3_point_colors]
    tom_point_colors_sampled = [np.array(color_list)[sample_indices] for color_list in tom_point_colors]

    pc_label_str = "-".join([f"{i+1}" for i in pc_indices])
    subplot_titles = (
        f"Mess3 #1 Coloring: PCs {pc_label_str}",
        f"Mess3 #2 Coloring: PCs {pc_label_str}",
        f"Mess3 #3 Coloring: PCs {pc_label_str}",
        f"Tom Quantum #1 Coloring: PCs {pc_label_str}",
        f"Tom Quantum #2 Coloring: PCs {pc_label_str}",
        "Blank"
    )

    scatter_type = 'scatter3d' if is_3d else 'scatter'
    specs = [[{'type': scatter_type}, {'type': scatter_type}],
             [{'type': scatter_type}, {'type': scatter_type}],
             [{'type': scatter_type}, {'type': scatter_type}]]

    fig = make_subplots(rows=3, cols=2, specs=specs, subplot_titles=subplot_titles)

    def marker_dict(color):
        return dict(size=marker_size, color=color, opacity=opacity, showscale=False)

    # --- Mess3 traces ---
    for i in range(3):
        if is_3d:
            trace = go.Scatter3d(
                x=pca_coords_sampled[:, pc_indices[0]],
                y=pca_coords_sampled[:, pc_indices[1]],
                z=pca_coords_sampled[:, pc_indices[2]],
                mode='markers',
                marker=marker_dict(mess3_point_colors_sampled[i]),
                showlegend=False
            )
        else:
            trace = go.Scatter(
                x=pca_coords_sampled[:, pc_indices[0]],
                y=pca_coords_sampled[:, pc_indices[1]],
                mode='markers',
                marker=marker_dict(mess3_point_colors_sampled[i]),
                showlegend=False
            )
        row, col = (1, i+1) if i < 2 else (2, 1)
        fig.add_trace(trace, row=row, col=col if i < 2 else 1)

    # --- Tom Quantum traces ---
    for i in range(2):
        if is_3d:
            trace = go.Scatter3d(
                x=pca_coords_sampled[:, pc_indices[0]],
                y=pca_coords_sampled[:, pc_indices[1]],
                z=pca_coords_sampled[:, pc_indices[2]],
                mode='markers',
                marker=marker_dict(tom_point_colors_sampled[i]),
                showlegend=False
            )
        else:
            trace = go.Scatter(
                x=pca_coords_sampled[:, pc_indices[0]],
                y=pca_coords_sampled[:, pc_indices[1]],
                mode='markers',
                marker=marker_dict(tom_point_colors_sampled[i]),
                showlegend=False
            )
        row, col = (2, 2) if i == 0 else (3, 1)
        fig.add_trace(trace, row=row, col=col)

    # --- Legend entries ---
    # Mess3 first
    for lbl, colr in mess3_label_to_color.items():
        name = f"Mess3 Label {lbl}"
        if is_3d:
            dummy = go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=legend_marker_size, color=colr, opacity=opacity),
                name=name,
                showlegend=True
            )
        else:
            dummy = go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=legend_marker_size, color=colr, opacity=opacity),
                name=name,
                showlegend=True
            )
        fig.add_trace(dummy, row=1, col=1)

    # Tom Quantum second
    for lbl, colr in tom_label_to_color.items():
        name = f"TomQ Label {lbl}"
        if is_3d:
            dummy = go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=legend_marker_size, color=colr, opacity=opacity),
                name=name,
                showlegend=True
            )
        else:
            dummy = go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=legend_marker_size, color=colr, opacity=opacity),
                name=name,
                showlegend=True
            )
        fig.add_trace(dummy, row=1, col=1)

    # --- Axis labels ---
    pc_labels = [f"PC{idx+1}" for idx in pc_indices]
    layout_kwargs = dict(height=height, width=width)
    if title_text is not None:
        layout_kwargs['title_text'] = title_text

    if is_3d:
        for i in range(1, 7):
            scene_name = f"scene{i}" if i > 1 else "scene"
            layout_kwargs[scene_name] = dict(
                xaxis_title=pc_labels[0],
                yaxis_title=pc_labels[1],
                zaxis_title=pc_labels[2]
            )
    else:
        for i in range(1, 7):
            layout_kwargs[f"xaxis{i}"] = dict(title=pc_labels[0])
            layout_kwargs[f"yaxis{i}"] = dict(title=pc_labels[1])

    fig.update_layout(**layout_kwargs)

    # --- Save logic ---
    if save is not None:
        if not isinstance(save, (list, tuple)):
            raise ValueError("save must be None or a list/tuple of 'png', 'html', 'json'")
        save_formats = [str(fmt).lower() for fmt in save]
        if save_formats:
            os.makedirs(output_dir, exist_ok=True)
            if title_text is not None:
                safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "_" for c in title_text)
                base_filename = f"pca_subplots_{safe_title.strip().replace(' ', '_')}_PCs_{pc_label_str}"
            else:
                base_filename = f"pca_subplots_PCs_{pc_label_str}"

            for fmt in save_formats:
                filepath = os.path.join(output_dir, base_filename + f".{fmt}")
                if fmt == "png":
                    fig.write_image(filepath)
                elif fmt == "html":
                    fig.write_html(filepath, include_plotlyjs=True, full_html=True)
                elif fmt == "json":
                    pio.write_json(fig, filepath)
                else:
                    print(f"Unknown save format: {fmt}")
                print(f"Figure saved to {filepath}")

    if show:
        fig.show()
    return fig





# ===== ePDFs ===== #
#####################

def sample_residual_stream_activations(model, mess3, seq_len, n_samples: int = 1000, device: str = "cuda"):
    """
    Get residual stream activations samples.
    """
    # Generate a batch for analysis
    key = jax.random.PRNGKey(43)
    key, subkey = jax.random.split(key)
    gen_states = jnp.repeat(mess3.initial_state[None, :], n_samples, axis=0)
    _, inputs, _ = generate_data_batch(gen_states, mess3, n_samples, seq_len, subkey)

    if isinstance(inputs, torch.Tensor):
        tokens = inputs.long().to(device)
    else:
        tokens = torch.from_numpy(np.array(inputs)).long().to(device)

    # Run with cache to get all activations
    logits, cache = model.run_with_cache(tokens)

    # Extract residual stream activations at different layers
    residual_streams = {
        'embeddings': cache['hook_embed'],  # Shape: [batch, seq, d_model]
        'layer_0': cache['blocks.0.hook_resid_post'],  # After first layer
        'layer_1': cache['blocks.1.hook_resid_post'],  # After second layer,
        'layer_2': cache['blocks.2.hook_resid_post'],  # After third layer
        'layer_3': cache['blocks.3.hook_resid_post'],  # After fourth layer
    }
    return residual_streams, tokens



class LatentEPDF:
    def __init__(self, site_name, sae_id, latent_idx, coords, weights):
        self.site_name = site_name
        self.sae_id = sae_id
        self.latent_idx = latent_idx
        self.coords = np.asarray(coords)
        self.weights = np.asarray(weights)
        self.kde = None

        # Detect triangle on init
        self.triangle_vertices = self._detect_triangle_vertices()

    def __str__(self):
        return f"({self.site_name}, {self.sae_id[0]}, {self.sae_id[1]}, {self.latent_idx})"

    def __repr__(self):
        return self.__str__()

    def fit_kde(self, bw_method="scott"):
        if self.coords.shape[0] < 5:
            return None
        self.kde = st.gaussian_kde(self.coords.T, weights=self.weights, bw_method=bw_method)
        return self.kde

    def evaluate_on_grid(self, grid_x, grid_y):
        if self.kde is None:
            self.fit_kde()
        xy = np.vstack([grid_x.ravel(), grid_y.ravel()])
        z = self.kde(xy).reshape(grid_x.shape)
        return z

    def _detect_triangle_vertices(self, tol=0.02, verbose=False):
        """Infer simplex triangle vertices from this latent's coords (robust to PCA noise)."""
        if self.coords.shape[0] < 3:
            if verbose:
                print("Not enough coordinates to detect triangle vertices.")
            return None

        all_coords = self.coords
        xmin, xmax = np.min(all_coords[:,0]), np.max(all_coords[:,0])
        ymin, ymax = np.min(all_coords[:,1]), np.max(all_coords[:,1])

        if verbose:
            print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

        candidates = []

        # Near xmin
        mask = np.isclose(all_coords[:,0], xmin, atol=tol)
        if mask.any():
            ys = all_coords[mask,1]
            if verbose:
                print(f"Near xmin mask: {mask}, ys: {ys}")
            candidates.append([xmin, ys.min()])
            candidates.append([xmin, ys.max()])

        # Near xmax
        mask = np.isclose(all_coords[:,0], xmax, atol=tol)
        if mask.any():
            ys = all_coords[mask,1]
            if verbose:
                print(f"Near xmax mask: {mask}, ys: {ys}")
            candidates.append([xmax, ys.min()])
            candidates.append([xmax, ys.max()])

        # Near ymin
        mask = np.isclose(all_coords[:,1], ymin, atol=tol)
        if mask.any():
            xs = all_coords[mask,0]
            if verbose:
                print(f"Near ymin mask: {mask}, xs: {xs}")
            candidates.append([xs.min(), ymin])
            candidates.append([xs.max(), ymin])

        # Near ymax
        mask = np.isclose(all_coords[:,1], ymax, atol=tol)
        if mask.any():
            xs = all_coords[mask,0]
            if verbose:
                print(f"Near ymax mask: {mask}, xs: {xs}")
            candidates.append([xs.min(), ymax])
            candidates.append([xs.max(), ymax])

        candidates = np.unique(np.array(candidates), axis=0)
        if verbose:
            print(f"Unique candidate vertices: {candidates}")

        if candidates.shape[0] < 3:
            if verbose:
                print("Not enough unique candidates to form a triangle.")
            return None

        hull = ConvexHull(candidates)
        verts = candidates[hull.vertices]
        if verbose:
            print(f"Convex hull vertices: {verts}")

        # Force to exactly 3 vertices if noisy hull produced >3
        if verts.shape[0] > 3:
            dists = np.sum((verts[:, None, :] - verts[None, :, :])**2, axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            if verbose:
                print(f"Max distance between hull vertices: {i}, {j}")

            def area(p, q, r):
                return abs(0.5 * ((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])))

            best_k, best_area = None, -1
            for k in range(len(verts)):
                if k in (i, j):
                    continue
                A = area(verts[i], verts[j], verts[k])
                if verbose:
                    print(f"Area for triangle ({i}, {j}, {k}): {A}")
                if A > best_area:
                    best_area, best_k = A, k

            verts = np.array([verts[i], verts[j], verts[best_k]])
            if verbose:
                print(f"Selected triangle vertices: {verts}")

        return verts

def _plot_hull_and_triangles(hull_pts, all_candidates, epdfs, sae_param, sae_type):
    plt.figure(figsize=(6, 6))
    plt.scatter(hull_pts[:, 0], hull_pts[:, 1], color='black', label='Hull Points')
    for i in range(hull_pts.shape[0]):
        plt.text(hull_pts[i, 0], hull_pts[i, 1], str(i), color='black', fontsize=10)

    # Draw each original triangle from all_candidates
    for tri in all_candidates:
        tri = np.asarray(tri)
        if tri.shape[0] == 3:
            closed_tri = np.vstack([tri, tri[0]])  # close the triangle
            plt.plot(closed_tri[:, 0], closed_tri[:, 1], alpha=0.7)

    plt.title(f"Original triangles from all_candidates (layer {epdfs[0].site_name[6:]}, param={sae_param}, type={sae_type})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()


def get_global_triangle_vertices(epdfs):
    """
    Build a global triangle that covers all EPDFs from the same SAE by taking the
    convex hull of all per-EPDF triangle vertices and selecting the maximum-area triangle
    from that hull.
    """
    all_candidates = []
    for e in epdfs:
        tv = getattr(e, "triangle_vertices", None)
        if tv is None:
            tv = e._detect_triangle_vertices(verbose=False)
        if tv is None:
            e_id = getattr(e, "sae_id", None)
            e_type = type(e).__name__
            e_site = getattr(e, "site_name", None)
            warnings.warn(
                f"Could not detect triangle vertices for EPDF: "
                f"sae_id={e_id}, type={e_type}, site_name={e_site}, latent_idx={e.latent_idx}"
            )
            continue
        else:
            all_candidates.append(np.asarray(tv))
    if not all_candidates:
        raise ValueError("Could not detect triangle vertices for plotting.")
    pts = np.vstack(all_candidates)
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except Exception:
        hull_pts = pts
    if hull_pts.shape[0] < 3:
        raise ValueError("Insufficient hull vertices to define a triangle.")
    if hull_pts.shape[0] == 3:
        triangle_vertices = hull_pts
    else:
        sae_type = "l1" if "lambda" in epdfs[0].sae_id[1] else "top_k"
        if sae_type == "l1":
            sae_param = epdfs[0].sae_id[1][7:]
        else:
            sae_param = epdfs[0].sae_id[1][1:]
        print(f"Got more than 3 hull vertices for layer {epdfs[0].site_name[6:]}, {sae_type} SAE with parameter {sae_param}")
        if False:
            _plot_hull_and_triangles(hull_pts, all_candidates, epdfs, sae_param, sae_type)
        
        def tri_area(a, b, c):
            return abs(0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])))
        best = None
        best_area = -1.0
        for i, j, k in combinations(range(hull_pts.shape[0]), 3):
            A = tri_area(hull_pts[i], hull_pts[j], hull_pts[k])
            if A > best_area:
                best_area = A
                best = (hull_pts[i], hull_pts[j], hull_pts[k])
        triangle_vertices = np.array(best)
    return triangle_vertices



def plot_epdfs(epdfs, mode="3d", grid_size=150, active_latents=None, triangle_vertices=None, color_start_ind=None):
    """
    epdfs: list[LatentEPDF] (must be from the same SAE if multiple)
    mode: "3d", "transparency", or "contours"
    """
    import matplotlib.tri as tri

    if not isinstance(epdfs, (list, tuple)):
        epdfs = [epdfs]

    # Consistency check
    sae_ids = {e.sae_id for e in epdfs}
    if len(sae_ids) > 1 and mode == "3d":
        raise ValueError("3D plotting requires all ePDFs from the same SAE.")


    # Use the new function in plot_epdfs
    if triangle_vertices is None:
        triangle_vertices = get_global_triangle_vertices(epdfs)

    # Generate a triangular grid inside detected simplex
    pts_x, pts_y = [], []
    for i in range(grid_size + 1):
        for j in range(grid_size + 1 - i):
            a = i / grid_size
            b = j / grid_size
            c = 1 - a - b
            x = a * triangle_vertices[0, 0] + b * triangle_vertices[1, 0] + c * triangle_vertices[2, 0]
            y = a * triangle_vertices[0, 1] + b * triangle_vertices[1, 1] + c * triangle_vertices[2, 1]
            pts_x.append(x)
            pts_y.append(y)
    pts_x, pts_y = np.array(pts_x), np.array(pts_y)

    fig = go.Figure()
    # 20 maximally distinctive colors (colorblind-friendly palette)
    colors = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#FFFF33",  # yellow
        "#A65628",  # brown
        "#F781BF",  # pink
        "#999999",  # gray
        "#66C2A5",  # teal
        "#E6AB02",  # mustard
        "#A6CEE3",  # light blue
        "#B2DF8A",  # light green
        "#FB9A99",  # salmon
        "#CAB2D6",  # lavender
        "#FFD92F",  # gold
        "#1B9E77",  # dark green
        "#D95F02",  # dark orange
        "#7570B3",  # indigo
        "#E7298A",  # magenta
    ]

    for idx, epdf in enumerate(epdfs):
        if epdf.kde is None:
            epdf.fit_kde()
        xy = np.vstack([pts_x, pts_y])
        z = epdf.kde(xy)

        if z is None or z.size == 0:
            continue
        color_ind = (idx + (color_start_ind if color_start_ind is not None else 0)) % len(colors)
        color = colors[color_ind]

        # Normalize inside triangle
        z_min, z_max = np.min(z), np.max(z)
        z_norm = (z - z_min) / (z_max - z_min + 1e-12)

        legend_name = f"Latent {epdf.latent_idx}"
        if active_latents is not None and str(epdf.latent_idx) in active_latents:
            legend_name += f", active rate: {active_latents[str(epdf.latent_idx)][0]:.1%}"
        if mode == "3d":
            fig.add_trace(go.Mesh3d(
                x=pts_x, y=pts_y, z=z_norm,
                color=color,
                opacity=0.6,
                name=legend_name,
                showlegend=False,
                alphahull=0
            ))
            # Opaque legend marker
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=6, color=color, opacity=1.0),
                name=legend_name,
                showlegend=True
            ))
        elif mode == "transparency":
            fig.add_trace(go.Scatter(
                x=pts_x, y=pts_y,
                mode="markers",
                marker=dict(size=5, color=color, opacity=z_norm),
                name=legend_name,
                showlegend=False
            ))
            # Opaque legend marker
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=8, color=color, opacity=1.0),
                name=legend_name,
                showlegend=True
            ))
        elif mode == "contours":
            triang = tri.Triangulation(pts_x, pts_y)
            fig.add_trace(go.Contour(
                x=pts_x,
                y=pts_y,
                z=z_norm,
                colorscale=[[0, color], [1, color]],
                contours_coloring="lines",
                line_width=2,
                name=legend_name,
                connectgaps=False
            ))

    # Add simplex boundary
    tri_x = np.append(triangle_vertices[:, 0], triangle_vertices[0, 0])
    tri_y = np.append(triangle_vertices[:, 1], triangle_vertices[0, 1])
    fig.add_trace(go.Scatter(
        x=tri_x, y=tri_y, mode="lines",
        line=dict(color="black"), showlegend=False
    ))

    fig.update_layout(
        title=f"Empirical PDFs for {epdfs[0].site_name} / SAE {epdfs[0].sae_id}",
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=700, width=800,
        legend=dict(font=dict(color="black"))
    )
    return fig



def build_epdfs_for_sae(
    model,
    sae,
    mess3,
    site_name: str,
    sae_type: str,
    param_value: str,
    active_latent_indices,
    pcas: dict,
    seq_len: int = 10,
    n_batches: int = 100,
    device: str = "cpu",
    hook_name: str | None = None,
) -> dict:
    """
    Build LatentEPDFs for all non-dead latents in one SAE.

    Returns:
      {site_name: {sae_type: {param_value: {latent_idx: LatentEPDF}}}}
    """

    # Infer hook name if not provided
    if hook_name is None:
        if site_name == "embeddings":
            hook_name = "hook_embed"
        else:
            layer_idx = int(site_name.split("_")[1])
            hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Generate activation samples
    residuals, tokens = sample_residual_stream_activations(model, mess3, seq_len=seq_len, n_samples=n_batches, device=device)
    acts = residuals[site_name].reshape(-1, residuals[site_name].shape[-1]).to(device)

    # Run through SAE
    with torch.no_grad():
        sae_out = sae(acts)
        latents = sae_out["feature_acts"].cpu().numpy()

    # 🔹 CHANGE: use layer-specific PCA
    if site_name not in pcas:
        raise KeyError(f"No PCA provided for site {site_name}")
    pca_proj = pcas[site_name]

    coords = pca_proj.transform(acts.cpu().numpy())

    # Normalize active latent indices: can be list/array or dict {idx: [count, sum]}
    if isinstance(active_latent_indices, dict):
        try:
            indices = [int(k) for k in active_latent_indices.keys()]
        except Exception:
            indices = list(active_latent_indices.keys())
    else:
        indices = [int(i) for i in active_latent_indices]

    # Build EPDFs
    latent_map = {}
    for li in indices:
        li_int = int(li)
        if li_int < 0 or li_int >= latents.shape[1]:
            continue
        mask = latents[:, li_int] > 1e-6
        if not np.any(mask):
            continue
        latent_map[li_int] = LatentEPDF(
            site_name=site_name,
            sae_id=(sae_type, param_value),
            latent_idx=li_int,
            coords=coords[mask],
            weights=latents[mask, li_int],
        )

    return {site_name: {sae_type: {param_value: latent_map}}}


def build_epdfs_for_all_saes(
    model,
    mess3,
    loaded_saes: dict,
    active_latents: dict,
    pcas: dict,
    seq_len: int = 10,
    n_batches: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Loop over all loaded_saes and build EPDFs.

    Returns:
      sequence_epdfs[site_name][sae_type][param_value][latent_idx] = LatentEPDF
    """

    sequence_epdfs = {}

    for site_name, sae_dict in loaded_saes.items():
        sequence_epdfs[site_name] = {}
        for (k, lmbda), sae_ckpt in sae_dict.items():
            if k is not None:
                sae_class = TopKSAE
                sae_type = "top_k"
                param_value = f"k{k}"
                active_key = (site_name, "sequence", "top_k", f"k{k}")
            else:
                sae_class = VanillaSAE
                sae_type = "l1"
                param_value = f"lambda_{lmbda}"
                active_key = (site_name, "sequence", "vanilla", f"lambda_{lmbda}")

            sae = sae_class(sae_ckpt["cfg"])
            sae.load_state_dict(sae_ckpt["state_dict"])
            sae.to(device).eval()

            active_inds = active_latents.get(active_key, [])
            if not active_inds:
                continue

            # 🔹 CHANGE: pass pcas dict instead of single PCA
            epdfs_dict = build_epdfs_for_sae(
                model,
                sae,
                mess3,
                site_name,
                sae_type,
                param_value,
                active_inds,
                pcas,
                seq_len=seq_len,
                n_batches=n_batches,
                device=device,
            )

            # Merge into top-level dict
            for site, type_map in epdfs_dict.items():
                for sae_type, param_map in type_map.items():
                    sequence_epdfs[site].setdefault(sae_type, {})
                    for param_value, latent_map in param_map.items():
                        sequence_epdfs[site][sae_type][param_value] = latent_map

    return sequence_epdfs


# ==== Some Spectral Clustering Utils ===== #
#############################################


def phi_similarity(latent_acts, efficient=True, eps=1e-12):
    """
    Compute the Phi (Pearson) correlation coefficient between all pairs of binary
    latent activations.

    Args:
        latent_acts : (N_samples, N_latents) array-like
            Binary activations (0/1 or bool). Nonzero is treated as 1.
        efficient : bool
            If True, use matrix multiplies (fast, memory-heavy).
            If False, use a simple double loop (slow, memory-light).
        eps : float
            Numerical guard for denominators.

    Returns:
        Phi : (N_latents, N_latents) float64 array, symmetric, diag=1.
    """
    A = (latent_acts > eps).astype(np.int64)            # N x p
    N, p = A.shape

    # This is slower but more memory efficient
    if not efficient:
        Phi = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            ai = A[:, i]
            for j in range(i, p):
                aj = A[:, j]
                n11 = np.sum((ai == 1) & (aj == 1))
                n10 = np.sum((ai == 1) & (aj == 0))
                n01 = np.sum((ai == 0) & (aj == 1))
                n00 = np.sum((ai == 0) & (aj == 0))
                denom = np.sqrt((n11+n10)*(n11+n01)*(n00+n10)*(n00+n01)) + eps
                phi = (n11*n00 - n10*n01) / denom
                Phi[i, j] = Phi[j, i] = phi
        np.fill_diagonal(Phi, 1.0)
        return Phi

    # Efficient path: matrix multiplies
    Ac = 1 - A

    # Co-activations
    N11 = (A.T @ A).astype(np.float64)

    # i=1, j=0
    N10 = (A.T @ Ac).astype(np.float64)

    # Symmetry: N01 = N10.T
    # N01 = N10.T

    # Both off
    N00 = N - (N11 + N10 + N10.T)

    # Phi coefficient
    denom = np.sqrt((N11+N10) * (N00+N10.T) * (N11+N10.T) * (N00+N10)) + eps
    Phi = (N11 * N00 - N10 * N10.T) / denom

    # Clean up numerical junk
    Phi[~np.isfinite(Phi)] = 0.0
    np.fill_diagonal(Phi, 1.0)
    np.clip(Phi, -1.0, 1.0, out=Phi)
    return Phi



def build_similarity_matrix(data, method="cosine", latent_acts=None, phi_compute_efficient=True):
    if method == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(data)

    elif method == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        D = euclidean_distances(data)
        return 1.0 / (1.0 + D)

    elif method == "phi":
        if latent_acts is None:
            raise ValueError("Phi similarity requires latent_acts.")
        return phi_similarity(latent_acts, efficient=phi_compute_efficient)

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def spectral_clustering_with_eigengap(sim_matrix, max_clusters=10, random_state=0, plot=False, plot_path=None):
    """
    Run spectral clustering using dense similarity matrix and eigengap heuristic.

    Args:
        sim_matrix (ndarray): n x n similarity matrix (cosine sim).
        max_clusters (int): check eigengaps among first max_clusters eigenvalues.
        random_state (int): for k-means stability.
        plot (bool): if True, plot eigenvalues and highlight the chosen gap.

    Returns:
        labels (ndarray): cluster assignments for each point.
        best_k (int): chosen number of clusters.
    """
    # Degree matrix and normalized Laplacian
    diag = np.diag(sim_matrix.sum(axis=1))
    laplacian = diag - sim_matrix
    sqrt_deg = np.diag(1.0 / np.sqrt(np.maximum(diag.diagonal(), 1e-12)))
    norm_lap = sqrt_deg @ laplacian @ sqrt_deg

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(norm_lap)
    eigvals = np.real(eigvals)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Eigengap heuristic
    usable_count = min(eigvals.shape[0], max_clusters + 1)
    gaps = np.diff(eigvals[:usable_count])
    if gaps.size == 0:
        best_k = 1
    else:
        best_k = int(np.argmax(gaps) + 1)
    print(f"Chosen number of clusters via eigengap: k={best_k}")

    if plot or plot_path is not None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        x_len = usable_count
        ax.plot(range(1, x_len + 1), eigvals[:usable_count], marker="o")
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Eigenvalue spectrum (normalized Laplacian)")
        # Highlight chosen eigengap
        ax.axvline(best_k, color="red", linestyle="--", label=f"Chosen k={best_k}")
        ax.legend()
        fig.tight_layout()
        if plot_path is not None:
            import os
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            fig.savefig(plot_path)
            plt.close(fig)
        elif plot:
            plt.show()

    # Spectral embedding
    embedding = eigvecs[:, :best_k]
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    # K-means on embedding
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(embedding)

    return labels, best_k


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{hex_color}'")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float64)


def project_simplex3_to_2d(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map 3-simplex points to 2D barycentric coordinates."""

    arr = _to_numpy_array(probs)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array; got shape {arr.shape}.")
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, float(np.sqrt(3.0) / 2.0)])
    x = arr[:, 0] * v1[0] + arr[:, 1] * v2[0] + arr[:, 2] * v3[0]
    y = arr[:, 0] * v1[1] + arr[:, 1] * v2[1] + arr[:, 2] * v3[1]
    return x, y


def _to_numpy_array(array: Any) -> np.ndarray:
    """Best-effort conversion of tensors/arrays to a float numpy array."""

    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if isinstance(array, jax.Array):
        return np.asarray(array)
    return np.asarray(array)


def project_vectors_onto_simplex(vectors: Any, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Project each vector onto the probability simplex along ``axis``.

    Uses the algorithm from [Wang & Carreira-Perpiñán, 2013]. Handles arbitrary
    leading dimensions by working on a reshaped view.
    """

    arr = _to_numpy_array(vectors).astype(np.float64, copy=True)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
        reshape_back = True
    else:
        reshape_back = False
        if axis != -1:
            arr = np.moveaxis(arr, axis, -1)

    flat = arr.reshape(-1, arr.shape[-1])
    sorted_vec = np.sort(flat, axis=1)[:, ::-1]
    cssv = np.cumsum(sorted_vec, axis=1)
    rhos = np.sum(sorted_vec * np.arange(1, flat.shape[1] + 1) > (cssv - 1), axis=1) - 1
    theta = (cssv[np.arange(flat.shape[0]), rhos] - 1) / (rhos + 1)
    projected = np.maximum(flat - theta[:, None], 0.0)
    projected = np.maximum(projected, 0.0)
    projected /= np.maximum(projected.sum(axis=1, keepdims=True), eps)
    projected = projected.reshape(arr.shape)

    if axis != -1 and not reshape_back:
        projected = np.moveaxis(projected, -1, axis)

    if reshape_back:
        projected = projected[0]
    return projected


def enforce_tom_quantum_physicality(params: Any, eps: float = 1e-9) -> np.ndarray:
    """Clamp Tom Quantum belief parameters to represent a valid density matrix."""

    arr = _to_numpy_array(params).astype(np.float64, copy=True)
    if arr.shape[-1] != 3:
        raise ValueError("Tom Quantum beliefs are expected to have last dimension size 3")

    a = np.clip(arr[..., 0], eps, 1.0 - eps)
    coherence = arr[..., 1:3]
    max_coh = np.sqrt(np.maximum(a * (1.0 - a), eps))
    coh_norm = np.linalg.norm(coherence, axis=-1)
    scale = np.ones_like(coh_norm)
    mask = coh_norm > max_coh
    scale[mask] = max_coh[mask] / coh_norm[mask]
    coherence = coherence * scale[..., None]

    arr[..., 0] = a
    arr[..., 1:3] = coherence
    return arr


def tom_quantum_params_to_bloch(coords: Any) -> np.ndarray:
    """Convert Tom Quantum belief parameters to Bloch sphere coordinates."""

    arr = _to_numpy_array(coords).astype(np.float64)
    if arr.shape[-1] != 3:
        raise ValueError("Expected Tom Quantum params with last dimension 3")
    x = 2.0 * arr[..., 1]
    y = -2.0 * arr[..., 2]
    z = 2.0 * arr[..., 0] - 1.0
    return np.stack([x, y, z], axis=-1)


def evaluate_belief_state_linear_models(
    activations: Any,
    component_belief_arrays: Sequence[Any],
    component_metadata: Sequence[Mapping[str, Any]],
    seq_positions: Sequence[int],
    *,
    skip_dims_by_type: Mapping[str, Sequence[int]] | None = None,
    min_variance: float = 1e-8,
    postprocess_by_type: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    store_predictions: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fit linear models that predict belief states from activations.

    Args:
        activations: Array-like of shape ``(batch, seq, hidden_dim)`` representing
            activations (typically the final residual stream) for an evaluation batch.
        component_belief_arrays: List of arrays, one per component, each of shape
            ``(batch, seq, belief_dim)`` holding the true belief states.
        component_metadata: Metadata describing each component. Each entry must at
            minimum provide ``{"name": str, "type": str}``.
        seq_positions: Iterable of sequence indices (0-based) at which to compute
            regressions.
        skip_dims_by_type: Optional mapping ``component_type -> iterable of belief
            dimension indices`` to exclude before fitting. Useful for known-constant
            coordinates.
        min_variance: If the variance of a belief coordinate at a given position
            falls below this threshold it will be skipped to avoid numerical issues.
        postprocess_by_type: Optional mapping from component type to a callable
            that is applied to the full belief prediction (e.g. simplex projection).
        store_predictions: If True, include the post-processed predictions and
            the corresponding targets for each component/position in the returned
            metrics dictionary (useful for downstream visualizations).

    Returns:
        Dictionary keyed by component name. Each entry contains the component type,
        original belief dimensionality, and a nested dictionary of per-position
        regression metrics (average R^2, per-dimension R^2 and correlation, RMSE,
        MAE, and bookkeeping on skipped dimensions).
    """

    skip_dims_by_type = skip_dims_by_type or {}
    postprocess_by_type = postprocess_by_type or {}

    acts_np = _to_numpy_array(activations)
    if acts_np.ndim != 3:
        raise ValueError(
            f"Expected activations with shape (batch, seq, hidden_dim); got {acts_np.shape}."
        )

    batch_size, seq_len, hidden_dim = acts_np.shape
    positions = list(seq_positions)
    for pos in positions:
        if pos < 0 or pos >= seq_len:
            raise IndexError(
                f"Sequence position {pos} is out of bounds for activations with length {seq_len}."
            )

    if len(component_belief_arrays) != len(component_metadata):
        raise ValueError(
            "component_belief_arrays and component_metadata must have the same length."
        )

    results: Dict[str, Dict[str, Any]] = {}

    for beliefs_arr, meta in zip(component_belief_arrays, component_metadata, strict=True):
        beliefs_np = _to_numpy_array(beliefs_arr)
        if beliefs_np.ndim != 3:
            raise ValueError(
                f"Belief array for component {meta.get('name', '<unknown>')} must have shape"
                f" (batch, seq, belief_dim); got {beliefs_np.shape}."
            )
        if beliefs_np.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch between activations ({batch_size}) and beliefs"
                f" ({beliefs_np.shape[0]}) for component {meta.get('name', '<unknown>')}."
            )
        if beliefs_np.shape[1] <= max(positions):
            raise ValueError(
                f"Belief sequence length {beliefs_np.shape[1]} is insufficient for requested"
                f" positions {positions}."
            )

        comp_name = str(meta.get("name", "<unknown>"))
        comp_type = str(meta.get("type", "unknown"))
        belief_dim = int(beliefs_np.shape[-1])

        explicit_skip = set(skip_dims_by_type.get(comp_type, ()))
        valid_candidates = [idx for idx in range(belief_dim) if idx not in explicit_skip]

        per_position: Dict[int, Dict[str, Any]] = {}

        for pos in positions:
            X = acts_np[:, pos, :]
            belief_slice = beliefs_np[:, pos, :]

            usable_dims: list[int] = []
            auto_skip: list[int] = []
            for dim_idx in valid_candidates:
                if float(np.var(belief_slice[:, dim_idx])) > float(min_variance):
                    usable_dims.append(dim_idx)
                else:
                    auto_skip.append(dim_idx)

            if not usable_dims:
                per_position[pos] = {
                    "target_dims": [],
                    "explicitly_dropped_dims": sorted(explicit_skip),
                    "dropped_low_variance_dims": sorted(auto_skip),
                    "note": "No belief dimensions with variance above threshold",
                }
                continue

            y = belief_slice[:, usable_dims]
            model = LinearRegression()
            model.fit(X, y)

            prediction_full = belief_slice.copy()
            prediction_full[:, usable_dims] = model.predict(X)
            post_fn = postprocess_by_type.get(comp_type)
            if post_fn is not None:
                prediction_full = post_fn(prediction_full)

            y_pred = prediction_full[:, usable_dims]
            residual = y - y_pred

            rmse = float(np.sqrt(np.mean(residual ** 2)))
            mae = float(np.mean(np.abs(residual)))

            r2_per_dim: Dict[int, float] = {}
            corr_per_dim: Dict[int, float] = {}
            for local_idx, original_dim in enumerate(usable_dims):
                true_vals = y[:, local_idx]
                pred_vals = y_pred[:, local_idx]
                try:
                    r2_val = float(r2_score(true_vals, pred_vals))
                except ValueError:
                    r2_val = float("nan")
                with np.errstate(invalid="ignore"):
                    corr_matrix = np.corrcoef(true_vals, pred_vals)
                    corr_val = corr_matrix[0, 1] if corr_matrix.ndim == 2 else float("nan")
                r2_per_dim[original_dim] = r2_val
                corr_per_dim[original_dim] = float(corr_val)

            r2_values = [val for val in r2_per_dim.values() if np.isfinite(val)]
            r2_mean = float(np.mean(r2_values)) if r2_values else float("nan")

            entry: Dict[str, Any] = {
                "target_dims": usable_dims,
                "explicitly_dropped_dims": sorted(explicit_skip),
                "dropped_low_variance_dims": sorted(auto_skip),
                "r2_mean": r2_mean,
                "r2_per_dim": r2_per_dim,
                "pearson_per_dim": corr_per_dim,
                "rmse": rmse,
                "mae": mae,
                "coef_norm": float(np.linalg.norm(model.coef_)),
            }

            if store_predictions:
                entry["predictions"] = prediction_full.astype(np.float32, copy=False)
                entry["targets"] = belief_slice.astype(np.float32, copy=False)

            per_position[pos] = entry

        results[comp_name] = {
            "type": comp_type,
            "belief_dim": belief_dim,
            "activation_dim": hidden_dim,
            "positions_evaluated": positions,
            "metrics": per_position,
        }

    return results


def _sample_single_component_beliefs(
    component_config: Mapping[str, Any],
    *,
    batch_size: int = 4096,
    seq_len: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample beliefs and observations for a single-component process configuration."""

    components = build_components_from_config([component_config])
    sampler = MultipartiteSampler(components)
    key = jax.random.PRNGKey(seed)
    key, beliefs, _, component_obs = sampler.sample(key, batch_size, seq_len)
    beliefs_np = np.asarray(beliefs)
    obs_np = np.asarray(component_obs[0]) if component_obs else np.empty((batch_size, seq_len - 1))
    return beliefs_np, obs_np


def plot_mess3_belief_grid(
    x_values: Sequence[float],
    a_values: Sequence[float],
    *,
    seq_position: int,
    batch_size: int = 4096,
    seq_len: int = 16,
    seed: int = 0,
    sample_size: int = 2000,
) -> plt.Figure:
    """Plot Mess3 belief simplices for a grid of ``(x, a)`` parameter choices."""

    x_list = list(x_values)
    a_list = list(a_values)
    if not x_list or not a_list:
        raise ValueError("x_values and a_values must be non-empty sequences")

    fig, axes = plt.subplots(len(x_list), len(a_list), figsize=(3.2 * len(a_list), 3.2 * len(x_list)))
    if len(x_list) == 1 and len(a_list) == 1:
        axes = np.array([[axes]])
    elif len(x_list) == 1 or len(a_list) == 1:
        axes = np.atleast_2d(axes)

    tri_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, float(np.sqrt(3.0) / 2.0)]])
    tri_x = np.r_[tri_vertices[:, 0], tri_vertices[0, 0]]
    tri_y = np.r_[tri_vertices[:, 1], tri_vertices[0, 1]]
    base_palette = ["#d62728", "#ffd92f", "#9467bd"]
    base_rgb = np.stack([_hex_to_rgb(color) for color in base_palette], axis=0)

    for i, x in enumerate(x_list):
        for j, a in enumerate(a_list):
            beliefs_np, obs_np = _sample_single_component_beliefs(
                {"type": "mess3", "instances": [{"x": float(x), "a": float(a)}]},
                batch_size=batch_size,
                seq_len=seq_len,
                seed=seed + i * len(a_list) + j,
            )
            if beliefs_np.ndim != 3 or beliefs_np.shape[-1] != 3:
                raise ValueError(f"Unexpected Mess3 belief shape {beliefs_np.shape}")
            seq_idx = min(seq_position, beliefs_np.shape[1] - 1)
            simplex_pts = beliefs_np[:, seq_idx, :]
            # Continuous color blending using barycentric coordinates
            color_vectors = simplex_pts @ base_rgb
            color_vectors = np.clip(color_vectors, 0.0, 1.0)

            rng = np.random.default_rng(seed + 17 * (i * len(a_list) + j))
            n_points = simplex_pts.shape[0]
            sample_idx = rng.choice(n_points, size=min(sample_size, n_points), replace=False)

            xs, ys = project_simplex3_to_2d(simplex_pts[sample_idx, :])
            ax = axes[i, j]
            ax.plot(tri_x, tri_y, color="black", linewidth=0.8)
            ax.scatter(xs, ys, c=color_vectors[sample_idx], s=10, alpha=0.9, edgecolors="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"x={x:.2f}, a={a:.2f}")

    fig.suptitle(f"Mess3 belief geometry @ position {seq_position}", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def plot_tom_quantum_coherence(
    alpha: float,
    beta: float,
    *,
    seq_position: int,
    batch_size: int = 4096,
    seq_len: int = 16,
    seed: int = 0,
    sample_size: int = 4000,
) -> plt.Figure:
    """Plot Tom Quantum coherence plane (real vs imaginary) for given parameters."""

    beliefs_np, obs_np = _sample_single_component_beliefs(
        {"type": "tom_quantum", "instances": [{"alpha": float(alpha), "beta": float(beta)}]},
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
    )
    if beliefs_np.ndim != 3 or beliefs_np.shape[-1] != 3:
        raise ValueError(f"Unexpected Tom Quantum belief shape {beliefs_np.shape}")

    seq_idx = min(seq_position, beliefs_np.shape[1] - 1)
    coherence = beliefs_np[:, seq_idx, 1:3]
    obs = obs_np[:, seq_idx] if obs_np.size else np.zeros(coherence.shape[0], dtype=int)

    rng = np.random.default_rng(seed)
    n_points = coherence.shape[0]
    sample_idx = rng.choice(n_points, size=min(sample_size, n_points), replace=False)

    # Continuous color based on azimuthal angle
    angles = np.arctan2(coherence[:, 1], coherence[:, 0])
    angle_norm = (angles + np.pi) / (2 * np.pi)
    cmap = plt.get_cmap("twilight")
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(
        coherence[sample_idx, 0],
        coherence[sample_idx, 1],
        c=cmap(angle_norm[sample_idx]),
        s=16,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlabel("Re coherence")
    ax.set_ylabel("Im coherence")
    lim = np.max(np.abs(coherence[sample_idx])) * 1.05
    lim = max(lim, 0.05)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_title(f"TomQ coherence (alpha={alpha:.2f}, beta={beta:.2f}, pos {seq_position})")
    fig.tight_layout()
    return fig


def plot_tom_quantum_coherence_grid(
    alpha_values: Sequence[float],
    beta_values: Sequence[float],
    *,
    seq_position: int,
    batch_size: int = 4096,
    seq_len: int = 16,
    seed: int = 0,
    sample_size: int = 3000,
) -> plt.Figure:
    """Plot a grid of Tom Quantum coherence scatter plots across parameter choices."""

    alphas = list(alpha_values)
    betas = list(beta_values)
    if not alphas or not betas:
        raise ValueError("alpha_values and beta_values must both be non-empty sequences")

    fig, axes = plt.subplots(
        len(alphas),
        len(betas),
        figsize=(3.2 * len(betas), 3.2 * len(alphas)),
        squeeze=False,
    )

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            beliefs_np, obs_np = _sample_single_component_beliefs(
                {"type": "tom_quantum", "instances": [{"alpha": float(alpha), "beta": float(beta)}]},
                batch_size=batch_size,
                seq_len=seq_len,
                seed=seed + i * len(betas) + j,
            )

            if beliefs_np.ndim != 3 or beliefs_np.shape[-1] != 3:
                raise ValueError(f"Unexpected Tom Quantum belief shape {beliefs_np.shape}")

            seq_idx = min(seq_position, beliefs_np.shape[1] - 1)
            coherence = beliefs_np[:, seq_idx, 1:3]

            rng = np.random.default_rng(seed + 23 * (i * len(betas) + j))
            n_points = coherence.shape[0]
            sample_idx = rng.choice(n_points, size=min(sample_size, n_points), replace=False)
            sub = coherence[sample_idx]

            angles = np.arctan2(sub[:, 1], sub[:, 0])
            cmap = plt.get_cmap("twilight")
            colors = cmap((angles + np.pi) / (2 * np.pi))

            ax = axes[i, j]
            ax.scatter(sub[:, 0], sub[:, 1], c=colors, s=12, alpha=0.85, edgecolors="none")
            ax.set_xlim(-0.35, 0.35)
            ax.set_ylim(-0.35, 0.35)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.25, alpha=0.35)
            if i == len(alphas) - 1:
                ax.set_xlabel("Re coherence")
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel("Im coherence")
            else:
                ax.set_yticklabels([])
            ax.set_title(f"α={alpha:.2f}, β={beta:.2f}", fontsize=9)

    fig.suptitle(f"TomQ coherence grid @ position {seq_position}", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig
