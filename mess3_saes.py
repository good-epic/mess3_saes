import os
from typing import Dict, Tuple

import torch
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from BatchTopK.sae import VanillaSAE, TopKSAE
from multipartite_generation import MultipartiteSampler


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

    # Training loop
    for ii in tqdm(range(steps), desc="SAEs (all sites)"):
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

    # Return grouped by site, mirroring prior structures
    sequence_saes_all = {site: {"top_k": seq_topk_all[site], "vanilla": seq_vanilla_all[site]} for site in site_to_hook.keys()}
    return sequence_saes_all, true_coord_saes_all, metrics_summary_all


# Don't need nearly as many training steps for the ground truth SAEs but just hacking it so we
# get them all together and it's short anyway.
def train_sae_on_site(
    site_name: str,
    hook_name: str,
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
    """
    Train sequence SAEs (TopK for k in {1,2,3,6}) on model activations at hook_name,
    plus Vanilla SAEs on the same activations with an L1 lambda sweep (lambda_values_seq), and
    Vanilla SAEs on belief-state vectors with configurable dimensionality/dict size
    for a sweep of L1 lambdas (lambda_values_beliefs).

    Returns:
      sequence_saes: {"top_k": {"k1": SAE, "k2": SAE, ...}, "vanilla": {...}}
      true_coords_saes: {"lambda_0.001": SAE, ...}
    """
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
            "train_sae_on_site requires at least one of k_values, lambda_values_seq, or "
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

    # Build SAEs
    seq_topk: Dict[str, TopKSAE] = {}
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
            seq_topk[f"k{k}"] = TopKSAE(cfg_topk)

    seq_vanilla: Dict[str, VanillaSAE] = {}
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
            seq_vanilla[f"lambda_{lam}"] = VanillaSAE(cfg_s_v)

    true_coord_saes: Dict[str, VanillaSAE] = {}
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
            true_coord_saes[f"lambda_{lam}"] = VanillaSAE(cfg_v)

    # Create optimizers
    opt_seq_topk = {k: torch.optim.Adam(s.parameters(), lr=3e-4, betas=(0.9, 0.99)) for k, s in seq_topk.items()}
    opt_true = {k: torch.optim.Adam(s.parameters(), lr=3e-4, betas=(0.9, 0.99)) for k, s in true_coord_saes.items()}
    opt_seq_vanilla = {k: torch.optim.Adam(s.parameters(), lr=3e-4, betas=(0.9, 0.99)) for k, s in seq_vanilla.items()}

    # Metrics containers (collect lists internally; return summaries only)
    metrics_raw: Dict[str, Dict] = {
        "sequence": {
            "top_k": {name: {"loss": [], "l2_loss": [], "l1_loss": [], "l0_norm": [], "l1_norm": [], "aux_loss": [], "num_dead_features": [], "active_latents": [], "active_counts": [], "active_sums": []} for name in seq_topk.keys()},
            "vanilla": {name: {"loss": [], "l2_loss": [], "l1_loss": [], "l0_norm": [], "l1_norm": [], "active_latents": [], "active_counts": [], "active_sums": []} for name in seq_vanilla.keys()},
        },
        "beliefs": {name: {"loss": [], "l2_loss": [], "l1_loss": [], "l0_norm": [], "l1_norm": [], "active_latents": [], "active_counts": [], "active_sums": []} for name in true_coord_saes.keys()},
    }

    # Joint training over the same generated batches
    for ii in tqdm(range(steps), desc=f"SAEs {site_name}"):
        rng_key, states, observations = _generate_sequences(
            rng_key,
            batch_size=batch_size,
            sequence_len=seq_len,
            source=data_source,
        )
        tokens = _tokens_from_observations(observations, device=device)

        # sequence activations for this site
        acts_seq = _acts_batch(hook_name, tokens, model)

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

        # Step sequence TopK SAEs
        for name, sae in seq_topk.items():
            sae.train()
            out = sae(acts_seq)
            loss = out["loss"]
            opt_seq_topk[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1e5)
            sae.make_decoder_weights_and_grad_unit_norm()
            opt_seq_topk[name].step()
            # record metrics
            metrics_raw["sequence"]["top_k"][name]["loss"].append(float(loss.detach().item()))
            for key in ("l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss"):
                if key in out:
                    metrics_raw["sequence"]["top_k"][name][key].append(float(out[key].detach().item()))
            if "num_dead_features" in out:
                v = out["num_dead_features"]
                v = v.item() if hasattr(v, "item") else int(v)
                metrics_raw["sequence"]["top_k"][name]["num_dead_features"].append(int(v))
            # track active latent mask for this batch
            with torch.no_grad():
                acts = out.get("feature_acts")
                if acts is not None:
                    mask = (acts > 0).any(dim=0).detach().cpu().numpy().astype(bool)
                    metrics_raw["sequence"]["top_k"][name]["active_latents"].append(mask)
                    counts = (acts > 0).sum(dim=0).detach().cpu().numpy()
                    sums = acts.clamp(min=0).sum(dim=0).detach().cpu().numpy()
                    metrics_raw["sequence"]["top_k"][name]["active_counts"].append(counts)
                    metrics_raw["sequence"]["top_k"][name]["active_sums"].append(sums)


        # Step sequence Vanilla SAEs (activations)
        for name, sae in seq_vanilla.items():
            sae.train()
            out = sae(acts_seq)
            loss = out["loss"]
            opt_seq_vanilla[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1e5)
            sae.make_decoder_weights_and_grad_unit_norm()
            opt_seq_vanilla[name].step()
            # record metrics
            metrics_raw["sequence"]["vanilla"][name]["loss"].append(float(loss.detach().item()))
            for key in ("l2_loss", "l1_loss", "l0_norm", "l1_norm"):
                if key in out:
                    metrics_raw["sequence"]["vanilla"][name][key].append(float(out[key].detach().item()))
            # track active latent mask for this batch
            with torch.no_grad():
                acts = out.get("feature_acts")
                if acts is not None:
                    mask = (acts > 0).any(dim=0).detach().cpu().numpy().astype(bool)
                    metrics_raw["sequence"]["vanilla"][name]["active_latents"].append(mask)
                    counts = (acts > 0).sum(dim=0).detach().cpu().numpy()
                    sums = acts.clamp(min=0).sum(dim=0).detach().cpu().numpy()
                    metrics_raw["sequence"]["vanilla"][name]["active_counts"].append(counts)
                    metrics_raw["sequence"]["vanilla"][name]["active_sums"].append(sums)

        # Step true-coordinates Vanilla SAEs (3 -> 8 -> 3)
        for name, sae in true_coord_saes.items():
            if beliefs_tensor is None:
                raise RuntimeError("Belief SAEs requested but belief tensor was not computed.")
            sae.train()
            out = sae(beliefs_tensor)
            loss = out["loss"]
            opt_true[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1e5)
            sae.make_decoder_weights_and_grad_unit_norm()
            opt_true[name].step()
            # record metrics
            metrics_raw["beliefs"][name]["loss"].append(float(loss.detach().item()))
            for key in ("l2_loss", "l1_loss", "l0_norm", "l1_norm"):
                if key in out:
                    metrics_raw["beliefs"][name][key].append(float(out[key].detach().item()))
            # track active latent mask for this batch
            with torch.no_grad():
                acts = out.get("feature_acts")
                if acts is not None:
                    mask = (acts > 0).any(dim=0).detach().cpu().numpy().astype(bool)
                    metrics_raw["beliefs"][name]["active_latents"].append(mask)
                    counts = (acts > 0).sum(dim=0).detach().cpu().numpy()
                    sums = acts.clamp(min=0).sum(dim=0).detach().cpu().numpy()
                    metrics_raw["beliefs"][name]["active_counts"].append(counts)
                    metrics_raw["beliefs"][name]["active_sums"].append(sums)
        
        # if ii % 100 == 0:
        #    check_cuda_memory()

    # Summarize metrics: averages over last quarter of steps, and loss trajectory over last 50 steps
    def summarize_series(series: list, steps: int):
        if not series:
            return None, []
        start = max(0, int(steps * 3 / 4))
        last_quarter = series[start:]
        avg_last_quarter = float(np.mean(last_quarter)) if len(last_quarter) > 0 else None
        last50 = series[-50:] if len(series) >= 50 else series[:]
        return avg_last_quarter, last50

    def summarize_active_indices(active_latents: list, steps: int):
        if not active_latents:
            return []
        start = max(0, int(steps * 3 / 4))
        window = active_latents[start:]
        try:
            stacked = np.stack(window, axis=0)
            union_mask = stacked.any(axis=0)
            inds = np.where(union_mask)[0].astype(int).tolist()
            return inds
        except Exception:
            return []

    def summarize_active_stats(active_counts: list, active_sums: list, steps: int):
        """Return dict latent_index -> (count_nonzero_samples, sum_magnitudes) over last quarter."""
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

    # Save all
    os.makedirs(sae_output_dir, exist_ok=True)
    for name, sae in seq_topk.items():
        torch.save({"state_dict": sae.state_dict(), "cfg": sae.cfg}, os.path.join(sae_output_dir, f"{site_name}_top_k_{name}.pt"))
    for name, sae in seq_vanilla.items():
        torch.save({"state_dict": sae.state_dict(), "cfg": sae.cfg}, os.path.join(sae_output_dir, f"{site_name}_vanilla_{name}.pt"))
    for name, sae in true_coord_saes.items():
        torch.save({"state_dict": sae.state_dict(), "cfg": sae.cfg}, os.path.join(sae_output_dir, f"{site_name}_beliefs_{name}.pt"))

    sequence_saes = {
        "top_k": seq_topk,
        "vanilla": seq_vanilla,
    }
    return sequence_saes, true_coord_saes, metrics_summary
