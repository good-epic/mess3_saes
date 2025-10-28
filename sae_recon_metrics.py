#!/usr/bin/env python3


#%%

"""
Estimate reconstruction quality metrics for sparse autoencoders (SAEs).

Given a trained transformer checkpoint and a directory of SAE checkpoints,
this script samples sequences from the configured generative process and
computes, per selected site:

* Fraction of variance explained (FVE)
* Normalized MSE (relative to mean activation norm)
* Mean cosine similarity between original and reconstructed activations
* Downstream cross-entropy delta when the model runs with SAE reconstructions
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import jax
import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

from BatchTopK.sae import BatchTopKSAE, TopKSAE, VanillaSAE
from mess3_gmg_analysis_utils import sae_decode_features, sae_encode_features
from multipartite_utils import (
    MultipartiteSampler,
    _load_process_stack,
    _load_transformer,
    _resolve_device,
)
from training_and_analysis_utils import _generate_sequences, _tokens_from_observations, load_metrics_summary

# Mapping from human-readable site labels to HookedTransformer cache names
SITE_HOOK_MAP: Dict[str, str] = {
    "embeddings": "hook_embed",
    "layer_0": "blocks.0.hook_resid_post",
    "layer_1": "blocks.1.hook_resid_post",
    "layer_2": "blocks.2.hook_resid_post",
    "layer_3": "blocks.3.hook_resid_post",
}

# Allow script defaults to mirror other utilities
PRESET_PROCESS_CONFIGS = {
    "single_mess3": [
        {"type": "mess3", "params": {"x": 0.1, "a": 0.7}},
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

@dataclass
class Aggregator:
    """Running totals for reconstruction metrics."""

    total_samples: int = 0
    total_tokens: int = 0
    sum_sq_error: float = 0.0
    sum_x2: float = 0.0
    sum_norm_sq: float = 0.0
    sum_cosine: float = 0.0
    cosine_count: int = 0
    act_dim: Optional[int] = None
    sum_per_dim: Optional[torch.Tensor] = None  # shape (act_dim,)
    perturbed_loss_sum: float = 0.0


@dataclass(frozen=True)
class SAEEntry:
    site: str
    sae_type: str  # top_k or vanilla
    label: str  # e.g. "k10" or "lambda_0.01"
    checkpoint_path: str
    k: Optional[int] = None
    lambda_value: Optional[float] = None

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.site, self.sae_type, self.label)


def iter_progress(iterable, **kwargs):
    """Return iterable wrapped in tqdm when available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SAE reconstruction and downstream metrics")

    # Required inputs
    parser.add_argument("--sae_folder", type=str, default="outputs/saes/multipartite_003e", help="Directory containing SAE checkpoints")
    parser.add_argument("--model_ckpt", type=str, default="outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt", help="Path to transformer checkpoint (.pt)")

    # Process configuration
    parser.add_argument("--process_config", type=str, default="process_configs.json",
                        help="Path to generative process configuration JSON")
    parser.add_argument("--process_config_name", type=str, default="3xmess3_2xtquant_003",
                        help="Configuration key within --process_config (if mapping)")
    parser.add_argument("--process_preset", type=str, default=None,
                        help="Use named preset configuration instead of explicit file")

    # Transformer architecture fallbacks (used if checkpoint lacks config)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--d_vocab", type=int, default=None)
    parser.add_argument("--act_fn", type=str, default="relu")

    # Evaluation controls
    parser.add_argument("--batch_size", type=int, default=2048, help="Samples per batch")
    parser.add_argument("--n_batches", type=int, default=150, help="Number of batches to evaluate")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length to sample")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for sampling")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")

    # SAE selection
    parser.add_argument("--sites", nargs="+", default=None,
                        help="Subset of sites to evaluate (default: all available)")
    parser.add_argument(
        "--sae_types",
        nargs="+",
        choices=["top_k", "vanilla"],
        default=("top_k", "vanilla"),
        help="SAE families to evaluate (default: both top_k and vanilla)",
    )
    parser.add_argument("--sae_override", action="append", default=None,
                        help="Override selection per site using SITE[:TYPE]=NAME "
                             "(e.g. layer_0:vanilla=lambda_0.01)")
    parser.add_argument("--metrics_summary", type=str, default=None,
                        help="Path to metrics_summary.json (defaults to <sae_folder>/metrics_summary.json)")

    # Output
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional path to save metrics JSON report")

    args, _ = parser.parse_known_args(argv)
    return args


def parse_overrides(entries: Optional[Iterable[str]]) -> Dict[str, list[Tuple[Optional[str], str]]]:
    """Parse SITE[:TYPE]=NAME override strings."""
    overrides: Dict[str, list[Tuple[Optional[str], str]]] = {}
    if not entries:
        return overrides
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}' (expected SITE[:TYPE]=NAME)")
        left, name = raw.split("=", 1)
        if ":" in left:
            site, site_type = left.split(":", 1)
            site_type = site_type.strip()
        else:
            site, site_type = left, None
        site = site.strip()
        if not site or not name:
            raise ValueError(f"Invalid override '{raw}'")
        overrides.setdefault(site, []).append((site_type, name.strip()))
    return overrides


def discover_saes(
    folder: str,
    *,
    allowed_sites: Optional[Iterable[str]],
    allowed_types: Iterable[str],
    overrides: Dict[str, list[Tuple[Optional[str], str]]],
) -> list[SAEEntry]:
    """Return all SAE checkpoints in ``folder`` matching the filters."""

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"SAE folder '{folder}' does not exist")

    site_filter = set(allowed_sites) if allowed_sites else None
    type_filter = set(allowed_types)
    entries: list[SAEEntry] = []

    topk_pattern = re.compile(r"^(?P<site>.+)_top_k_k(?P<k>\d+)\.pt$")
    vanilla_pattern = re.compile(r"^(?P<site>.+)_vanilla_lambda_(?P<lam>[0-9eE.+-]+)\.pt$")

    for path in glob.glob(os.path.join(folder, "*.pt")):
        base = os.path.basename(path)

        match_topk = topk_pattern.match(base)
        if match_topk:
            site = match_topk.group("site")
            if site not in SITE_HOOK_MAP:
                continue
            if site_filter and site not in site_filter:
                continue
            if "top_k" not in type_filter:
                continue
            k_val = int(match_topk.group("k"))
            label = f"k{k_val}"
            if overrides and site in overrides:
                if not any((typ is None or typ == "top_k") and entry_label == label for typ, entry_label in overrides[site]):
                    continue
            entries.append(
                SAEEntry(
                    site=site,
                    sae_type="top_k",
                    label=label,
                    checkpoint_path=path,
                    k=k_val,
                )
            )
            continue

        match_vanilla = vanilla_pattern.match(base)
        if match_vanilla:
            site = match_vanilla.group("site")
            if site not in SITE_HOOK_MAP:
                continue
            if site_filter and site not in site_filter:
                continue
            if "vanilla" not in type_filter:
                continue
            lam_str = match_vanilla.group("lam")
            try:
                lam_value = float(lam_str)
            except ValueError:
                lam_value = None
            label = f"lambda_{lam_str}"
            if overrides and site in overrides:
                if not any((typ is None or typ == "vanilla") and entry_label == label for typ, entry_label in overrides[site]):
                    continue
            entries.append(
                SAEEntry(
                    site=site,
                    sae_type="vanilla",
                    label=label,
                    checkpoint_path=path,
                    lambda_value=lam_value,
                )
            )
            continue

    if not entries:
        raise RuntimeError("No SAE checkpoints found matching the requested filters.")

    def sort_key(entry: SAEEntry) -> Tuple[str, int, float]:
        if entry.sae_type == "top_k":
            key_val = entry.k if entry.k is not None else -1
            return (entry.site, 0, float(key_val))
        else:
            val = entry.lambda_value if entry.lambda_value is not None else math.inf
            return (entry.site, 1, float(val))

    entries.sort(key=sort_key)
    return entries


def instantiate_sae(entry: SAEEntry, device: str):
    ckpt = torch.load(entry.checkpoint_path, map_location=device)
    if "cfg" not in ckpt or "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {entry.checkpoint_path} missing cfg/state_dict keys")
    cfg = dict(ckpt["cfg"])
    cfg["device"] = device
    if entry.sae_type == "vanilla":
        sae = VanillaSAE(cfg)
    elif entry.sae_type == "top_k":
        sae = TopKSAE(cfg)
    else:
        # Fallback for unexpected types (e.g., batch_top_k)
        sae = BatchTopKSAE(cfg)
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    for param in sae.parameters():
        param.requires_grad_(False)
    return sae


def compute_cross_entropy(logits: torch.Tensor, tokens: torch.Tensor) -> Tuple[float, int]:
    """
    Return total cross-entropy (sum over tokens) and token count.
    """
    vocab_size = logits.shape[-1]
    logits_shift = logits[:, :-1, :].reshape(-1, vocab_size)
    targets = tokens[:, 1:].reshape(-1)
    loss = F.cross_entropy(logits_shift, targets, reduction="sum")
    n_tokens = targets.numel()
    return float(loss.item()), n_tokens


def reconstruct_activations(sae, activations: torch.Tensor) -> torch.Tensor:
    """Encode and decode activations without mutating SAE book-keeping."""
    with torch.no_grad():
        features, x_mean, x_std = sae_encode_features(sae, activations)
        recon = sae_decode_features(sae, features, x_mean, x_std)
    return recon


def run_evaluation(
    model,
    data_source,
    entries: list[SAEEntry],
    device: str,
    batch_size: int,
    n_batches: int,
    seq_len: int,
    seed: int,
) -> Tuple[Dict[Tuple[str, str, str], Aggregator], float, int]:
    """Main evaluation loop across all SAE checkpoints."""

    if not entries:
        raise RuntimeError("run_evaluation called with zero SAE entries")

    entries_by_site: dict[str, list[SAEEntry]] = defaultdict(list)
    for entry in entries:
        entries_by_site[entry.site].append(entry)

    aggregators: Dict[Tuple[str, str, str], Aggregator] = {entry.key: Aggregator() for entry in entries}
    saes = {entry.key: instantiate_sae(entry, device) for entry in entries}
    site_hook_map = {site: SITE_HOOK_MAP[site] for site in entries_by_site}
    hook_names = list(site_hook_map.values())

    rng_key = jax.random.PRNGKey(seed)
    baseline_loss_sum = 0.0
    baseline_token_count = 0

    for batch_idx in iter_progress(range(n_batches), total=n_batches, desc="Evaluating batches", leave=False):
        rng_key, states, observations = _generate_sequences(
            rng_key,
            batch_size=batch_size,
            sequence_len=seq_len,
            source=data_source,
        )
        tokens = _tokens_from_observations(observations, device=device)
        if tokens.shape[1] < 2:
            continue  # need at least two tokens for loss comparison

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                return_type="logits",
                names_filter=hook_names,
            )
        batch_loss_sum, batch_tokens = compute_cross_entropy(logits, tokens)
        baseline_loss_sum += batch_loss_sum
        baseline_token_count += batch_tokens

        site_acts: dict[str, torch.Tensor] = {}
        site_acts_flat: dict[str, torch.Tensor] = {}
        for site, hook_name in site_hook_map.items():
            if hook_name not in cache:
                raise KeyError(f"Hook '{hook_name}' not present in transformer cache")
            acts = cache[hook_name].detach()
            site_acts[site] = acts
            site_acts_flat[site] = acts.reshape(-1, acts.shape[-1])

        for site, site_entries in entries_by_site.items():
            acts = site_acts[site]
            acts_flat = site_acts_flat[site]

            for entry in site_entries:
                sae = saes[entry.key]
                state = aggregators[entry.key]

                recon_flat = reconstruct_activations(sae, acts_flat)
                error = recon_flat - acts_flat

                sample_count = acts_flat.shape[0]
                state.total_samples += sample_count

                if state.act_dim is None:
                    state.act_dim = acts_flat.shape[-1]
                    state.sum_per_dim = torch.zeros(state.act_dim, dtype=torch.float64)

                state.sum_sq_error += float(error.pow(2).sum().item())
                state.sum_x2 += float(acts_flat.pow(2).sum().item())
                state.sum_norm_sq += float(acts_flat.pow(2).sum(dim=-1).sum().item())
                state.sum_per_dim += acts_flat.sum(dim=0).detach().cpu().double()

                dot = (acts_flat * recon_flat).sum(dim=-1)
                act_norm = acts_flat.norm(dim=-1)
                recon_norm = recon_flat.norm(dim=-1)
                denom = act_norm * recon_norm + 1e-8
                cosine = dot / denom
                state.sum_cosine += float(cosine.sum().item())
                state.cosine_count += int(cosine.numel())

                recon_tensor = recon_flat.reshape_as(acts).detach()

                def hook_fn(value, hook=None, tensor=recon_tensor):
                    return tensor

                with torch.no_grad():
                    sae_logits = model.run_with_hooks(
                        tokens,
                        return_type="logits",
                        fwd_hooks=[(site_hook_map[site], hook_fn)],
                    )
                perturbed_loss_sum, perturbed_tokens = compute_cross_entropy(sae_logits, tokens)
                state.perturbed_loss_sum += perturbed_loss_sum
                state.total_tokens += perturbed_tokens

        # Free cache tensors explicitly
        del cache
        del logits

    return aggregators, baseline_loss_sum, baseline_token_count


def finalize_metrics(
    entries: list[SAEEntry],
    aggregators: Dict[Tuple[str, str, str], Aggregator],
    baseline_loss_sum: float,
    baseline_token_count: int,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    baseline_loss = baseline_loss_sum / max(baseline_token_count, 1)
    for entry in entries:
        state = aggregators[entry.key]
        if state.total_samples == 0:
            raise RuntimeError(f"No samples collected for site '{entry.site}' ({entry.label})")
        act_dim = state.act_dim or 0
        elements = state.total_samples * act_dim if act_dim else 1
        recon_mse = state.sum_sq_error / elements
        mean_norm_sq = state.sum_norm_sq / max(state.total_samples, 1)

        if state.sum_per_dim is not None:
            sum_per_dim = state.sum_per_dim.cpu().numpy()
        else:
            sum_per_dim = np.zeros(act_dim, dtype=np.float64)
        mu_sq_sum = float(np.square(sum_per_dim / max(state.total_samples, 1)).sum())
        variance = (state.sum_x2 - state.total_samples * mu_sq_sum) / max(elements, 1)
        variance = max(variance, 0.0)

        fve = None
        if variance > 0:
            fve = 1.0 - recon_mse / variance

        norm_mse = None
        if mean_norm_sq > 0:
            norm_mse = recon_mse / mean_norm_sq

        cosine = None
        if state.cosine_count > 0:
            cosine = state.sum_cosine / state.cosine_count

        sae_loss = state.perturbed_loss_sum / max(state.total_tokens, 1)
        delta_loss = sae_loss - baseline_loss
        percent_change = None
        if baseline_loss > 0:
            percent_change = 100.0 * delta_loss / baseline_loss

        lambda_display = None
        if entry.sae_type == "vanilla":
            if entry.label.startswith("lambda_"):
                lambda_display = entry.label.split("lambda_", 1)[1]
            elif entry.lambda_value is not None:
                lambda_display = f"{entry.lambda_value:g}"

        rows.append(
            {
                "site": entry.site,
                "sae_type": entry.sae_type,
                "label": entry.label,
                "k": entry.k,
                "lambda": entry.lambda_value,
                "lambda_display": lambda_display,
                "samples": float(state.total_samples),
                "activation_dim": float(act_dim),
                "reconstruction_mse": float(recon_mse),
                "original_variance": float(variance),
                "fraction_variance_explained": None if fve is None else float(fve),
                "normalized_mse": None if norm_mse is None else float(norm_mse),
                "mean_cosine_similarity": None if cosine is None else float(cosine),
                "baseline_cross_entropy": float(baseline_loss),
                "sae_cross_entropy": float(sae_loss),
                "cross_entropy_delta": float(delta_loss),
                "cross_entropy_delta_pct": None if percent_change is None else float(percent_change),
            }
        )
    return rows


def print_summary(rows: list[Dict[str, Any]]) -> None:
    print("=" * 72)
    print("SAE Reconstruction Metrics")
    print("=" * 72)

    if not rows:
        print("No metrics to display.")
        return

    columns = [
        ("site", "Site"),
        ("sae_type", "SAE Type"),
        ("k", "K"),
        ("lambda_display", "Lambda"),
        ("samples", "Samples"),
        ("activation_dim", "Activation Dim"),
        ("reconstruction_mse", "Recon MSE"),
        ("original_variance", "Original Var"),
        ("fraction_variance_explained", "FVE"),
        ("normalized_mse", "Normalized MSE"),
        ("mean_cosine_similarity", "Cosine"),
        ("baseline_cross_entropy", "Baseline CE"),
        ("sae_cross_entropy", "SAE CE"),
        ("cross_entropy_delta", "Δ CE"),
        ("cross_entropy_delta_pct", "Δ CE %"),
    ]

    def fmt_float(value: Optional[float], precision: int = 6, signed: bool = False) -> str:
        if value is None or not math.isfinite(value):
            return ""
        fmt = f"{{:{'+' if signed else ''}.{precision}f}}"
        return fmt.format(value)

    formatted_rows: list[list[str]] = []
    widths = [len(header) for _, header in columns]

    for row in rows:
        formatted_row: list[str] = []
        for idx, (field, _) in enumerate(columns):
            value = row.get(field)
            if field in {"samples", "activation_dim"}:
                cell = "" if value is None else f"{int(value):d}"
            elif field == "k":
                cell = "" if value is None else str(int(value))
            elif field == "lambda_display":
                if value is not None:
                    cell = value
                else:
                    lambda_val = row.get("lambda")
                    cell = "" if lambda_val is None else f"{lambda_val:g}"
            elif field == "fraction_variance_explained":
                cell = fmt_float(value, precision=6)
            elif field == "normalized_mse":
                cell = fmt_float(value, precision=6)
            elif field == "mean_cosine_similarity":
                cell = fmt_float(value, precision=6)
            elif field in {"reconstruction_mse", "original_variance", "baseline_cross_entropy", "sae_cross_entropy"}:
                cell = fmt_float(value, precision=6)
            elif field == "cross_entropy_delta":
                cell = fmt_float(value, precision=6, signed=True)
            elif field == "cross_entropy_delta_pct":
                cell = fmt_float(value, precision=2, signed=True)
            else:
                cell = "" if value is None else str(value)
            widths[idx] = max(widths[idx], len(cell))
            formatted_row.append(cell)
        formatted_rows.append(formatted_row)

    header_line = " | ".join(header.ljust(widths[idx]) for idx, (_, header) in enumerate(columns))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(columns)))
    print(header_line)
    print(separator)
    for formatted_row in formatted_rows:
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(formatted_row)))
    print()


def save_results(output_path: str, args: argparse.Namespace, rows: list[Dict[str, Any]]) -> None:
    payload = {
        "config": {
            "model_ckpt": args.model_ckpt,
            "sae_folder": args.sae_folder,
            "process_config": args.process_config,
            "process_config_name": args.process_config_name,
            "process_preset": args.process_preset,
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "sae_types": list(args.sae_types),
        },
        "metrics": rows,
        "metrics_by_site": {
            site: [row for row in rows if row["site"] == site] for site in sorted({row["site"] for row in rows})
        },
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics report to {output_path}")


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    metrics_path = args.metrics_summary or os.path.join(args.sae_folder, "metrics_summary.json")
    metrics_summary = load_metrics_summary(metrics_path)
    if metrics_summary is None:
        print(f"Warning: metrics summary not available at {metrics_path}")

    overrides = parse_overrides(args.sae_override)
    allowed_sites = None
    if args.sites:
        allowed_sites = [site for site in args.sites if site in SITE_HOOK_MAP]
        missing = sorted(set(args.sites) - set(allowed_sites))
        if missing:
            print(f"Warning: ignoring unknown sites {missing}")
        if not allowed_sites:
            raise RuntimeError("No valid sites selected after filtering; aborting.")

    entries = discover_saes(
        args.sae_folder,
        allowed_sites=allowed_sites,
        allowed_types=args.sae_types,
        overrides=overrides,
    )
    print("Evaluating SAEs:")
    for entry in entries:
        basename = os.path.basename(entry.checkpoint_path)
        descriptor = entry.label if entry.label else entry.sae_type
        print(f"  • {entry.site}: {entry.sae_type} ({descriptor}) from {basename}")

    process_cfg_raw, components, data_source = _load_process_stack(args, PRESET_PROCESS_CONFIGS)
    if isinstance(data_source, MultipartiteSampler):
        vocab_size = data_source.vocab_size
    else:
        vocab_size = data_source.vocab_size

    model, cfg = _load_transformer(args, device, vocab_size)
    model.eval()

    aggregators, baseline_loss_sum, baseline_token_count = run_evaluation(
        model=model,
        data_source=data_source,
        entries=entries,
        device=device,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    results_rows = finalize_metrics(entries, aggregators, baseline_loss_sum, baseline_token_count)
    print_summary(results_rows)

    output_path = args.output_path or os.path.join(args.sae_folder, "reconstruction_metrics.json")
    save_results(output_path, args, results_rows)


#%%
