#!/usr/bin/env python3
"""
Evaluate transformer predictive accuracy on multipartite generative processes.

This module can be invoked as a script (see parse_args/main) or imported and
called programmatically via `evaluate_model_accuracy_from_args` /
`evaluate_model_accuracy`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

os.environ["JAX_PLATFORM_NAME"] = "cpu"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from multipartite_utils import (
    MultipartiteSampler,
    _load_process_stack,
    _load_transformer,
    _resolve_device,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model predictive accuracy")

    # Model and data
    parser.add_argument("--model_ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--process_config", type=str, default="process_configs.json",
                        help="Path to process config JSON")
    parser.add_argument("--process_config_name", type=str, default=None,
                        help="Named configuration in process config")
    parser.add_argument("--process_preset", type=str, default=None,
                        help="Optional preset name (overrides JSON)")

    # Model architecture overrides
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--d_vocab", type=int, default=None)
    parser.add_argument("--act_fn", type=str, default="relu")

    # Evaluation config
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/model_eval",
                        help="Directory to save metrics/results")

    return parser.parse_args()


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compute per-batch predictive metrics."""
    batch_size, seq_len = targets.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    predictions = logits_flat.argmax(dim=-1)
    correct = (predictions == targets_flat).float()

    k_top = min(10, vocab_size)
    top_k_preds = logits_flat.topk(k=k_top, dim=-1).indices
    targets_expanded = targets_flat.unsqueeze(1)
    top_1_correct = (top_k_preds[:, :1] == targets_expanded).any(dim=1).float()
    top_5_correct = (top_k_preds[:, : min(5, k_top)] == targets_expanded).any(dim=1).float()
    top_10_correct = (top_k_preds == targets_expanded).any(dim=1).float()

    metrics = {
        "accuracy": float(correct.mean().item()),
        "top_5_accuracy": float(top_5_correct.mean().item()),
        "top_10_accuracy": float(top_10_correct.mean().item()),
        "cross_entropy": float(ce_loss.mean().item()),
        "perplexity": float(torch.exp(ce_loss.mean()).item()),
        "n_tokens": int(len(targets_flat)),
    }

    per_token_stats = {
        "targets": targets_flat.cpu().numpy(),
        "correct": correct.cpu().numpy(),
        "ce_loss": ce_loss.cpu().numpy(),
    }
    return metrics, per_token_stats


def aggregate_per_token_metrics(
    all_targets: np.ndarray,
    all_correct: np.ndarray,
    all_ce_loss: np.ndarray,
    vocab_size: int,
) -> Dict[int, Dict[str, float]]:
    """Aggregate per-token metrics."""
    per_token_metrics: Dict[int, Dict[str, float]] = {}
    n_total = len(all_targets)

    for token_id in range(vocab_size):
        mask = all_targets == token_id
        count = int(mask.sum())
        if count == 0:
            per_token_metrics[token_id] = {
                "count": 0,
                "accuracy": 0.0,
                "cross_entropy": 0.0,
                "frequency": 0.0,
            }
            continue

        per_token_metrics[token_id] = {
            "count": count,
            "accuracy": float(all_correct[mask].mean()),
            "cross_entropy": float(all_ce_loss[mask].mean()),
            "frequency": float(count / n_total),
        }

    return per_token_metrics


def plot_per_token_histograms(
    per_token_metrics: Dict[int, Dict[str, float]],
    output_dir: str,
) -> None:
    """Save histograms summarizing per-token metrics."""
    os.makedirs(output_dir, exist_ok=True)

    tokens_with_data = [tid for tid, m in per_token_metrics.items() if m["count"] > 0]
    if not tokens_with_data:
        print("No per-token statistics to plot.")
        return

    accuracies = [per_token_metrics[tid]["accuracy"] for tid in tokens_with_data]
    cross_entropies = [per_token_metrics[tid]["cross_entropy"] for tid in tokens_with_data]
    frequencies = [per_token_metrics[tid]["frequency"] for tid in tokens_with_data]
    counts = [per_token_metrics[tid]["count"] for tid in tokens_with_data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(accuracies, bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Per-token Accuracy")
    axes[0, 0].axvline(np.mean(accuracies), color="red", linestyle="--",
                       label=f"Mean={np.mean(accuracies):.3f}")
    axes[0, 0].legend()

    axes[0, 1].hist(cross_entropies, bins=30, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Per-token Cross-Entropy")
    axes[0, 1].axvline(np.mean(cross_entropies), color="red", linestyle="--",
                       label=f"Mean={np.mean(cross_entropies):.3f}")
    axes[0, 1].legend()

    axes[1, 0].hist(frequencies, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Token Frequency")
    axes[1, 0].set_yscale("log")

    axes[1, 1].hist(counts, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 1].set_title("Token Counts")
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_token_histograms.png"), dpi=150)
    plt.close()


def load_sampler(args: argparse.Namespace) -> Tuple[Any, int, Dict[str, Any]]:
    process_cfg_raw, components, data_source = _load_process_stack(args, PRESET_PROCESS_CONFIGS)
    if not hasattr(data_source, "sample"):
        # Wrap single component into a MultipartiteSampler for consistent API
        data_source = MultipartiteSampler(components)
    if not hasattr(data_source, "vocab_size"):
        raise ValueError("Sampler must expose `vocab_size` attribute.")
    return data_source, data_source.vocab_size, process_cfg_raw


def load_model(args: argparse.Namespace, device: torch.device, vocab_size: int) -> torch.nn.Module:
    model, cfg = _load_transformer(args, device, vocab_size)
    checkpoint = torch.load(args.model_ckpt, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_evaluation_loop(
    model: torch.nn.Module,
    sampler: Any,
    vocab_size: int,
    batch_size: int,
    n_batches: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    total_tokens = 0
    total_correct = 0.0
    total_top5 = 0.0
    total_top10 = 0.0
    total_ce = 0.0

    all_targets = []
    all_correct = []
    all_ce_loss = []

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Evaluating"):
            key, belief_states, product_tokens, component_observations = sampler.sample(
                key, batch_size, seq_len
            )
            tokens = torch.from_numpy(np.array(product_tokens)).long().to(device)
            logits = model(tokens)

            logits_for_pred = logits[:, :-1, :]
            targets = tokens[:, 1:]

            batch_metrics, per_token_stats = compute_batch_metrics(logits_for_pred, targets, vocab_size)

            n_tokens = batch_metrics["n_tokens"]
            total_tokens += n_tokens
            total_correct += batch_metrics["accuracy"] * n_tokens
            total_top5 += batch_metrics["top_5_accuracy"] * n_tokens
            total_top10 += batch_metrics["top_10_accuracy"] * n_tokens
            total_ce += batch_metrics["cross_entropy"] * n_tokens

            all_targets.append(per_token_stats["targets"])
            all_correct.append(per_token_stats["correct"])
            all_ce_loss.append(per_token_stats["ce_loss"])

    all_targets = np.concatenate(all_targets)
    all_correct = np.concatenate(all_correct)
    all_ce_loss = np.concatenate(all_ce_loss)

    overall_metrics = {
        "accuracy": float(total_correct / total_tokens),
        "top_5_accuracy": float(total_top5 / total_tokens),
        "top_10_accuracy": float(total_top10 / total_tokens),
        "cross_entropy": float(total_ce / total_tokens),
        "perplexity": float(np.exp(total_ce / total_tokens)),
        "total_tokens": int(total_tokens),
    }

    per_token_metrics = aggregate_per_token_metrics(
        all_targets, all_correct, all_ce_loss, vocab_size
    )
    return overall_metrics, per_token_metrics


def summarize_results(
    overall_metrics: Dict[str, float],
    per_token_metrics: Dict[int, Dict[str, float]],
    vocab_size: int,
) -> Dict[str, Any]:
    tokens_with_data = [tid for tid, m in per_token_metrics.items() if m["count"] > 0]
    if tokens_with_data:
        accuracies = np.array([per_token_metrics[tid]["accuracy"] for tid in tokens_with_data])
        cross_entropies = np.array([per_token_metrics[tid]["cross_entropy"] for tid in tokens_with_data])
        per_token_summary = {
            "n_tokens_observed": int(len(tokens_with_data)),
            "accuracy_mean": float(accuracies.mean()),
            "accuracy_std": float(accuracies.std()),
            "accuracy_min": float(accuracies.min()),
            "accuracy_max": float(accuracies.max()),
            "cross_entropy_mean": float(cross_entropies.mean()),
            "cross_entropy_std": float(cross_entropies.std()),
            "cross_entropy_min": float(cross_entropies.min()),
            "cross_entropy_max": float(cross_entropies.max()),
        }
    else:
        per_token_summary = {"n_tokens_observed": 0}

    uniform_ce = float(np.log(vocab_size))
    empirical_ce = 0.0
    for tid in tokens_with_data:
        freq = per_token_metrics[tid]["frequency"]
        if freq > 0:
            empirical_ce += freq * np.log(freq)
    empirical_ce = float(-empirical_ce)

    baselines = {
        "uniform_cross_entropy": uniform_ce,
        "empirical_cross_entropy": empirical_ce,
        "ce_reduction_vs_uniform": float(uniform_ce - overall_metrics["cross_entropy"]),
        "ce_reduction_vs_empirical": float(empirical_ce - overall_metrics["cross_entropy"]),
    }
    return {"per_token_summary": per_token_summary, "baselines": baselines}


def save_results(
    output_dir: str,
    results: Dict[str, Any],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    plot_per_token_histograms(results["per_token_metrics"], output_dir)


def evaluate_model_accuracy_from_args(
    args: argparse.Namespace,
    *,
    save_outputs: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    device = _resolve_device(args.device)
    if verbose:
        print(f"Using device: {device}")

    sampler, vocab_size, process_cfg_raw = load_sampler(args)
    if verbose:
        if isinstance(sampler, MultipartiteSampler):
            print(f"Multipartite process with {len(sampler.components)} components → vocab={vocab_size}")
        else:
            print(f"Process vocab size: {vocab_size}")

    model = load_model(args, device, vocab_size)
    if verbose:
        print(f"Loaded model from {args.model_ckpt}")

    overall_metrics, per_token_metrics = run_evaluation_loop(
        model=model,
        sampler=sampler,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        seq_len=args.seq_len,
        seed=args.seed,
        device=device,
    )

    summary = summarize_results(overall_metrics, per_token_metrics, vocab_size)

    results = {
        "overall_metrics": overall_metrics,
        "per_token_metrics": per_token_metrics,
        "summary": summary,
        "evaluation_config": {
            "model_ckpt": args.model_ckpt,
            "process_config": args.process_config,
            "process_config_name": args.process_config_name,
            "process_preset": args.process_preset,
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "seq_len": args.seq_len,
            "vocab_size": vocab_size,
            "seed": args.seed,
        },
    }

    if verbose:
        print("\n=== Evaluation Summary ===")
        print(json.dumps(results["overall_metrics"], indent=2))
        print("Baseline comparisons:")
        print(json.dumps(summary["baselines"], indent=2))

    if save_outputs and args.output_dir:
        save_results(args.output_dir, results)

    return results


def evaluate_model_accuracy(**kwargs) -> Dict[str, Any]:
    """
    Convenience wrapper that accepts keyword arguments instead of a Namespace.
    """
    args = argparse.Namespace(**kwargs)
    return evaluate_model_accuracy_from_args(args)


def main() -> None:
    args = parse_args()
    evaluate_model_accuracy_from_args(args, save_outputs=bool(args.output_dir))


if __name__ == "__main__":
    main()
