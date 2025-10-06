#!/usr/bin/env python3
"""
Evaluate transformer model predictive accuracy on multipartite generative process.

This script loads a trained model checkpoint and evaluates its predictive performance
by generating batches from the multipartite sampler and computing:
- Overall accuracy (fraction of tokens predicted correctly)
- Per-token accuracy and cross-entropy
- Top-k accuracy (k=1, 5, 10)
- Perplexity

Outputs include summary statistics, per-token metrics JSON, and histograms.
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
import json
from collections import defaultdict
from typing import Dict, Any

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

# Preset process configurations (same as fit_mess3_gmg.py)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model predictive accuracy")

    # Model and data
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--process_config", type=str, default="process_configs.json",
                        help="Path to process config JSON")
    parser.add_argument("--process_config_name", type=str, default=None,
                        help="Name of process config to use")
    parser.add_argument("--process_preset", type=str, default=None,
                        help="Named preset for generative process configuration")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--d_vocab", type=int, default=None)
    parser.add_argument("--act_fn", type=str, default="relu")

    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--n_batches", type=int, default=100,
                        help="Number of batches to evaluate")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/model_eval",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    return parser.parse_args()


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> Dict[str, Any]:
    """
    Compute predictive metrics for a single batch.

    Args:
        logits: (batch, seq, vocab_size) model predictions
        targets: (batch, seq) ground truth token IDs
        vocab_size: Size of vocabulary

    Returns:
        Dictionary with metrics and per-token statistics
    """
    batch_size, seq_len = targets.shape

    # Flatten for easier computation
    logits_flat = logits.reshape(-1, vocab_size)  # (batch*seq, vocab)
    targets_flat = targets.reshape(-1)  # (batch*seq,)

    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (batch*seq,)

    # Compute predictions
    predictions = logits_flat.argmax(dim=-1)  # (batch*seq,)
    correct = (predictions == targets_flat).float()  # (batch*seq,)

    # Top-k accuracy
    top_k_preds = logits_flat.topk(k=10, dim=-1).indices  # (batch*seq, 10)
    targets_expanded = targets_flat.unsqueeze(1)  # (batch*seq, 1)
    top_1_correct = (top_k_preds[:, :1] == targets_expanded).any(dim=1).float()
    top_5_correct = (top_k_preds[:, :5] == targets_expanded).any(dim=1).float()
    top_10_correct = (top_k_preds[:, :10] == targets_expanded).any(dim=1).float()

    # Aggregate overall metrics
    metrics = {
        'accuracy': correct.mean().item(),
        'top_5_accuracy': top_5_correct.mean().item(),
        'top_10_accuracy': top_10_correct.mean().item(),
        'cross_entropy': ce_loss.mean().item(),
        'perplexity': torch.exp(ce_loss.mean()).item(),
        'n_tokens': len(targets_flat),
    }

    # Per-token statistics (for aggregation across batches)
    per_token_stats = {
        'targets': targets_flat.cpu().numpy(),
        'correct': correct.cpu().numpy(),
        'ce_loss': ce_loss.cpu().numpy(),
    }

    return metrics, per_token_stats


def aggregate_per_token_metrics(
    all_targets: np.ndarray,
    all_correct: np.ndarray,
    all_ce_loss: np.ndarray,
    vocab_size: int,
) -> Dict[int, Dict[str, float]]:
    """
    Aggregate per-token metrics across all evaluation samples.

    Args:
        all_targets: All target token IDs
        all_correct: All correctness indicators
        all_ce_loss: All cross-entropy losses
        vocab_size: Size of vocabulary

    Returns:
        Dictionary mapping token_id -> metrics dict
    """
    per_token_metrics = {}

    for token_id in range(vocab_size):
        mask = all_targets == token_id
        if mask.sum() == 0:
            per_token_metrics[token_id] = {
                'count': 0,
                'accuracy': 0.0,
                'cross_entropy': 0.0,
                'frequency': 0.0,
            }
        else:
            per_token_metrics[token_id] = {
                'count': int(mask.sum()),
                'accuracy': float(all_correct[mask].mean()),
                'cross_entropy': float(all_ce_loss[mask].mean()),
                'frequency': float(mask.sum() / len(all_targets)),
            }

    return per_token_metrics


def plot_per_token_histograms(
    per_token_metrics: Dict[int, Dict[str, float]],
    output_dir: str,
):
    """
    Create histograms for per-token metrics.

    Args:
        per_token_metrics: Dictionary mapping token_id -> metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics for tokens that appeared at least once
    tokens_with_data = [tid for tid, m in per_token_metrics.items() if m['count'] > 0]
    accuracies = [per_token_metrics[tid]['accuracy'] for tid in tokens_with_data]
    cross_entropies = [per_token_metrics[tid]['cross_entropy'] for tid in tokens_with_data]
    frequencies = [per_token_metrics[tid]['frequency'] for tid in tokens_with_data]
    counts = [per_token_metrics[tid]['count'] for tid in tokens_with_data]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Accuracy histogram
    axes[0, 0].hist(accuracies, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Number of tokens')
    axes[0, 0].set_title('Per-token Accuracy Distribution')
    axes[0, 0].axvline(np.mean(accuracies), color='red', linestyle='--',
                       label=f'Mean: {np.mean(accuracies):.3f}')
    axes[0, 0].legend()

    # Cross-entropy histogram
    axes[0, 1].hist(cross_entropies, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Cross-entropy')
    axes[0, 1].set_ylabel('Number of tokens')
    axes[0, 1].set_title('Per-token Cross-Entropy Distribution')
    axes[0, 1].axvline(np.mean(cross_entropies), color='red', linestyle='--',
                       label=f'Mean: {np.mean(cross_entropies):.3f}')
    axes[0, 1].legend()

    # Frequency histogram (log scale)
    axes[1, 0].hist(frequencies, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Number of tokens')
    axes[1, 0].set_title('Per-token Frequency Distribution')
    axes[1, 0].set_yscale('log')

    # Count histogram (log scale)
    axes[1, 1].hist(counts, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_ylabel('Number of tokens')
    axes[1, 1].set_title('Per-token Count Distribution')
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_token_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-token histograms to {output_dir}/per_token_histograms.png")


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load process config and build sampler
    print("Loading process configuration...")
    process_cfg_raw, components, data_source = _load_process_stack(args, PRESET_PROCESS_CONFIGS)

    if isinstance(data_source, MultipartiteSampler):
        vocab_size = data_source.vocab_size
        print(f"Multipartite process with {len(data_source.components)} components → vocab={vocab_size}")
    else:
        vocab_size = data_source.vocab_size
        print(f"Single process → vocab={vocab_size}")

    # Load model
    print(f"Loading model from {args.model_ckpt}...")
    model, cfg = _load_transformer(args, device, vocab_size)

    # Load checkpoint
    ckpt = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    # Evaluation loop
    print(f"\nEvaluating on {args.n_batches} batches of size {args.batch_size}...")

    # Accumulators for overall metrics
    total_tokens = 0
    total_correct = 0
    total_top5_correct = 0
    total_top10_correct = 0
    total_ce_loss = 0.0

    # Accumulators for per-token metrics
    all_targets = []
    all_correct = []
    all_ce_loss = []

    key = jax.random.PRNGKey(args.seed)

    with torch.no_grad():
        for batch_idx in tqdm(range(args.n_batches), desc="Evaluating"):
            # Sample batch
            key, belief_states, product_tokens, component_observations = data_source.sample(
                key, args.batch_size, args.seq_len
            )
            tokens = torch.from_numpy(np.array(product_tokens)).long().to(device)

            # Forward pass
            logits = model(tokens)

            # Compute metrics (predict next token, so targets are shifted)
            # logits: (batch, seq, vocab), we use logits[:, :-1] to predict tokens[:, 1:]
            logits_for_pred = logits[:, :-1, :]
            targets = tokens[:, 1:]

            batch_metrics, per_token_stats = compute_batch_metrics(
                logits_for_pred, targets, vocab_size
            )

            # Accumulate overall metrics
            n_tokens = batch_metrics['n_tokens']
            total_tokens += n_tokens
            total_correct += batch_metrics['accuracy'] * n_tokens
            total_top5_correct += batch_metrics['top_5_accuracy'] * n_tokens
            total_top10_correct += batch_metrics['top_10_accuracy'] * n_tokens
            total_ce_loss += batch_metrics['cross_entropy'] * n_tokens

            # Accumulate per-token statistics
            all_targets.append(per_token_stats['targets'])
            all_correct.append(per_token_stats['correct'])
            all_ce_loss.append(per_token_stats['ce_loss'])

    # Concatenate all per-token data
    all_targets = np.concatenate(all_targets)
    all_correct = np.concatenate(all_correct)
    all_ce_loss = np.concatenate(all_ce_loss)

    # Compute overall metrics
    overall_accuracy = total_correct / total_tokens
    overall_top5_accuracy = total_top5_correct / total_tokens
    overall_top10_accuracy = total_top10_correct / total_tokens
    overall_ce = total_ce_loss / total_tokens
    overall_perplexity = np.exp(overall_ce)

    # Compute per-token metrics
    per_token_metrics = aggregate_per_token_metrics(
        all_targets, all_correct, all_ce_loss, vocab_size
    )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total tokens evaluated: {total_tokens:,}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy (top-1):     {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"  Top-5 Accuracy:       {overall_top5_accuracy:.4f} ({overall_top5_accuracy*100:.2f}%)")
    print(f"  Top-10 Accuracy:      {overall_top10_accuracy:.4f} ({overall_top10_accuracy*100:.2f}%)")
    print(f"  Cross-Entropy:        {overall_ce:.4f}")
    print(f"  Perplexity:           {overall_perplexity:.4f}")

    # Per-token summary statistics
    tokens_with_data = [tid for tid, m in per_token_metrics.items() if m['count'] > 0]
    accuracies = [per_token_metrics[tid]['accuracy'] for tid in tokens_with_data]
    cross_entropies = [per_token_metrics[tid]['cross_entropy'] for tid in tokens_with_data]

    print(f"\nPer-token Statistics ({len(tokens_with_data)} tokens observed):")
    print(f"  Accuracy:       mean={np.mean(accuracies):.4f}, "
          f"std={np.std(accuracies):.4f}, "
          f"min={np.min(accuracies):.4f}, "
          f"max={np.max(accuracies):.4f}")
    print(f"  Cross-Entropy:  mean={np.mean(cross_entropies):.4f}, "
          f"std={np.std(cross_entropies):.4f}, "
          f"min={np.min(cross_entropies):.4f}, "
          f"max={np.max(cross_entropies):.4f}")

    # Compute theoretical baselines
    uniform_ce = np.log(vocab_size)
    empirical_dist_ce = -np.sum([per_token_metrics[tid]['frequency'] * np.log(per_token_metrics[tid]['frequency'] + 1e-10)
                                   for tid in tokens_with_data])

    print(f"\nBaseline Comparisons:")
    print(f"  Uniform distribution CE:   {uniform_ce:.4f} (random guessing)")
    print(f"  Empirical marginal CE:     {empirical_dist_ce:.4f} (if model just learned token frequencies)")
    print(f"  Model CE:                  {overall_ce:.4f}")
    print(f"  CE reduction vs uniform:   {uniform_ce - overall_ce:.4f} nats ({(1 - overall_ce/uniform_ce)*100:.1f}% improvement)")
    print(f"  CE reduction vs empirical: {empirical_dist_ce - overall_ce:.4f} nats")
    print("="*60)

    # Save results
    results = {
        'overall_metrics': {
            'accuracy': float(overall_accuracy),
            'top_5_accuracy': float(overall_top5_accuracy),
            'top_10_accuracy': float(overall_top10_accuracy),
            'cross_entropy': float(overall_ce),
            'perplexity': float(overall_perplexity),
            'total_tokens': int(total_tokens),
        },
        'per_token_metrics': per_token_metrics,
        'evaluation_config': {
            'model_ckpt': args.model_ckpt,
            'process_config': args.process_config,
            'process_config_name': args.process_config_name,
            'batch_size': args.batch_size,
            'n_batches': args.n_batches,
            'seq_len': args.seq_len,
            'vocab_size': vocab_size,
            'seed': args.seed,
        }
    }

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Create histograms
    print("\nGenerating per-token histograms...")
    plot_per_token_histograms(per_token_metrics, args.output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
