#!/usr/bin/env python3
"""PCA truncation experiment for Mess3 transformer."""
from __future__ import annotations
import os
import sys
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp

from multipartite_utils import _load_process_stack, _load_transformer, _resolve_device, MultipartiteSampler

PRESET = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy under PCA rank-k truncation")
    parser.add_argument("--model-ckpt", type=str,
                        default="outputs/checkpoints/mess3_x_0.05_a_0.95/checkpoint_step_50039_best.pt")
    parser.add_argument("--process-config", type=str, default="process_configs.json")
    parser.add_argument("--process-config-name", type=str, default="mess3_x_0.05_a_0.95")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--d_vocab", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--pca-samples", type=int, default=4096)
    parser.add_argument("--eval-batches", type=int, default=250)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="outputs/mess3_rankk_pca")
    parser.add_argument("--kl", action="store_true",
                        help="If set, report per-token KL divergence instead of accuracy")
    return parser.parse_args()


def ensure_sampler(data_source, components):
    if isinstance(data_source, MultipartiteSampler):
        return data_source
    return MultipartiteSampler(components)


def collect_pca_data(model, sampler, layer_hook, batch_size, seq_len, num_samples, seed, device):
    activations = []
    beliefs = []
    key = jax.random.PRNGKey(seed)
    needed = num_samples
    model.eval()
    from tqdm import tqdm
    pbar = tqdm(total=num_samples, desc="Collecting PCA data")
    while needed > 0:
        key, belief_states, product_tokens, component_obs = sampler.sample(key, batch_size, seq_len)
        tokens = torch.from_numpy(np.array(product_tokens)).long().to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[layer_hook])
        layer_act = cache[layer_hook].cpu().numpy()
        belief = np.array(belief_states)
        steps = min(layer_act.shape[1], belief.shape[1])
        layer_act = layer_act[:, :steps, :]
        belief = belief[:, :steps, :]
        activations.append(layer_act.reshape(-1, layer_act.shape[-1]))
        beliefs.append(belief.reshape(-1, belief.shape[-1]))
        got = layer_act.shape[0] * steps
        needed -= got
        pbar.update(got)
    pbar.close()
    activations = np.concatenate(activations, axis=0)[:num_samples]
    beliefs = np.concatenate(beliefs, axis=0)[:num_samples]
    return activations, beliefs


def fit_pca(activations):
    mean = activations.mean(axis=0)
    centered = activations - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt
    singular_values = S
    explained = (S ** 2) / (centered.shape[0] - 1)
    return mean, components, singular_values, explained


def reconstruct(vec, mean, components, k):
    if k is None:
        return vec
    if isinstance(vec, torch.Tensor):
        device = vec.device
        x = vec.detach().cpu().numpy()
    else:
        device = None
        x = np.asarray(vec)
    original_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    centered = x2d - mean
    coeffs = centered @ components[:k].T
    recon = coeffs @ components[:k] + mean
    recon = recon.reshape(original_shape)
    if isinstance(vec, torch.Tensor):
        return torch.from_numpy(recon).to(device)
    return recon


def fit_linear_predictors(activations, beliefs):
    X = np.hstack([activations, np.ones((activations.shape[0], 1))])
    weights = []
    for dim in range(beliefs.shape[1]):
        y = beliefs[:, dim]
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        weights.append(coef)
    return np.stack(weights)


def predict_belief(weights, activation):
    x = activation.detach().cpu().numpy()
    if x.ndim == 1:
        x_aug = np.append(x, 1.0)
        return weights @ x_aug
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    x_aug = np.concatenate([x, ones], axis=1)
    return x_aug @ weights.T


def evaluate(model, sampler, args, mean, components, lin_weights, k_values):
    device = next(model.parameters()).device
    layer_hook = f"blocks.{args.layer}.hook_resid_post"
    key = jax.random.PRNGKey(args.seed + 100)
    use_kl = bool(getattr(args, "kl", False))
    results = {}
    for k in k_values:
        entry = {
            "correct": 0.0,
            "count": 0,
            "ce": 0.0,
            "belief_rmse_sum": 0.0,
            "belief_rmse_count": 0,
        }
        if use_kl:
            entry["kl_sum"] = 0.0
            entry["kl_count"] = 0
        results[k] = entry
    from tqdm import tqdm
    pbar = tqdm(range(args.eval_batches), desc="Evaluating rank truncations (token-wise)")
    for batch_idx in pbar:
        key, belief_states, product_tokens, _ = sampler.sample(key, args.batch_size, args.seq_len)
        tokens = torch.from_numpy(np.array(product_tokens)).long().to(device)
        true_beliefs = np.array(belief_states)
        with torch.no_grad():
            base_logits, cache = model.run_with_cache(tokens, names_filter=[layer_hook])
        layer_act = cache[layer_hook].detach()
        seq_len = layer_act.shape[1]
        tokens_len = tokens.shape[1]
        belief_len = true_beliefs.shape[1]
        usable_len = min(seq_len, tokens_len, belief_len)
        if usable_len != seq_len:
            layer_act = layer_act[:, :usable_len, :]
        if usable_len != tokens_len:
            tokens = tokens[:, :usable_len]
            base_logits = base_logits[:, :usable_len, :]
        if usable_len != belief_len:
            true_beliefs = true_beliefs[:, :usable_len, :]
        batch_size, usable_len, d_model = layer_act.shape
        flat_act = layer_act.reshape(-1, d_model)
        reconstructed_cache = {None: layer_act}
        base_log_probs = F.log_softmax(base_logits[:, :usable_len, :], dim=-1)
        base_probs = base_log_probs.exp()
        for k in k_values:
            if k is None:
                continue
            recon = reconstruct(flat_act, mean, components, k).reshape(batch_size, usable_len, d_model)
            reconstructed_cache[k] = recon
        for pos in range(usable_len):
            belief_target = true_beliefs[:, pos, :]
            base_lp_pos = base_log_probs[:, pos, :]
            base_prob_pos = base_probs[:, pos, :]
            for k in k_values:
                acts_for_pos = reconstructed_cache[k][:, pos, :]
                if k is None:
                    logits = base_logits
                else:
                    repl = acts_for_pos

                    def hook_fn(value, hook, replacement=repl, position=pos):
                        value = value.clone()
                        value[:, position, :] = replacement.to(value.device)
                        return value

                    with torch.no_grad():
                        logits = model.run_with_hooks(tokens, fwd_hooks=[(layer_hook, hook_fn)])
                logits_pos = logits[:, pos, :]
                if pos < usable_len - 1:
                    targets = tokens[:, pos + 1]
                    preds = logits_pos.argmax(dim=-1)
                    results[k]["correct"] += (preds == targets).float().sum().item()
                    results[k]["ce"] += F.cross_entropy(logits_pos, targets, reduction='sum').item()
                    results[k]["count"] += targets.numel()
                if use_kl:
                    if k is None:
                        log_probs_mod = base_lp_pos
                    else:
                        log_probs_mod = F.log_softmax(logits_pos, dim=-1)
                    kl = torch.sum(base_prob_pos * (base_lp_pos - log_probs_mod), dim=-1)
                    results[k]["kl_sum"] += float(kl.sum().item())
                    results[k]["kl_count"] += kl.numel()
                belief_pred = predict_belief(lin_weights, acts_for_pos)
                rmse = np.sqrt(np.mean((belief_pred - belief_target) ** 2, axis=-1))
                results[k]["belief_rmse_sum"] += float(rmse.sum())
                results[k]["belief_rmse_count"] += rmse.size
    for k in k_values:
        total = results[k]["count"]
        if total:
            results[k]["accuracy"] = results[k]["correct"] / total
            results[k]["cross_entropy"] = results[k]["ce"] / total
        else:
            results[k]["accuracy"] = float("nan")
            results[k]["cross_entropy"] = float("nan")
        if results[k]["belief_rmse_count"]:
            results[k]["belief_rmse"] = results[k]["belief_rmse_sum"] / results[k]["belief_rmse_count"]
        else:
            results[k]["belief_rmse"] = float("nan")
        del results[k]["belief_rmse_sum"]
        del results[k]["belief_rmse_count"]
        if use_kl:
            if results[k]["kl_count"]:
                results[k]["kl_divergence"] = results[k]["kl_sum"] / results[k]["kl_count"]
            else:
                results[k]["kl_divergence"] = float("nan")
            del results[k]["kl_sum"]
            del results[k]["kl_count"]
    return results


def plot_results(results, k_display, output_dir, use_kl=False, filename_suffix=None):
    last = k_display[-1]
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    def lookup(metric):
        series = []
        for x in k_display:
            key = None if x == last else x
            series.append(results[key][metric])
        return series

    if use_kl:
        acc = lookup("accuracy")
        belief_rmse = lookup("belief_rmse")
        kl = lookup("kl_divergence")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
        axes[0].plot(k_display, acc, marker='o', color='tab:blue')
        axes[0].set_title('Accuracy')
        axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axes[1].plot(k_display, belief_rmse, marker='s', color='tab:orange')
        axes[1].set_title('Belief RMSE')
        axes[2].plot(k_display, kl, marker='^', color='tab:green')
        axes[2].set_title('KL Divergence')
        for ax in axes:
            ax.set_xlabel('PCA rank k')
        axes[0].set_ylabel('Value')
        fig.tight_layout()
    else:
        acc = lookup("accuracy")
        belief_rmse = lookup("belief_rmse")
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(k_display, acc, marker='o', color='tab:blue', label='Transformer accuracy')
        ax1.set_xlabel('PCA rank k')
        ax1.set_ylabel('Accuracy', color='tab:blue')
        ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(k_display, belief_rmse, marker='s', color='tab:orange', label='Belief RMSE')
        ax2.set_ylabel('Belief RMSE', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        fig.tight_layout()
    fig.savefig(Path(output_dir) / f'rank_k_tradeoff{suffix}.png', dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    process_cfg_raw, components, data_source = _load_process_stack(args, PRESET)
    sampler = ensure_sampler(data_source, components)
    model, cfg = _load_transformer(args, device, sampler.vocab_size)
    checkpoint = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    if hasattr(cfg, "n_ctx") and args.n_ctx != cfg.n_ctx:
        print(f"Overriding provided n_ctx={args.n_ctx} with model n_ctx={cfg.n_ctx}")
        args.n_ctx = cfg.n_ctx
        if args.seq_len > args.n_ctx:
            print(f"Clamping seq_len from {args.seq_len} to model n_ctx={args.n_ctx}")
            args.seq_len = args.n_ctx
    layer_hook = f"blocks.{args.layer}.hook_resid_post"
    acts, beliefs = collect_pca_data(model, sampler, layer_hook, args.batch_size, args.seq_len,
                                     args.pca_samples, args.seed, device)
    mean, components, sing_vals, explained = fit_pca(acts)
    lin_weights = fit_linear_predictors(acts, beliefs)
    k_values = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, None]
    results = evaluate(model, sampler, args, mean, components, lin_weights, k_values)
    metrics_path = Path(args.output_dir) / f'rank_k_metrics_{args.process_config_name}.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    k_display = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24]
    plot_results(results, k_display, args.output_dir, use_kl=args.kl, filename_suffix=args.process_config_name)
    print(f"Saved metrics to {metrics_path} and plots to {args.output_dir} (suffix {args.process_config_name})")

if __name__ == '__main__':
    main()
