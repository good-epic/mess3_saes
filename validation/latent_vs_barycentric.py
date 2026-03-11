#!/usr/bin/env python3
"""
2Bb: Barycentric coordinate vs. single-latent predictive power.

For each priority cluster, compares how well:
  (a) any single cluster latent's scalar activation, vs.
  (b) the full K-dim barycentric coordinate,
predicts:
  1. Vertex membership (K-class classification)
  2. Primary linguistic feature from 1c (spacy-based, e.g. gram_function)
  3. Next-token log-probabilities for the top-50 most diagnostic vocabulary items

Results are reported separately for:
  - near-vertex samples (from prepared_samples_dir)
  - simplex-interior samples (from simplex_dir, if provided)
  - combined (pooled) samples

GPU required (runs model forward passes; SAE+AANet skipped for simplex samples
since latent_acts and barycentric_coords are pre-saved in the JSONL).

Usage:
    python validation/latent_vs_barycentric.py \\
        --prepared_samples_dir outputs/interpretations/prepared_samples_broad_2 \\
        --simplex_dir outputs/simplex_samples \\
        --source_dir /workspace/outputs/selected_clusters_broad_2 \\
        --csv_dir /workspace/outputs/real_data_analysis_canonical \\
        --output_dir outputs/validation/latent_vs_barycentric \\
        --clusters 512_181,768_140,512_17,768_596 \\
        --model_name gemma-2-9b \\
        --sae_release gemma-scope-9b-pt-res \\
        --sae_id layer_20/width_16k/average_l0_68 \\
        --device cuda \\
        --cache_dir /workspace/hf_cache \\
        --hf_token $HF_TOKEN
"""

import os
import sys
import json
import argparse
import glob
import ast
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "AAnet"))

from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import login
from mess3_gmg_analysis_utils import sae_encode_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_model_path(csv_dir, n_clusters, cluster_id, k):
    """Find AANet .pt model in csv_dir."""
    path = Path(csv_dir) / f"clusters_{n_clusters}" / f"aanet_cluster_{cluster_id}_k{k}.pt"
    return path if path.exists() else None


def get_latent_indices(csv_dir, n_clusters, cluster_id):
    """Get latent indices for a cluster from the consolidated metrics CSV."""
    import pandas as pd
    csv_path = Path(csv_dir) / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    rows = df[df["cluster_id"] == cluster_id]
    if rows.empty:
        return None
    return ast.literal_eval(rows.iloc[0]["latent_indices"])

# Import feature extractors from 1c
from feature_regression import CLUSTER_EXTRACTORS


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="2Bb: Barycentric coordinate vs single-latent predictive power"
    )
    parser.add_argument("--prepared_samples_dir", type=str, required=True,
                        help="Dir with prepared vertex sample JSON files "
                             "(e.g. prepared_samples_broad_2/)")
    parser.add_argument("--simplex_dir", type=str, default=None,
                        help="Dir with simplex_samples JSONL files "
                             "(e.g. outputs/simplex_samples). If provided, also "
                             "analyzes interior simplex samples and reports combined results.")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Dir with refitted AANet .pt models. "
                             "Structure: source_dir/n{N}/cluster_{id}_k{k}_category*.pt")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Dir with consolidated_metrics CSV files")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/validation/latent_vs_barycentric")
    parser.add_argument("--clusters", type=str, default=None,
                        help="Comma-separated cluster keys. Default: all with feature extractors.")

    # Model & SAE
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/average_l0_68")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="/workspace/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)

    # AANet architecture
    parser.add_argument("--aanet_layer_widths", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--aanet_simplex_scale", type=float, default=1.0)
    parser.add_argument("--aanet_noise", type=float, default=0.05)

    # Analysis parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for model forward passes")
    parser.add_argument("--n_top_tokens", type=int, default=50,
                        help="Number of top-variance vocabulary tokens for next-token R² analysis")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Cross-validation folds for classification and R² estimates")
    parser.add_argument("--skip_nexttoken", action="store_true",
                        help="Skip next-token R² analysis (faster)")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip matplotlib output")
    parser.add_argument("--max_samples_per_vertex", type=int, default=None,
                        help="Cap vertex samples per vertex before inference.")
    parser.add_argument("--max_simplex_samples", type=int, default=500,
                        help="Cap total simplex interior samples per cluster.")

    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def load_vertex_samples(prepared_path):
    """Load vertex samples JSON. Returns (samples_by_vertex, metadata)."""
    with open(prepared_path) as f:
        data = json.load(f)
    samples_by_vertex = {
        int(v): samples for v, samples in data["vertices"].items()
    }
    return samples_by_vertex, data


def load_simplex_samples(simplex_dir, n_clusters, cluster_id, k,
                         max_samples=None, rng=None):
    """Load simplex interior samples from JSONL. Returns list of sample dicts."""
    path = (Path(simplex_dir) / f"n{n_clusters}"
            / f"cluster_{cluster_id}_k{k}_simplex_samples.jsonl")
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if max_samples and len(records) > max_samples:
        if rng is None:
            rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(records), max_samples, replace=False)
        records = [records[int(i)] for i in indices]
    return records


# =============================================================================
# Cross-validation helpers
# =============================================================================

def cv_balanced_accuracy(X, y, n_splits=5):
    """Stratified CV balanced accuracy for classification."""
    classes, counts = np.unique(y, return_counts=True)
    n_splits = min(n_splits, int(counts.min()))
    if n_splits < 2:
        return float(balanced_accuracy_score(y, np.full_like(y, classes[counts.argmax()])))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        X, y, cv=skf,
    )
    return float(balanced_accuracy_score(y, preds))


def cv_r2_single(x, y, n_splits=5):
    """CV R² for univariate linear regression (1 predictor)."""
    X = x.reshape(-1, 1)
    n = len(y)
    n_splits = min(n_splits, n)
    if n_splits < 2:
        return 0.0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s = []
    for train_idx, val_idx in kf.split(X):
        reg = LinearRegression().fit(X[train_idx], y[train_idx])
        ss_res = np.sum((y[val_idx] - reg.predict(X[val_idx])) ** 2)
        ss_tot = np.sum((y[val_idx] - y[val_idx].mean()) ** 2)
        r2s.append(0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot)
    return float(np.mean(r2s))


def cv_r2_multi(X, y, n_splits=5):
    """CV adjusted R² for K-predictor linear regression."""
    n, p = X.shape
    n_splits = min(n_splits, n)
    if n_splits < 2 or n <= p + 1:
        return 0.0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s = []
    for train_idx, val_idx in kf.split(X):
        reg = LinearRegression().fit(X[train_idx], y[train_idx])
        ss_res = np.sum((y[val_idx] - reg.predict(X[val_idx])) ** 2)
        ss_tot = np.sum((y[val_idx] - y[val_idx].mean()) ** 2)
        raw_r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
        n_val = len(val_idx)
        if n_val > p + 1:
            adj = 1.0 - (1.0 - raw_r2) * (n_val - 1) / (n_val - p - 1)
        else:
            adj = raw_r2
        r2s.append(adj)
    return float(np.mean(r2s))


# =============================================================================
# Model inference
# =============================================================================

def run_inference_on_vertex_samples(
    all_samples, model, sae, aanet, hook_name,
    cluster_indices_tensor, args,
    collect_logits=True,
):
    """Run model + SAE + AANet on vertex samples.

    Returns:
        latent_acts:   (N, n_latents) float32
        bary_coords:   (N, k)         float32
        next_logprobs: (N, vocab)     float16, or None
        valid_mask:    (N,)           bool
    """
    d_model = sae.W_dec.shape[1]
    W_c = sae.W_dec[cluster_indices_tensor, :]  # (n_latents, d_model)

    all_latent_acts = []
    all_bary = []
    all_lp = [] if collect_logits else None
    valid_mask = []

    for batch_start in range(0, len(all_samples), args.batch_size):
        batch = all_samples[batch_start: batch_start + args.batch_size]

        tokens = torch.tensor(
            [s["chunk_token_ids"] for s in batch],
            dtype=torch.long, device=args.device,
        )
        seq_len = tokens.shape[1]
        trigger_indices = [s["trigger_token_indices"][0] for s in batch]

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens, return_type="logits", names_filter=[hook_name]
            )
            acts = cache[hook_name]  # (B, seq_len, d_model)

            trigger_hidden = torch.stack([
                acts[i, ti, :] for i, ti in enumerate(trigger_indices)
            ])  # (B, d_model)

            feature_acts, _, _ = sae_encode_features(sae, trigger_hidden)
            lat_acts = feature_acts[:, cluster_indices_tensor]  # (B, n_latents)

            X_recon = lat_acts @ W_c  # (B, d_model)
            _, _, emb = aanet(X_recon)
            bary = aanet.euclidean_to_barycentric(emb)  # (B, k)

            all_latent_acts.append(lat_acts.float().cpu().numpy())
            all_bary.append(bary.float().cpu().numpy())

            if collect_logits:
                for i, ti in enumerate(trigger_indices):
                    if ti >= seq_len - 1:
                        valid_mask.append(False)
                        all_lp.append(torch.zeros(logits.shape[-1], dtype=torch.float16))
                    else:
                        valid_mask.append(True)
                        lp = F.log_softmax(logits[i, ti, :].float(), dim=-1)
                        all_lp.append(lp.half().cpu())
            else:
                valid_mask.extend([True] * len(batch))

            del logits, cache, acts, feature_acts

    latent_acts = np.concatenate(all_latent_acts, axis=0)
    bary_coords = np.concatenate(all_bary, axis=0)
    next_logprobs = None
    if collect_logits:
        next_logprobs = torch.stack(all_lp).numpy()

    return latent_acts, bary_coords, next_logprobs, np.array(valid_mask, dtype=bool)


def run_model_logits_for_simplex_samples(all_samples, model, hook_name, args):
    """Run model forward only on simplex samples; use saved latent_acts and bary_coords.

    Returns:
        latent_acts:   (N, n_latents) float32  — from saved JSONL fields
        bary_coords:   (N, k)         float32  — from saved JSONL fields
        next_logprobs: (N, vocab)     float16
        valid_mask:    (N,)           bool
        vertex_labels: (N,)           int — argmax of bary_coords per sample
    """
    all_lp = []
    valid_mask = []

    for batch_start in range(0, len(all_samples), args.batch_size):
        batch = all_samples[batch_start: batch_start + args.batch_size]

        tokens = torch.tensor(
            [s["chunk_token_ids"] for s in batch],
            dtype=torch.long, device=args.device,
        )
        seq_len = tokens.shape[1]
        trigger_indices = [s["trigger_token_indices"][0] for s in batch]

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens, return_type="logits", names_filter=[hook_name]
            )
            for i, ti in enumerate(trigger_indices):
                if ti >= seq_len - 1:
                    valid_mask.append(False)
                    all_lp.append(torch.zeros(logits.shape[-1], dtype=torch.float16))
                else:
                    valid_mask.append(True)
                    lp = F.log_softmax(logits[i, ti, :].float(), dim=-1)
                    all_lp.append(lp.half().cpu())
            del logits, cache

    latent_acts = np.array(
        [s["latent_acts"] for s in all_samples], dtype=np.float32
    )
    bary_coords = np.array(
        [s["barycentric_coords"] for s in all_samples], dtype=np.float32
    )
    next_logprobs = torch.stack(all_lp).numpy()
    vertex_labels = np.argmax(bary_coords, axis=1).astype(int)

    return (latent_acts, bary_coords, next_logprobs,
            np.array(valid_mask, dtype=bool), vertex_labels)


# =============================================================================
# Classification comparisons
# =============================================================================

def run_classification_comparison(latent_acts, bary_coords, labels, label_name, cv_folds):
    """Compare vertex/text-feature classification across predictors."""
    n_latents = latent_acts.shape[1]
    classes, counts = np.unique(labels, return_counts=True)
    majority_acc = float(counts.max() / counts.sum())

    best_latent_acc = 0.0
    best_latent_idx = -1
    for l in range(n_latents):
        x = latent_acts[:, l]
        acc = cv_balanced_accuracy(x.reshape(-1, 1), labels, n_splits=cv_folds)
        if acc > best_latent_acc:
            best_latent_acc = acc
            best_latent_idx = l

    bary_acc = cv_balanced_accuracy(bary_coords, labels, n_splits=cv_folds)
    improvement = bary_acc - best_latent_acc

    print(f"    [{label_name}] majority={majority_acc:.3f}  "
          f"best_latent={best_latent_acc:.3f} (l={best_latent_idx})  "
          f"bary={bary_acc:.3f}  Δ={improvement:+.3f}")

    return {
        "label_name": label_name,
        "majority_acc": majority_acc,
        "best_latent_balanced_acc": best_latent_acc,
        "best_latent_idx": int(best_latent_idx),
        "bary_balanced_acc": bary_acc,
        "improvement": improvement,
        "n_classes": len(classes),
        "n_samples": len(labels),
    }


# =============================================================================
# Next-token R² analysis
# =============================================================================

def run_nexttoken_r2(latent_acts, bary_coords, next_logprobs, valid_mask,
                     n_top_tokens, cv_folds, tokenizer):
    """Compare R² for predicting next-token log-probs: single latent vs. bary coord."""
    lat = latent_acts[valid_mask]
    bary = bary_coords[valid_mask]
    lp = next_logprobs[valid_mask].astype(np.float32)

    if lat.shape[0] < 2 * cv_folds:
        print(f"    Too few valid samples ({lat.shape[0]}) for next-token analysis")
        return None

    lp_var = lp.var(axis=0)
    top_token_ids = np.argsort(lp_var)[-n_top_tokens:]
    lp_top = lp[:, top_token_ids]

    top_token_strs = []
    for tid in top_token_ids:
        try:
            top_token_strs.append(repr(tokenizer.decode([int(tid)])))
        except Exception:
            top_token_strs.append(str(int(tid)))

    n_latents = lat.shape[1]

    print(f"    Computing R² for {n_top_tokens} tokens × {n_latents} latents + bary ...")

    r2_bary = np.zeros(n_top_tokens, dtype=np.float32)
    r2_latents = np.zeros((n_top_tokens, n_latents), dtype=np.float32)

    for t in range(n_top_tokens):
        y = lp_top[:, t]
        r2_bary[t] = cv_r2_multi(bary, y, n_splits=cv_folds)
        for l in range(n_latents):
            r2_latents[t, l] = cv_r2_single(lat[:, l], y, n_splits=cv_folds)

    r2_best_latent = r2_latents.max(axis=1)

    try:
        w_stat, w_pval = wilcoxon(r2_bary, r2_best_latent, alternative="greater")
    except Exception:
        w_stat, w_pval = float("nan"), float("nan")

    mean_bary = float(r2_bary.mean())
    mean_best = float(r2_best_latent.mean())
    frac_bary_wins = float((r2_bary > r2_best_latent).mean())

    print(f"    mean_R²: bary={mean_bary:.4f}  best_latent={mean_best:.4f}  "
          f"bary_wins={frac_bary_wins:.0%}  Wilcoxon_p={w_pval:.4f}")

    return {
        "n_valid_samples": int(lat.shape[0]),
        "n_top_tokens": n_top_tokens,
        "top_token_ids": top_token_ids.tolist(),
        "top_token_strs": top_token_strs,
        "r2_bary": r2_bary.tolist(),
        "r2_best_latent": r2_best_latent.tolist(),
        "r2_latents_matrix": r2_latents.tolist(),
        "mean_r2_bary": mean_bary,
        "mean_r2_best_latent": mean_best,
        "frac_bary_wins": frac_bary_wins,
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pval": float(w_pval),
    }


# =============================================================================
# Analysis split
# =============================================================================

def analyze_split(
    tag,
    all_samples,
    latent_acts,
    bary_coords,
    next_logprobs,
    valid_mask,
    vertex_labels,
    n_latents, k,
    args, tokenizer, nlp, cluster_key,
):
    """Run classification + next-token R² for one data split.

    Returns result dict with keys: tag, n_samples, classification, nexttoken_r2.
    """
    print(f"\n  [{tag}] {len(all_samples)} samples")

    # --- Classification: vertex membership ---
    print(f"  [{tag}] Classification comparisons:")
    classification_results = []

    vc = run_classification_comparison(
        latent_acts, bary_coords, vertex_labels,
        label_name="vertex_id", cv_folds=args.cv_folds,
    )
    classification_results.append(vc)

    # --- Classification: primary text feature ---
    if nlp is not None and cluster_key in CLUSTER_EXTRACTORS and all_samples:
        # Only vertex-derived samples have full_text trigger word metadata
        _, extractor_fn = CLUSTER_EXTRACTORS[cluster_key]
        rows = []
        sample_to_row = {}
        for si, s in enumerate(all_samples):
            for tw, twi in zip(s.get("trigger_words", []), s.get("trigger_word_indices", [])):
                if tw.strip():
                    sample_to_row[si] = len(rows)
                    rows.append({
                        "vertex_id": int(vertex_labels[si]),
                        "trigger_word": tw,
                        "trigger_word_idx": twi,
                        "full_text": s.get("full_text", ""),
                    })
                    break

        if rows:
            try:
                df, feature_cols, primary_feature = extractor_fn(rows, nlp)
                valid_tf = [
                    si for si, ri in sample_to_row.items()
                    if ri < len(df) and df.iloc[ri][primary_feature] is not None
                ]
                if len(valid_tf) > 10:
                    tf_labels_raw = [
                        df.iloc[sample_to_row[si]][primary_feature]
                        for si in valid_tf
                    ]
                    le = LabelEncoder()
                    tf_labels = le.fit_transform(tf_labels_raw)
                    lat_tf = latent_acts[valid_tf]
                    bary_tf = bary_coords[valid_tf]

                    tfc = run_classification_comparison(
                        lat_tf, bary_tf, tf_labels,
                        label_name=primary_feature, cv_folds=args.cv_folds,
                    )
                    tfc["label_classes"] = le.classes_.tolist()
                    classification_results.append(tfc)
            except Exception as e:
                print(f"    [{tag}] Text feature extraction failed: {e}")

    # --- Next-token R² ---
    nexttoken_result = None
    if not args.skip_nexttoken and next_logprobs is not None:
        print(f"\n  [{tag}] Next-token R² analysis (top-{args.n_top_tokens} tokens):")
        nexttoken_result = run_nexttoken_r2(
            latent_acts, bary_coords, next_logprobs, valid_mask,
            n_top_tokens=args.n_top_tokens,
            cv_folds=args.cv_folds,
            tokenizer=tokenizer,
        )

    return {
        "tag": tag,
        "n_samples": len(all_samples),
        "classification": classification_results,
        "nexttoken_r2": nexttoken_result,
    }


# =============================================================================
# Plots
# =============================================================================

def plot_r2_comparison(splits_results, cluster_key, output_dir):
    """Box/scatter comparison of R² distributions for all data splits."""
    splits = [r for r in splits_results if r.get("nexttoken_r2") is not None]
    if not splits:
        return

    n_splits = len(splits)
    fig, axes = plt.subplots(n_splits, 2, figsize=(10, 4.5 * n_splits), squeeze=False)

    for row, split in enumerate(splits):
        tag = split["tag"]
        nr = split["nexttoken_r2"]
        r2_bary = np.array(nr["r2_bary"])
        r2_best = np.array(nr["r2_best_latent"])

        # Box
        ax = axes[row, 0]
        ax.boxplot(
            [r2_best, r2_bary],
            labels=["Best single latent", "Barycentric coord"],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='navy', linewidth=2),
        )
        ax.set_ylabel("CV R²")
        ax.set_title(f"Cluster {cluster_key} [{tag}]\n"
                     f"Next-token R² (top-{len(r2_bary)} tokens, N={nr['n_valid_samples']})")
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

        # Scatter
        ax = axes[row, 1]
        lim = max(r2_best.max(), r2_bary.max()) * 1.1 + 0.01
        ax.scatter(r2_best, r2_bary, s=30, alpha=0.7, color='steelblue')
        ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8)
        ax.set_xlabel("Best single latent R²")
        ax.set_ylabel("Barycentric R²")
        ax.set_title("Paired comparison (each point = one token)")
        ax.set_xlim(-0.02, lim)
        ax.set_ylim(-0.02, lim)
        ax.set_aspect('equal')

    fig.tight_layout()
    out_path = Path(output_dir) / f"cluster_{cluster_key}_r2_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    R² plot → {out_path}")


def plot_classification_bar(splits_results, cluster_key, output_dir):
    """Bar chart comparing classification accuracies for all data splits."""
    for split in splits_results:
        tag = split["tag"]
        classification_results = split["classification"]
        targets = [r["label_name"] for r in classification_results]
        n = len(targets)
        if n == 0:
            continue

        x = np.arange(n)
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(6, n * 2), 4))

        majority = [r["majority_acc"] for r in classification_results]
        best_lat = [r["best_latent_balanced_acc"] for r in classification_results]
        bary = [r["bary_balanced_acc"] for r in classification_results]

        ax.bar(x - width, majority, width, label="Majority baseline", color='lightgray')
        ax.bar(x, best_lat, width, label="Best single latent", color='steelblue')
        ax.bar(x + width, bary, width, label="Barycentric coord", color='tomato')

        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(f"Cluster {cluster_key} [{tag}]: classification comparison")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color='gray', linewidth=0.5, linestyle='--')

        fig.tight_layout()
        out_path = Path(output_dir) / f"cluster_{cluster_key}_{tag}_classification.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Classification plot [{tag}] → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clusters:
        cluster_keys = [c.strip() for c in args.clusters.split(',') if c.strip()]
    else:
        cluster_keys = [k for k in CLUSTER_EXTRACTORS if k not in {"512_464", "768_484", "512_292"}]
        print(f"Using clusters with feature extractors: {cluster_keys}")

    if args.hf_token:
        login(token=args.hf_token)

    print(f"\nLoading model: {args.model_name}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
    )
    model.eval()
    tokenizer = model.tokenizer

    print(f"Loading SAE: {args.sae_release} / {args.sae_id}")
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    sae.eval()

    layer_idx = int(args.sae_id.split("/")[0].split("_")[1])
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spacy en_core_web_sm")
    except (ImportError, OSError) as e:
        print(f"WARNING: spacy not available ({e}). Text-feature analysis will be skipped.")
        nlp = None

    all_results = {}

    for cluster_key in cluster_keys:
        print(f"\n{'=' * 60}")
        print(f"Cluster {cluster_key}")

        n_clusters_str, cluster_id_str = cluster_key.split("_")
        n_clusters = int(n_clusters_str)
        cluster_id = int(cluster_id_str)

        # Load prepared vertex samples
        prepared_path = Path(args.prepared_samples_dir) / f"cluster_{cluster_key}.json"
        if not prepared_path.exists():
            print(f"  Prepared samples not found: {prepared_path}, skipping")
            continue

        samples_by_vertex, metadata = load_vertex_samples(prepared_path)
        k = metadata["k"]
        latent_indices = metadata.get("latent_indices") or get_latent_indices(
            args.csv_dir, n_clusters, cluster_id
        )
        if not latent_indices:
            print(f"  Cannot determine latent_indices, skipping")
            continue

        n_latents = len(latent_indices)
        total_vertex_samples = sum(len(v) for v in samples_by_vertex.values())
        print(f"  k={k}, n_latents={n_latents}, total vertex samples={total_vertex_samples}")

        # Load AANet
        from AAnet.AAnet_torch.models.AAnet_vanilla import AAnet_vanilla
        model_path = find_model_path(args.csv_dir, n_clusters, cluster_id, k)
        if model_path is None:
            print(f"  AANet model not found in {args.csv_dir}, skipping")
            continue

        d_model = sae.W_dec.shape[1]
        aanet = AAnet_vanilla(
            input_shape=d_model,
            n_archetypes=k,
            noise=args.aanet_noise,
            layer_widths=args.aanet_layer_widths,
            activation_out="tanh",
            simplex_scale=args.aanet_simplex_scale,
            device=args.device,
        )
        aanet.load_state_dict(torch.load(model_path, map_location=args.device))
        aanet.eval()

        cluster_indices_tensor = torch.tensor(
            latent_indices, device=args.device, dtype=torch.long
        )

        # ----------------------------------------------------------------
        # Build vertex sample list
        # ----------------------------------------------------------------
        rng = np.random.default_rng(seed=42)
        vertex_samples = []
        for vertex_id, samples in samples_by_vertex.items():
            valid = [s for s in samples if s.get("trigger_token_indices")]
            if args.max_samples_per_vertex and len(valid) > args.max_samples_per_vertex:
                indices = rng.choice(len(valid), args.max_samples_per_vertex, replace=False)
                valid = [valid[i] for i in indices]
            for sample in valid:
                vertex_samples.append({
                    "vertex_id": vertex_id,
                    "chunk_token_ids": sample["chunk_token_ids"],
                    "trigger_token_indices": sample["trigger_token_indices"],
                    "trigger_words": sample.get("trigger_words", [""]),
                    "trigger_word_indices": sample.get("trigger_word_indices", [0]),
                    "full_text": sample.get("full_text", ""),
                })

        splits_results = []

        # ----------------------------------------------------------------
        # Vertex split
        # ----------------------------------------------------------------
        if vertex_samples:
            print(f"\n  Running vertex inference on {len(vertex_samples)} samples ...")
            collect_logits = not args.skip_nexttoken
            lat_v, bary_v, lp_v, mask_v = run_inference_on_vertex_samples(
                vertex_samples, model, sae, aanet, hook_name,
                cluster_indices_tensor, args,
                collect_logits=collect_logits,
            )
            vlabels_v = np.array([s["vertex_id"] for s in vertex_samples])

            vertex_result = analyze_split(
                tag="vertex",
                all_samples=vertex_samples,
                latent_acts=lat_v,
                bary_coords=bary_v,
                next_logprobs=lp_v,
                valid_mask=mask_v,
                vertex_labels=vlabels_v,
                n_latents=n_latents, k=k,
                args=args, tokenizer=tokenizer, nlp=nlp,
                cluster_key=cluster_key,
            )
            splits_results.append(vertex_result)
        else:
            lat_v = bary_v = lp_v = mask_v = vlabels_v = None
            print("  No valid vertex samples, skipping vertex split")

        # ----------------------------------------------------------------
        # Simplex split (if simplex_dir provided)
        # ----------------------------------------------------------------
        lat_s = bary_s = lp_s = mask_s = vlabels_s = None
        if args.simplex_dir:
            simplex_records = load_simplex_samples(
                args.simplex_dir, n_clusters, cluster_id, k,
                max_samples=args.max_simplex_samples, rng=rng,
            )
            if simplex_records:
                print(f"\n  Running simplex model-logit inference on {len(simplex_records)} samples ...")
                lat_s, bary_s, lp_s, mask_s, vlabels_s = run_model_logits_for_simplex_samples(
                    simplex_records, model, hook_name, args,
                )
                simplex_result = analyze_split(
                    tag="simplex",
                    all_samples=simplex_records,
                    latent_acts=lat_s,
                    bary_coords=bary_s,
                    next_logprobs=lp_s,
                    valid_mask=mask_s,
                    vertex_labels=vlabels_s,
                    n_latents=n_latents, k=k,
                    args=args, tokenizer=tokenizer, nlp=nlp,
                    cluster_key=cluster_key,
                )
                splits_results.append(simplex_result)
            else:
                print(f"  No simplex samples found for cluster {cluster_key}")

        # ----------------------------------------------------------------
        # Plots
        # ----------------------------------------------------------------
        cluster_out_dir = output_dir / f"cluster_{cluster_key}"
        cluster_out_dir.mkdir(exist_ok=True)

        if not args.skip_plots and splits_results:
            plot_classification_bar(splits_results, cluster_key, cluster_out_dir)
            plot_r2_comparison(splits_results, cluster_key, cluster_out_dir)

        # ----------------------------------------------------------------
        # Save per-cluster results
        # ----------------------------------------------------------------
        result = {
            "cluster_key": cluster_key,
            "k": k,
            "n_latents": n_latents,
            "latent_indices": latent_indices,
            "splits": splits_results,
        }
        result_path = output_dir / f"cluster_{cluster_key}_results.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results → {result_path}")

        all_results[cluster_key] = {
            "k": k,
            "n_latents": n_latents,
            "splits": [
                {
                    "tag": s["tag"],
                    "n_samples": s["n_samples"],
                    "classification": s["classification"],
                    "nexttoken_r2": {
                        kk: vv for kk, vv in (s["nexttoken_r2"] or {}).items()
                        if kk not in {"r2_latents_matrix", "top_token_strs"}
                    } if s["nexttoken_r2"] else None,
                }
                for s in splits_results
            ],
        }

    # Save combined JSON
    combined_path = output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results → {combined_path}")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n{'=' * 110}")
    print("SUMMARY")
    print(f"{'=' * 110}")
    header = (f"{'Cluster':>12}  {'Split':>8}  {'k':>3}  "
              f"{'Target':>18}  {'Majority':>9}  {'BestLat':>8}  {'Bary':>7}  "
              f"{'Δ':>6}  {'BaryR²':>7}  {'LatR²':>7}  {'BaryWins':>9}  {'p':>8}")
    print(header)
    print("  " + "-" * 100)
    for ck, res in all_results.items():
        k_ = res["k"]
        n_lat = res["n_latents"]
        for split in res["splits"]:
            tag = split["tag"]
            nr = split.get("nexttoken_r2")
            bary_r2_str = f"{nr['mean_r2_bary']:.4f}" if nr else "    -   "
            lat_r2_str  = f"{nr['mean_r2_best_latent']:.4f}" if nr else "    -   "
            wins_str    = f"{nr['frac_bary_wins']:.0%}" if nr else "    -   "
            p_str       = f"{nr['wilcoxon_pval']:.4f}" if nr else "    -   "
            first_clf = split["classification"][0] if split["classification"] else {}
            print(
                f"  {ck:>12}  {tag:>8}  {k_:>3}  "
                f"{first_clf.get('label_name','?'):>18}  "
                f"{first_clf.get('majority_acc',0):>9.3f}  "
                f"{first_clf.get('best_latent_balanced_acc',0):>8.3f}  "
                f"{first_clf.get('bary_balanced_acc',0):>7.3f}  "
                f"{first_clf.get('improvement',0):>+6.3f}  "
                f"{bary_r2_str:>7}  {lat_r2_str:>7}  {wins_str:>9}  {p_str:>8}"
            )


if __name__ == "__main__":
    main()
