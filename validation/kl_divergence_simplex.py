#!/usr/bin/env python3
"""
KL divergence distributions across the simplex (Validation 2A).

For each cluster, computes three distributions of pairwise divergences:
  1. Same-vertex pairs:   both samples from the same vertex
  2. Cross-vertex pairs:  samples from two different vertices
  3. Within-simplex pairs: samples from the random simplex pool

For each pair, computes:
  - Symmetric KL = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
  - JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = (P+Q)/2

Also runs a head/tail diagnostic to show what fraction of KL comes from
top-K tokens vs. the tail.

Requires GPU (model forward passes).

Usage:
    python validation/kl_divergence_simplex.py \\
        --clusters 512_464,512_504,512_292,768_484 \\
        --vertex_samples_dir outputs/interpretations/prepared_samples_current_no_whitespace \\
        --simplex_samples_dir /workspace/outputs/simplex_samples \\
        --output_dir outputs/validation/kl_divergence \\
        --model_name gemma-2-9b \\
        --device cuda \\
        --cache_dir /workspace/hf_cache \\
        --hf_token $HF_TOKEN
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer
from huggingface_hub import login


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="KL divergence distributions across the simplex")

    parser.add_argument("--clusters", type=str, required=True,
                        help="Comma-separated cluster keys, e.g. '512_464,512_504,768_484'")
    parser.add_argument("--control_clusters", type=str, default=None,
                        help="Comma-separated control cluster keys. These have no vertex samples; "
                             "only within-simplex distribution is computed for them.")
    parser.add_argument("--vertex_samples_dir", type=str, required=True,
                        help="Prepared samples directory (same format as top_tokens_by_vertex.py). "
                             "Contains {cluster_key}_prepared.json files.")
    parser.add_argument("--simplex_samples_dir", type=str, required=True,
                        help="Output of collect_simplex_samples.py. "
                             "Structure: simplex_samples_dir/n{N}/cluster_{id}_k{k}_simplex_samples.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")

    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for model forward passes and pairwise KL computation")
    parser.add_argument("--n_pairs_per_sample", type=int, default=20,
                        help="Number of random partners to compare each sample against")
    parser.add_argument("--max_samples_per_vertex", type=int, default=200,
                        help="Cap on near-vertex samples per vertex (after flattening to trigger level)")
    parser.add_argument("--max_simplex_samples", type=int, default=5000,
                        help="Cap on simplex samples to use")
    parser.add_argument("--n_diagnostic_pairs", type=int, default=1000,
                        help="Number of pairs for head/tail KL diagnostic")
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="/workspace/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)

    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def load_vertex_entries(vertex_samples_dir, cluster_key, max_per_vertex, rng):
    """Load near-vertex samples from vertex_samples.jsonl, flattened to one entry per trigger.

    Reads from selected_clusters_broad_2-style structure:
      vertex_samples_dir/n{N}/cluster_{id}_k*_*_vertex_samples.jsonl

    Returns dict: vertex_id (int) -> list of dicts with keys:
        chunk_ids: list[int]
        trigger_idx: int
        trigger_word: str
    """
    import re as _re
    n_clusters_str, cluster_id_str = cluster_key.split("_")
    n_clusters = int(n_clusters_str)
    cluster_id = int(cluster_id_str)

    pattern = str(
        Path(vertex_samples_dir) / f"n{n_clusters}"
        / f"cluster_{cluster_id}_k*_*_vertex_samples.jsonl"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No vertex_samples.jsonl found for {cluster_key} at {pattern}"
        )

    entries_by_vertex = {}
    with open(matches[0]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            vertex_id = int(record["vertex_id"])
            chunk_ids = record["chunk_token_ids"]
            trigger_indices = record.get("trigger_token_indices", [])
            trigger_words = record.get("trigger_words", [])
            for j, trig_idx in enumerate(trigger_indices):
                tw = trigger_words[j] if j < len(trigger_words) else ""
                entries_by_vertex.setdefault(vertex_id, []).append({
                    "chunk_ids": chunk_ids,
                    "trigger_idx": int(trig_idx),
                    "trigger_word": tw,
                })

    # Cap per vertex
    for vertex_id in list(entries_by_vertex.keys()):
        entries = entries_by_vertex[vertex_id]
        if max_per_vertex and len(entries) > max_per_vertex:
            idxs = rng.choice(len(entries), max_per_vertex, replace=False)
            entries_by_vertex[vertex_id] = [entries[i] for i in sorted(idxs)]

    return entries_by_vertex


def load_simplex_entries(simplex_samples_dir, cluster_key, max_samples, rng):
    """Load random simplex samples for a cluster.

    Returns list of dicts with keys: chunk_ids, trigger_idx, trigger_word,
    barycentric_coords.
    """
    n_clusters_str, cluster_id_str = cluster_key.split('_')
    n_clusters = int(n_clusters_str)
    cluster_id = int(cluster_id_str)

    pattern = str(
        Path(simplex_samples_dir) / f"n{n_clusters}" /
        f"cluster_{cluster_id}_k*_simplex_samples.jsonl"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No simplex samples found for {cluster_key} at {pattern}")

    entries = []
    with open(matches[0]) as f:
        for line in f:
            record = json.loads(line)
            chunk_ids = record["chunk_token_ids"]
            trig_idx = record["trigger_token_indices"][0]
            trig_word = record["trigger_words"][0] if record.get("trigger_words") else ""
            entries.append({
                "chunk_ids": chunk_ids,
                "trigger_idx": trig_idx,
                "trigger_word": trig_word,
                "barycentric_coords": record.get("barycentric_coords"),
            })

    if max_samples and len(entries) > max_samples:
        idxs = rng.choice(len(entries), max_samples, replace=False)
        entries = [entries[i] for i in sorted(idxs)]

    return entries


# =============================================================================
# Model forward passes
# =============================================================================

def get_log_probs_batch(entries, model, device):
    """Run model forward pass on a batch of entries and return log-probs at trigger positions.

    Args:
        entries: list of dicts with 'chunk_ids' (list[int]) and 'trigger_idx' (int)
        model: HookedTransformer

    Returns:
        log_probs: float32 tensor, shape (len(entries), vocab_size)
    """
    B = len(entries)
    seq_len = len(entries[0]["chunk_ids"])

    tokens = torch.tensor(
        [e["chunk_ids"] for e in entries],
        dtype=torch.long,
        device=device,
    )
    trigger_idxs = torch.tensor(
        [e["trigger_idx"] for e in entries],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits = model(tokens, return_type="logits")  # (B, seq_len, vocab)

    batch_idx = torch.arange(B, device=device)
    logits_at_trigger = logits[batch_idx, trigger_idxs, :].float()  # (B, vocab)
    log_probs = F.log_softmax(logits_at_trigger, dim=-1)  # (B, vocab)

    del logits, tokens
    torch.cuda.empty_cache()

    return log_probs.cpu()


# =============================================================================
# KL and JS divergence (operates on CPU float32 tensors)
# =============================================================================

EPS = 1e-10


def kl_div(log_p, log_q):
    """KL(P || Q). Shapes: (..., vocab). Returns (...)."""
    p = torch.exp(log_p)
    return (p * (log_p - log_q)).sum(dim=-1)


def sym_kl(log_p, log_q):
    """Symmetric KL = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)."""
    return 0.5 * (kl_div(log_p, log_q) + kl_div(log_q, log_p))


def js_div(log_p, log_q):
    """JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = (P+Q)/2."""
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=EPS))
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def compute_pairs_within_batch(log_probs, n_pairs_per_sample, rng):
    """Compute pairwise sym_kl and js_div for random within-batch pairs.

    For each sample i, picks n_pairs_per_sample random j != i from the batch.

    Args:
        log_probs: tensor (B, vocab)
        n_pairs_per_sample: int
        rng: np.random.RandomState

    Returns:
        sym_kl_vals: list of float
        js_vals: list of float
    """
    B = log_probs.shape[0]
    sym_kl_vals = []
    js_vals = []

    for i in range(B):
        candidates = [j for j in range(B) if j != i]
        if not candidates:
            continue
        n = min(n_pairs_per_sample, len(candidates))
        partner_idxs = rng.choice(candidates, n, replace=False).tolist()

        lp_i = log_probs[i].unsqueeze(0).expand(n, -1)  # (n, vocab)
        lp_j = log_probs[partner_idxs]                   # (n, vocab)

        sym_kl_vals.extend(sym_kl(lp_i, lp_j).tolist())
        js_vals.extend(js_div(lp_i, lp_j).tolist())

    return sym_kl_vals, js_vals


def compute_pairs_cross_batch(log_probs_a, log_probs_b, n_pairs_per_sample, rng):
    """Compute pairwise divergences between samples from two different groups.

    For each sample in log_probs_a, picks n_pairs_per_sample random partners
    from log_probs_b (and vice versa).

    Returns:
        sym_kl_vals: list of float
        js_vals: list of float
    """
    A = log_probs_a.shape[0]
    B = log_probs_b.shape[0]
    sym_kl_vals = []
    js_vals = []

    for i in range(A):
        n = min(n_pairs_per_sample, B)
        partner_idxs = rng.choice(B, n, replace=False).tolist()

        lp_i = log_probs_a[i].unsqueeze(0).expand(n, -1)
        lp_j = log_probs_b[partner_idxs]

        sym_kl_vals.extend(sym_kl(lp_i, lp_j).tolist())
        js_vals.extend(js_div(lp_i, lp_j).tolist())

    return sym_kl_vals, js_vals


# =============================================================================
# Head/tail diagnostic
# =============================================================================

def head_tail_diagnostic(log_probs_a, log_probs_b, n_pairs, rng):
    """For a sample of pairs, compute what fraction of KL weight sits in the head vs tail.

    Reports two versions:
      - signed: cumulative sum of contributions (p_i * log(p_i/q_i)) sorted by p descending.
        Can be misleading if positive and negative contributions cancel within the head.
      - absolute: cumulative sum of |contributions| / sum of |contributions|.
        Shows where the divergence mass lives regardless of sign direction.

    Both are sorted by p_i descending (most probable tokens first).
    """
    A = log_probs_a.shape[0]
    B = log_probs_b.shape[0]

    # Accumulators for signed and absolute versions
    signed_top10 = 0.0
    signed_top100 = 0.0
    signed_top1000 = 0.0

    abs_top10 = 0.0
    abs_top100 = 0.0
    abs_top1000 = 0.0

    cumulative_total_kl = 0.0
    count = 0

    n_pairs = min(n_pairs, A * B)

    for _ in range(n_pairs):
        i = int(rng.randint(0, A))
        j = int(rng.randint(0, B))

        p = torch.exp(log_probs_a[i])
        log_q = log_probs_b[j]

        # Per-token KL contributions: p_i * (log_p_i - log_q_i)
        contribs = p * (log_probs_a[i] - log_q)  # (vocab,)
        total_kl = contribs.sum().item()
        total_abs = contribs.abs().sum().item()

        if abs(total_kl) < 1e-10 or total_abs < 1e-10:
            continue

        # Sort by p descending (most probable tokens first)
        order = torch.argsort(p, descending=True)
        sorted_contribs = contribs[order]
        sorted_abs = sorted_contribs.abs()

        signed_cumsum = torch.cumsum(sorted_contribs, dim=0)
        abs_cumsum = torch.cumsum(sorted_abs, dim=0)

        signed_top10  += signed_cumsum[9].item()   / total_kl
        signed_top100 += signed_cumsum[99].item()  / total_kl
        signed_top1000 += signed_cumsum[999].item() / total_kl

        abs_top10  += abs_cumsum[9].item()   / total_abs
        abs_top100 += abs_cumsum[99].item()  / total_abs
        abs_top1000 += abs_cumsum[999].item() / total_abs

        cumulative_total_kl += total_kl
        count += 1

    if count == 0:
        return {}

    return {
        # Signed: cumulative KL / total KL (can exceed 100% or be negative due to cancellation)
        "signed_top10_pct":   round(100.0 * signed_top10   / count, 1),
        "signed_top100_pct":  round(100.0 * signed_top100  / count, 1),
        "signed_top1000_pct": round(100.0 * signed_top1000 / count, 1),
        # Absolute: cumulative |contrib| / total |contrib| (always 0-100%, no cancellation)
        "abs_top10_pct":   round(100.0 * abs_top10   / count, 1),
        "abs_top100_pct":  round(100.0 * abs_top100  / count, 1),
        "abs_top1000_pct": round(100.0 * abs_top1000 / count, 1),
        "abs_tail_pct":    round(100.0 * (1.0 - abs_top1000 / count), 1),
        "avg_kl": round(cumulative_total_kl / count, 4),
        "n_pairs_used": count,
    }


# =============================================================================
# Summary statistics
# =============================================================================

def summarize_distributions(same_sym_kl, cross_sym_kl, simplex_sym_kl,
                             same_js, cross_js, simplex_js):
    """Compute summary stats and z-scores between distributions."""
    def safe_stats(vals):
        if not vals:
            return {"mean": None, "std": None, "median": None, "n": 0}
        a = np.array(vals)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "n": len(a),
        }

    def z_score(vals_a, vals_b):
        """Z-score of means: (mean_a - mean_b) / sqrt(std_a^2/n_a + std_b^2/n_b)"""
        if not vals_a or not vals_b:
            return None
        a, b = np.array(vals_a), np.array(vals_b)
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        if se < 1e-10:
            return None
        return float((a.mean() - b.mean()) / se)

    return {
        "same_vertex_sym_kl": safe_stats(same_sym_kl),
        "cross_vertex_sym_kl": safe_stats(cross_sym_kl),
        "within_simplex_sym_kl": safe_stats(simplex_sym_kl),
        "same_vertex_js": safe_stats(same_js),
        "cross_vertex_js": safe_stats(cross_js),
        "within_simplex_js": safe_stats(simplex_js),
        "z_cross_vs_same_sym_kl": z_score(cross_sym_kl, same_sym_kl),
        "z_cross_vs_simplex_sym_kl": z_score(cross_sym_kl, simplex_sym_kl),
        "z_cross_vs_same_js": z_score(cross_js, same_js),
        "z_cross_vs_simplex_js": z_score(cross_js, simplex_js),
        "ratio_cross_vs_same_mean_sym_kl": (
            np.mean(cross_sym_kl) / np.mean(same_sym_kl)
            if same_sym_kl and cross_sym_kl else None
        ),
        "ratio_cross_vs_same_mean_js": (
            np.mean(cross_js) / np.mean(same_js)
            if same_js and cross_js else None
        ),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_distributions(same_sym_kl, cross_sym_kl, simplex_sym_kl,
                       same_js, cross_js, simplex_js,
                       cluster_key, output_path):
    """Plot overlaid histograms for the three distributions (sym_kl and JS)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"KL Divergence Distributions — {cluster_key}", fontsize=13)

    for ax, (same, cross, simplex, metric_name) in zip(
        axes,
        [
            (same_sym_kl, cross_sym_kl, simplex_sym_kl, "Symmetric KL"),
            (same_js, cross_js, simplex_js, "JS Divergence"),
        ],
    ):
        all_vals = same + cross + simplex
        if not all_vals:
            continue
        vmax = np.percentile(all_vals, 99)
        bins = np.linspace(0, vmax, 60)

        for vals, label, color in [
            (same, "Same vertex", "steelblue"),
            (cross, "Cross vertex", "firebrick"),
            (simplex, "Within simplex", "seagreen"),
        ]:
            if vals:
                ax.hist(vals, bins=bins, alpha=0.55, label=label, color=color,
                        density=True)

        ax.set_xlabel(metric_name)
        ax.set_ylabel("Density")
        ax.set_title(metric_name)
        ax.legend(fontsize=9)

        # Annotate means
        for vals, label, color in [
            (same, "same", "steelblue"),
            (cross, "cross", "firebrick"),
            (simplex, "simplex", "seagreen"),
        ]:
            if vals:
                ax.axvline(np.mean(vals), color=color, linestyle='--', linewidth=1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")


# =============================================================================
# Per-cluster processing
# =============================================================================

def process_control_cluster(cluster_key, args, model, rng):
    """Run within-simplex-only KL analysis for a control cluster (no vertex samples)."""
    print(f"\n{'='*60}")
    print(f"Processing control cluster: {cluster_key}")
    print(f"{'='*60}")

    simplex_entries = load_simplex_entries(
        args.simplex_samples_dir, cluster_key,
        args.max_simplex_samples, rng,
    )
    print(f"  Simplex samples: {len(simplex_entries)}")

    simplex_sym_kl_all = []
    simplex_js_all = []

    for batch_start in tqdm(range(0, len(simplex_entries), args.batch_size),
                            desc="    simplex pairs", leave=False):
        batch = simplex_entries[batch_start: batch_start + args.batch_size]
        if len(batch) < 2:
            continue
        batch_lp = get_log_probs_batch(batch, model, args.device)
        sk, js = compute_pairs_within_batch(batch_lp, args.n_pairs_per_sample, rng)
        simplex_sym_kl_all.extend(sk)
        simplex_js_all.extend(js)
        del batch_lp

    print(f"    {len(simplex_sym_kl_all)} within-simplex pairs")

    def safe_stats(vals):
        if not vals:
            return {"mean": None, "std": None, "median": None, "n": 0}
        a = np.array(vals)
        return {"mean": float(np.mean(a)), "std": float(np.std(a)),
                "median": float(np.median(a)), "n": len(a)}

    summary = {
        "within_simplex_sym_kl": safe_stats(simplex_sym_kl_all),
        "within_simplex_js": safe_stats(simplex_js_all),
        "is_control": True,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "cluster": cluster_key,
        "is_control": True,
        "n_simplex_samples": len(simplex_entries),
        "within_simplex": {"sym_kl": simplex_sym_kl_all, "js": simplex_js_all},
        "summary": summary,
    }

    json_path = out_dir / f"{cluster_key}_kl_distributions.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {json_path}")

    return results


def process_cluster(cluster_key, args, model, rng):
    """Run the full KL divergence analysis for one cluster."""
    print(f"\n{'='*60}")
    print(f"Processing cluster: {cluster_key}")
    print(f"{'='*60}")

    # Load data
    entries_by_vertex = load_vertex_entries(
        args.vertex_samples_dir, cluster_key,
        args.max_samples_per_vertex, rng,
    )
    simplex_entries = load_simplex_entries(
        args.simplex_samples_dir, cluster_key,
        args.max_simplex_samples, rng,
    )

    vertex_ids = sorted(entries_by_vertex.keys())
    n_vertices = len(vertex_ids)
    print(f"  Vertices: {vertex_ids}")
    print(f"  Samples per vertex: { {v: len(entries_by_vertex[v]) for v in vertex_ids} }")
    print(f"  Simplex samples: {len(simplex_entries)}")

    # -------------------------------------------------------------------------
    # Compute log-probs in batches
    # -------------------------------------------------------------------------
    print(f"\n  Computing log-probs for vertex samples...")
    log_probs_by_vertex = {}
    for v in vertex_ids:
        entries = entries_by_vertex[v]
        if not entries:
            log_probs_by_vertex[v] = torch.zeros(0)
            continue
        all_lp = []
        for batch_start in tqdm(range(0, len(entries), args.batch_size),
                                desc=f"    V{v}", leave=False):
            batch = entries[batch_start: batch_start + args.batch_size]
            lp = get_log_probs_batch(batch, model, args.device)
            all_lp.append(lp)
        log_probs_by_vertex[v] = torch.cat(all_lp, dim=0)  # (N_v, vocab)
        print(f"    V{v}: {log_probs_by_vertex[v].shape[0]} log-prob vectors")

    # Simplex log-probs are computed on-the-fly during pair computation (see below)
    # to avoid materializing 5000 × vocab (~5GB) in CPU RAM simultaneously.
    n_simplex_loaded = len(simplex_entries)
    print(f"  Simplex samples ready for batch-by-batch processing: {n_simplex_loaded}")

    # -------------------------------------------------------------------------
    # Distribution 1: Same-vertex pairs
    # -------------------------------------------------------------------------
    print(f"\n  Computing same-vertex pairs...")
    same_sym_kl_all = []
    same_js_all = []

    for v in vertex_ids:
        lp = log_probs_by_vertex[v]
        if lp.shape[0] < 2:
            continue
        for batch_start in range(0, lp.shape[0], args.batch_size):
            batch_lp = lp[batch_start: batch_start + args.batch_size]
            if batch_lp.shape[0] < 2:
                continue
            sk, js = compute_pairs_within_batch(batch_lp, args.n_pairs_per_sample, rng)
            same_sym_kl_all.extend(sk)
            same_js_all.extend(js)

    print(f"    {len(same_sym_kl_all)} same-vertex pairs")

    # -------------------------------------------------------------------------
    # Distribution 2: Cross-vertex pairs
    # -------------------------------------------------------------------------
    print(f"\n  Computing cross-vertex pairs...")
    cross_sym_kl_all = []
    cross_js_all = []

    for i_idx in range(n_vertices):
        for j_idx in range(i_idx + 1, n_vertices):
            v_i, v_j = vertex_ids[i_idx], vertex_ids[j_idx]
            lp_i = log_probs_by_vertex[v_i]
            lp_j = log_probs_by_vertex[v_j]
            if lp_i.shape[0] == 0 or lp_j.shape[0] == 0:
                continue

            # Process in structured batches: half from v_i, half from v_j
            half = args.batch_size // 2
            n_i = lp_i.shape[0]
            n_j = lp_j.shape[0]

            for batch_start in range(0, max(n_i, n_j), half):
                batch_i = lp_i[batch_start % n_i: (batch_start % n_i) + half]
                batch_j = lp_j[batch_start % n_j: (batch_start % n_j) + half]
                if batch_i.shape[0] == 0 or batch_j.shape[0] == 0:
                    continue
                sk, js = compute_pairs_cross_batch(
                    batch_i, batch_j, args.n_pairs_per_sample, rng
                )
                cross_sym_kl_all.extend(sk)
                cross_js_all.extend(js)

    print(f"    {len(cross_sym_kl_all)} cross-vertex pairs")

    # -------------------------------------------------------------------------
    # Distribution 3: Within-simplex pairs (batch-by-batch to avoid ~5GB materialization)
    # -------------------------------------------------------------------------
    print(f"\n  Computing within-simplex pairs (batch-by-batch)...")
    simplex_sym_kl_all = []
    simplex_js_all = []

    for batch_start in tqdm(range(0, len(simplex_entries), args.batch_size),
                            desc="    simplex pairs", leave=False):
        batch = simplex_entries[batch_start: batch_start + args.batch_size]
        if len(batch) < 2:
            continue
        batch_lp = get_log_probs_batch(batch, model, args.device)
        sk, js = compute_pairs_within_batch(batch_lp, args.n_pairs_per_sample, rng)
        simplex_sym_kl_all.extend(sk)
        simplex_js_all.extend(js)
        del batch_lp

    print(f"    {len(simplex_sym_kl_all)} within-simplex pairs")

    # -------------------------------------------------------------------------
    # Head/tail diagnostic
    # -------------------------------------------------------------------------
    print(f"\n  Running head/tail KL diagnostic...")
    diagnostic = {}
    if log_probs_by_vertex[vertex_ids[0]].shape[0] > 0 and log_probs_by_vertex[vertex_ids[-1]].shape[0] > 0:
        diag_lp_a = log_probs_by_vertex[vertex_ids[0]]
        diag_lp_b = log_probs_by_vertex[vertex_ids[-1]]
        diagnostic = head_tail_diagnostic(
            diag_lp_a, diag_lp_b, args.n_diagnostic_pairs, rng
        )
    print(f"    {diagnostic}")

    # -------------------------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------------------------
    summary = summarize_distributions(
        same_sym_kl_all, cross_sym_kl_all, simplex_sym_kl_all,
        same_js_all, cross_js_all, simplex_js_all,
    )

    print(f"\n  Summary:")
    for k_name, v in summary.items():
        if isinstance(v, dict):
            mean_val = v.get('mean')
            std_val = v.get('std')
            mean_str = f"{mean_val:.4f}" if mean_val is not None else "N/A"
            std_str = f"{std_val:.4f}" if std_val is not None else "N/A"
            print(f"    {k_name}: mean={mean_str}, std={std_str}, n={v.get('n', 0)}")
        elif v is not None:
            print(f"    {k_name}: {v:.3f}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "cluster": cluster_key,
        "n_vertices": n_vertices,
        "vertex_ids": vertex_ids,
        "n_vertex_samples": {str(v): log_probs_by_vertex[v].shape[0] for v in vertex_ids},
        "n_simplex_samples": n_simplex_loaded,
        "same_vertex": {
            "sym_kl": same_sym_kl_all,
            "js": same_js_all,
        },
        "cross_vertex": {
            "sym_kl": cross_sym_kl_all,
            "js": cross_js_all,
        },
        "within_simplex": {
            "sym_kl": simplex_sym_kl_all,
            "js": simplex_js_all,
        },
        "summary": {
            k: (float(v) if isinstance(v, float) else
                {kk: (float(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                if isinstance(v, dict) else v)
            for k, v in summary.items()
        },
        "head_tail_diagnostic": diagnostic,
    }

    json_path = out_dir / f"{cluster_key}_kl_distributions.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # Plot
    plot_path = out_dir / f"{cluster_key}_kl_plot.png"
    plot_distributions(
        same_sym_kl_all, cross_sym_kl_all, simplex_sym_kl_all,
        same_js_all, cross_js_all, simplex_js_all,
        cluster_key, plot_path,
    )

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    clusters = [c.strip() for c in args.clusters.split(',')]
    control_clusters = (
        [c.strip() for c in args.control_clusters.split(',')]
        if args.control_clusters else []
    )

    # Login to HuggingFace
    if args.hf_token:
        login(token=args.hf_token)

    # Load model
    print(f"\nLoading Model: {args.model_name}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        dtype=torch.float16,
    )
    model.eval()

    # Process priority clusters (full 3-distribution analysis)
    all_summaries = {}
    for cluster_key in clusters:
        try:
            results = process_cluster(cluster_key, args, model, rng)
            all_summaries[cluster_key] = results["summary"]
        except FileNotFoundError as e:
            print(f"  ERROR: {e}, skipping {cluster_key}")
        except Exception as e:
            print(f"  ERROR processing {cluster_key}: {e}")
            import traceback
            traceback.print_exc()

    # Process control clusters (within-simplex only, no vertex samples)
    control_summaries = {}
    for cluster_key in control_clusters:
        try:
            results = process_control_cluster(cluster_key, args, model, rng)
            control_summaries[cluster_key] = results["summary"]
        except FileNotFoundError as e:
            print(f"  ERROR: {e}, skipping control {cluster_key}")
        except Exception as e:
            print(f"  ERROR processing control {cluster_key}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined summary across all clusters
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / "kl_divergence_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({"priority": all_summaries, "controls": control_summaries}, f, indent=2)
    print(f"\nCombined summary saved: {summary_path}")

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A"

    # Print comparison table — priority clusters
    print(f"\n{'='*80}")
    print(f"PRIORITY CLUSTERS")
    print(f"{'Cluster':<15} {'mean_cross_kl':>14} {'mean_same_kl':>13} {'z_cross_vs_same':>16} {'mean_simplex_kl':>16}")
    print(f"{'='*80}")
    for cluster_key, s in all_summaries.items():
        cross_kl = s.get("cross_vertex_sym_kl", {}).get("mean")
        same_kl = s.get("same_vertex_sym_kl", {}).get("mean")
        z_score = s.get("z_cross_vs_same_sym_kl")
        simplex_kl = s.get("within_simplex_sym_kl", {}).get("mean")
        print(f"{cluster_key:<15} {fmt(cross_kl):>14} {fmt(same_kl):>13} {fmt(z_score):>16} {fmt(simplex_kl):>16}")

    # Print control cluster table
    if control_summaries:
        print(f"\n{'='*80}")
        print(f"CONTROL CLUSTERS (within-simplex only)")
        print(f"{'Cluster':<15} {'mean_simplex_kl':>16} {'mean_simplex_js':>16}")
        print(f"{'='*80}")
        for cluster_key, s in control_summaries.items():
            simplex_kl = s.get("within_simplex_sym_kl", {}).get("mean")
            simplex_js = s.get("within_simplex_js", {}).get("mean")
            print(f"{cluster_key:<15} {fmt(simplex_kl):>16} {fmt(simplex_js):>16}")


if __name__ == "__main__":
    main()
