#!/usr/bin/env python3
"""
Top predicted tokens by vertex — qualitative validation for AANet simplex interpretations.

For each vertex of each priority cluster, collects the model's top-k next-token
predictions at trigger positions and checks whether they categorically differ
between vertices in a way that matches the interpretation.

E.g., for 512_464 ("common vs proper nouns"), do model predictions after V0
trigger words look different from V1? If V0 triggers are followed by
prepositions/articles and V1 by possessives/verbs, that supports the
interpretation.

Only needs the model (no SAE or AANet) — just logits at trigger positions.

Usage:
    python validation/top_tokens_by_vertex.py \
        --prepared_samples_dir outputs/interpretations/prepared_samples_current_no_whitespace \
        --output_dir outputs/validation/top_tokens \
        --clusters 512_464

    # Quick test:
    python validation/top_tokens_by_vertex.py \
        --prepared_samples_dir outputs/interpretations/prepared_samples_current_no_whitespace \
        --output_dir outputs/validation/top_tokens \
        --clusters 512_464 --max_samples_per_vertex 10
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm


# =============================================================================
# Data Loading
# =============================================================================

def load_vertex_samples(prepared_path):
    """Load prepared vertex samples from JSON file.

    Returns dict mapping vertex_id (int) -> list of sample dicts.
    """
    with open(prepared_path) as f:
        data = json.load(f)

    samples_by_vertex = {}
    for vertex_str, samples in data["vertices"].items():
        vertex_id = int(vertex_str)
        samples_by_vertex[vertex_id] = samples

    return samples_by_vertex, data


# =============================================================================
# Model Predictions
# =============================================================================

def collect_predictions(samples_by_vertex, model, tokenizer, top_k, max_samples_per_vertex, batch_size, device):
    """Run model forward passes and extract top-k predictions at trigger positions.

    For each trigger position i, takes logits[batch, i, :] — the model's predicted
    next-token distribution after processing the trigger token.

    Returns dict: vertex_id -> list of dicts with keys:
        top_tokens: list of (token_str, probability) for top-k
        trigger_word: the trigger word string
        trigger_pos: the trigger token position
    """
    results_by_vertex = {}

    for vertex_id in sorted(samples_by_vertex.keys()):
        samples = samples_by_vertex[vertex_id]

        # Cap samples
        if max_samples_per_vertex and len(samples) > max_samples_per_vertex:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(samples), max_samples_per_vertex, replace=False)
            samples = [samples[i] for i in sorted(indices)]

        # Flatten: one entry per trigger position within each sample
        entries = []
        for sample in samples:
            chunk_ids = sample["chunk_token_ids"]
            trigger_indices = sample["trigger_token_indices"]
            trigger_words = sample.get("trigger_words", [])

            for j, trig_idx in enumerate(trigger_indices):
                tw = trigger_words[j] if j < len(trigger_words) else ""
                entries.append({
                    "chunk_ids": chunk_ids,
                    "trigger_idx": trig_idx,
                    "trigger_word": tw,
                })

        if not entries:
            results_by_vertex[vertex_id] = []
            continue

        print(f"  Vertex {vertex_id}: {len(entries)} trigger positions from {len(samples)} samples")

        # Batch forward passes
        vertex_results = []
        for batch_start in tqdm(range(0, len(entries), batch_size),
                                desc=f"    V{vertex_id}", leave=False):
            batch_entries = entries[batch_start : batch_start + batch_size]
            batch_tokens = torch.tensor(
                [e["chunk_ids"] for e in batch_entries],
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                logits = model(batch_tokens, return_type="logits")  # (B, seq_len, vocab)

            # Extract logits at trigger positions and apply softmax
            for i, entry in enumerate(batch_entries):
                trig_idx = entry["trigger_idx"]
                token_logits = logits[i, trig_idx, :]  # (vocab,)
                probs = torch.softmax(token_logits.float(), dim=-1)

                top_probs, top_ids = torch.topk(probs, top_k)
                top_tokens = []
                for tok_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
                    tok_str = tokenizer.decode([tok_id])
                    top_tokens.append((tok_str, prob))

                vertex_results.append({
                    "top_tokens": top_tokens,
                    "trigger_word": entry["trigger_word"],
                    "trigger_pos": trig_idx,
                })

            del logits
            torch.cuda.empty_cache()

        results_by_vertex[vertex_id] = vertex_results

    return results_by_vertex


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_by_vertex(results_by_vertex, top_k):
    """Aggregate top-k predictions per vertex.

    Returns dict: vertex_id -> {
        n_samples: int,
        top1_counts: Counter of most-predicted-token -> count,
        topk_counts: Counter of any-top-k-token -> count,
        top1_total: total number of predictions,
        avg_top1_prob: average probability of the top-1 prediction,
        avg_top1_entropy: average entropy of the top-k distribution,
    }
    """
    aggregated = {}

    for vertex_id, results in results_by_vertex.items():
        top1_counts = Counter()
        topk_counts = Counter()
        top1_probs = []
        entropies = []

        for r in results:
            tokens_probs = r["top_tokens"]
            if not tokens_probs:
                continue

            # Top-1
            top1_tok, top1_prob = tokens_probs[0]
            top1_counts[top1_tok] += 1
            top1_probs.append(top1_prob)

            # All top-k
            for tok, prob in tokens_probs:
                topk_counts[tok] += 1

            # Entropy of the top-k distribution
            probs_arr = np.array([p for _, p in tokens_probs])
            probs_arr = probs_arr / probs_arr.sum()  # renormalize over top-k
            entropy = -np.sum(probs_arr * np.log(probs_arr + 1e-12))
            entropies.append(entropy)

        aggregated[vertex_id] = {
            "n_samples": len(results),
            "top1_counts": top1_counts,
            "topk_counts": topk_counts,
            "top1_total": sum(top1_counts.values()),
            "avg_top1_prob": float(np.mean(top1_probs)) if top1_probs else 0.0,
            "avg_topk_entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    return aggregated


# =============================================================================
# Reporting
# =============================================================================

def print_report(cluster_key, description, aggregated, n_display=10):
    """Pretty-print per-vertex token distributions."""
    print(f"\nCLUSTER {cluster_key}: {description}")
    print("=" * 80)

    for vertex_id in sorted(aggregated.keys()):
        agg = aggregated[vertex_id]
        n = agg["n_samples"]
        total = agg["top1_total"]

        print(f"\nVertex {vertex_id} — {n} trigger positions:")
        print(f"  Avg top-1 probability: {agg['avg_top1_prob']:.3f}")
        print(f"  Avg top-k entropy (renormalized): {agg['avg_topk_entropy']:.3f}")

        # Most common top-1 predictions
        print(f"  Most common top-1 predictions:")
        top1_items = agg["top1_counts"].most_common(n_display)
        lines = []
        for tok, count in top1_items:
            pct = 100.0 * count / total if total > 0 else 0
            # Repr to show whitespace clearly
            tok_display = repr(tok)
            lines.append(f"    {tok_display:<20s} {pct:5.1f}%  ({count})")
        print("\n".join(lines))

        # Most common in top-k pool
        print(f"  Most common tokens in top-k pool:")
        topk_items = agg["topk_counts"].most_common(n_display)
        lines = []
        for tok, count in topk_items:
            tok_display = repr(tok)
            lines.append(f"    {tok_display:<20s} {count}")
        print("\n".join(lines))


def build_json_results(cluster_key, metadata, aggregated, results_by_vertex, top_k):
    """Build a JSON-serializable results dict for one cluster."""
    vertices_json = {}
    for vertex_id in sorted(aggregated.keys()):
        agg = aggregated[vertex_id]
        vertices_json[str(vertex_id)] = {
            "n_samples": agg["n_samples"],
            "avg_top1_prob": agg["avg_top1_prob"],
            "avg_topk_entropy": agg["avg_topk_entropy"],
            "top1_distribution": [
                {"token": tok, "count": count,
                 "pct": round(100.0 * count / agg["top1_total"], 2) if agg["top1_total"] > 0 else 0}
                for tok, count in agg["top1_counts"].most_common(30)
            ],
            "topk_distribution": [
                {"token": tok, "count": count}
                for tok, count in agg["topk_counts"].most_common(50)
            ],
        }

    return {
        "cluster_key": cluster_key,
        "k": metadata.get("k"),
        "n_latents": metadata.get("n_latents"),
        "latent_indices": metadata.get("latent_indices", []),
        "top_k": top_k,
        "vertices": vertices_json,
    }


# =============================================================================
# Cluster Descriptions
# =============================================================================

CLUSTER_DESCRIPTIONS = {
    "512_464": "Common nouns vs proper nouns",
    "512_504": "Function words vs content nouns vs prepositions",
    "512_292": "Procedural instructions vs deliberative processes",
    "768_484": "Structured data vs narrative prose",
    "512_261": "Cluster 512_261",
    "768_210": "Cluster 768_210",
    "768_455": "Cluster 768_455",
}


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Top predicted tokens by vertex — qualitative validation"
    )
    parser.add_argument("--prepared_samples_dir", type=str, required=True,
                        help="Directory with prepared sample JSON files (use no_whitespace version)")
    parser.add_argument("--output_dir", type=str, default="outputs/validation/top_tokens")
    parser.add_argument("--clusters", type=str, default="512_464,512_504,512_292,768_484",
                        help="Comma-separated cluster keys or 'all'")
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top tokens to collect per position")
    parser.add_argument("--max_samples_per_vertex", type=int, default=200,
                        help="Max samples per vertex (0 = no limit)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for model forward passes")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine clusters
    if args.clusters == "all":
        sample_dir = Path(args.prepared_samples_dir)
        clusters = [p.stem.replace("cluster_", "") for p in sorted(sample_dir.glob("cluster_*.json"))]
    else:
        clusters = [c.strip() for c in args.clusters.split(",")]

    # Load model
    print(f"Loading model: {args.model_name}")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        dtype=torch.float16,
    )
    tokenizer = model.tokenizer
    print(f"  Model loaded. Vocab size: {model.cfg.d_vocab}")

    max_samples = args.max_samples_per_vertex if args.max_samples_per_vertex > 0 else None

    print("\n" + "=" * 80)
    print("TOP PREDICTED TOKENS BY VERTEX")
    print("=" * 80)

    all_results = {}

    for cluster_key in clusters:
        sample_path = Path(args.prepared_samples_dir) / f"cluster_{cluster_key}.json"
        if not sample_path.exists():
            print(f"\nSkipping {cluster_key}: file not found at {sample_path}")
            continue

        description = CLUSTER_DESCRIPTIONS.get(cluster_key, f"Cluster {cluster_key}")

        print(f"\n{'=' * 80}")
        print(f"CLUSTER {cluster_key}: {description}")
        print(f"{'=' * 80}")

        # Load samples
        samples_by_vertex, metadata = load_vertex_samples(sample_path)
        print(f"  Loaded {sum(len(s) for s in samples_by_vertex.values())} samples "
              f"across {len(samples_by_vertex)} vertices (k={metadata['k']})")

        # Collect predictions
        results_by_vertex = collect_predictions(
            samples_by_vertex, model, tokenizer,
            top_k=args.top_k,
            max_samples_per_vertex=max_samples,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Aggregate
        aggregated = aggregate_by_vertex(results_by_vertex, args.top_k)

        # Print report
        print_report(cluster_key, description, aggregated)

        # Build JSON results
        cluster_results = build_json_results(
            cluster_key, metadata, aggregated, results_by_vertex, args.top_k
        )
        all_results[cluster_key] = cluster_results

        # Save per-cluster JSON
        cluster_output_path = output_dir / f"top_tokens_{cluster_key}.json"
        with open(cluster_output_path, "w") as f:
            json.dump(cluster_results, f, indent=2)
        print(f"\n  Saved: {cluster_output_path}")

    # Save combined results
    combined_path = output_dir / "top_tokens_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {combined_path}")

    # Print compact summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Cluster':<12} {'Vertex':>7} {'Samples':>8} {'Avg P(top1)':>12} {'Top-1 most common':>30}")
    print("-" * 75)
    for cluster_key, cr in all_results.items():
        for vid_str in sorted(cr["vertices"].keys(), key=int):
            v = cr["vertices"][vid_str]
            top1_str = ""
            if v["top1_distribution"]:
                top3 = v["top1_distribution"][:3]
                top1_str = ", ".join(f"{repr(t['token'])}({t['pct']:.0f}%)" for t in top3)
            print(f"{cluster_key:<12} V{vid_str:>5} {v['n_samples']:>8} "
                  f"{v['avg_top1_prob']:>12.3f} {top1_str:>30}")


if __name__ == "__main__":
    main()
