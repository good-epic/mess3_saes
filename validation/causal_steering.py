#!/usr/bin/env python3
"""
Phase 3a: Causal Steering via AANet Subspace Patching.

For each near-vertex example, subtract the cluster's current AANet reconstruction
from the layer-20 residual at the trigger position, and add back a scaled version
of the target vertex's reconstruction. Record the full next-token distribution and
greedy continuation (steered and unsteered) for downstream analysis.

Usage:
    python validation/causal_steering.py \
        --source_dir /workspace/outputs/selected_clusters_broad_2 \
        --csv_dir /workspace/outputs/real_data_analysis_canonical \
        --output_dir /workspace/outputs/validation/causal_steering \
        --clusters 512_17,512_181,768_140,768_596 \
        --scales 0 1 5 20 \
        --n_gen_tokens 128 \
        --max_examples_per_vertex 100 \
        --model_name gemma-2-9b \
        --sae_release gemma-scope-9b-pt-res \
        --sae_id layer_20/width_16k/average_l0_68 \
        --device cuda \
        --cache_dir /workspace/hf_cache \
        --hf_token $HF_TOKEN
"""

import os
import sys
import argparse
import json
import glob
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import login
from mess3_gmg_analysis_utils import sae_encode_features


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3a: Causal Steering")

    parser.add_argument("--source_dir", type=str, required=True,
                        help="Dir with refitted AANet models and vertex_samples.jsonl. "
                             "Structure: source_dir/n{N}/cluster_{id}_k{k}_category*.pt")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Dir with consolidated_metrics CSV files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save steering results")
    parser.add_argument("--clusters", type=str, required=True,
                        help="Comma-separated cluster keys, e.g. 512_17,768_140")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 5.0, 20.0],
                        help="Steering scales to evaluate")
    parser.add_argument("--n_gen_tokens", type=int, default=128,
                        help="Number of greedy generation tokens")
    parser.add_argument("--max_examples_per_vertex", type=int, default=100,
                        help="Cap on examples per source vertex")

    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/average_l0_68")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="/workspace/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--aanet_layer_widths", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--aanet_simplex_scale", type=float, default=1.0)
    parser.add_argument("--aanet_noise", type=float, default=0.05)
    parser.add_argument("--k_sustain", type=int, default=16,
                        help="Number of generated positions to sustain the delta "
                             "for steering type2 (type3 sustains for all tokens)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Unused; reserved for future batching")

    return parser.parse_args()


# =============================================================================
# File finding helpers (inlined from collect_simplex_samples.py)
# =============================================================================

def find_model_path(source_dir, n_clusters, cluster_id, k):
    """Find refitted .pt model for a cluster in source_dir."""
    pattern = str(
        Path(source_dir) / f"n{n_clusters}" / f"cluster_{cluster_id}_k{k}_category*.pt"
    )
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def find_vertex_samples_path(source_dir, n_clusters, cluster_id, k):
    """Find vertex_samples.jsonl for a cluster in source_dir."""
    pattern = str(
        Path(source_dir) / f"n{n_clusters}"
        / f"cluster_{cluster_id}_k{k}_category*_vertex_samples.jsonl"
    )
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


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


def get_k(csv_dir, n_clusters, cluster_id):
    """Get elbow k for a cluster from the consolidated metrics CSV."""
    import pandas as pd
    csv_path = Path(csv_dir) / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    rows = df[df["cluster_id"] == cluster_id]
    if rows.empty:
        return None
    return int(rows.iloc[0]["aanet_recon_loss_elbow_k"])


# =============================================================================
# Load vertex samples
# =============================================================================

def load_vertex_samples(source_dir, n_clusters, cluster_id, k, max_examples_per_vertex):
    """Load vertex samples, selecting best (min-distance) trigger per record.

    Returns list of dicts: {vertex_id, chunk_token_ids, trigger_idx, distance, full_text}
    """
    path = find_vertex_samples_path(source_dir, n_clusters, cluster_id, k)
    if path is None:
        return []

    raw = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            trigger_indices = record.get("trigger_token_indices", [])
            if not trigger_indices:
                continue

            distances = record.get("distances_to_vertex", [])
            if len(distances) == len(trigger_indices) and distances:
                best_idx = int(np.argmin(distances))
            else:
                best_idx = 0

            raw.append({
                "vertex_id": record["vertex_id"],
                "chunk_token_ids": record["chunk_token_ids"],
                "trigger_idx": trigger_indices[best_idx],
                "distance": distances[best_idx] if distances else None,
                "full_text": record.get("full_text", ""),
            })

    # Group by vertex and cap
    by_vertex = defaultdict(list)
    for s in raw:
        by_vertex[s["vertex_id"]].append(s)

    result = []
    for v_id in sorted(by_vertex.keys()):
        result.extend(by_vertex[v_id][:max_examples_per_vertex])

    return result


# =============================================================================
# Hook factory
# =============================================================================

# Steering types:
#   type1 — patch only position trigger_idx (original approach).
#            At every generation step, delta is added to the cached trigger
#            position; future tokens attend to it via attention.
#   type2 — patch trigger_idx through min(last_pos, trigger_idx + k_sustain).
#            Delta is injected into each newly computed position for the first
#            k_sustain generated tokens, then generation proceeds freely.
#   type3 — patch trigger_idx through last_pos at every step.
#            Delta rides into every generated position throughout the full
#            continuation.

STEERING_TYPES = ("type1", "type2", "type3")


def make_patch_hook(trigger_idx, delta, steering_type="type1", k_sustain=16):
    """Return hook that adds delta to residual at the appropriate position range.

    Args:
        trigger_idx:   position of the trigger token in the context
        delta:         (d_model,) float32 steering vector
        steering_type: one of "type1", "type2", "type3"
        k_sustain:     number of generated positions to sustain for type2
    """
    def hook_fn(act, hook):
        seq_len = act.shape[1]
        if seq_len <= trigger_idx:
            return act
        last_pos = seq_len - 1
        if steering_type == "type1":
            end_pos = trigger_idx
        elif steering_type == "type2":
            end_pos = min(last_pos, trigger_idx + k_sustain)
        else:  # type3
            end_pos = last_pos
        act[:, trigger_idx:end_pos + 1, :] = (
            act[:, trigger_idx:end_pos + 1, :] + delta.to(act.dtype)
        )
        return act
    return hook_fn


# =============================================================================
# Forward pass helpers
# =============================================================================

def run_patched_forward(model, tokens, hook_name, hook_fn, trigger_idx):
    """Run model (optionally with hook) and return log-softmax at trigger_idx.

    Args:
        tokens: (seq_len,) long tensor — will be unsqueezed to (1, seq_len)
        hook_fn: if None, run without hook
        trigger_idx: position to extract logprobs at

    Returns:
        log-probs shape (vocab_size,) float16
    """
    with torch.no_grad():
        if hook_fn is not None:
            logits = model.run_with_hooks(
                tokens.unsqueeze(0),
                fwd_hooks=[(hook_name, hook_fn)],
                return_type="logits",
            )
        else:
            logits = model(tokens.unsqueeze(0))

        lp = F.log_softmax(logits[0, trigger_idx, :].float(), dim=-1)
        return lp.half().cpu()


def greedy_generate(model, hook_name, hook_fn, context_tokens, n_tokens, device):
    """Greedy generation with optional hook applied at each step.

    Args:
        context_tokens: (ctx_len,) long tensor — starting context
        hook_fn: if None, generate without hook
        n_tokens: number of tokens to generate

    Returns:
        list of generated token ids (ints)
    """
    generated = []
    current = context_tokens.clone().to(device)

    with torch.no_grad():
        for _ in range(n_tokens):
            if hook_fn is not None:
                logits = model.run_with_hooks(
                    current.unsqueeze(0),
                    fwd_hooks=[(hook_name, hook_fn)],
                    return_type="logits",
                )
            else:
                logits = model(current.unsqueeze(0))

            next_token = int(logits[0, -1, :].argmax().item())
            generated.append(next_token)
            current = torch.cat([
                current,
                torch.tensor([next_token], device=device, dtype=torch.long),
            ])

    return generated


# =============================================================================
# Steering vector computation
# =============================================================================

def compute_steering_vectors(aanet, acts_c, W_c, k, device):
    """Compute current and target reconstructions in the AANet subspace.

    Args:
        acts_c: (n_latents,) float tensor — SAE activations at trigger position
        W_c: (n_latents, d_model) float tensor — cluster decoder rows
        k: number of archetypes

    Returns:
        X_recon_curr: (d_model,) float32 — current AANet reconstruction
        bary_curr: (k,) float32 — barycentric coordinates of current embedding
        targets: dict {v_target: X_recon_target (d_model,) float32}
    """
    with torch.no_grad():
        X_c = (acts_c.unsqueeze(0).float() @ W_c.float())  # (1, d_model)
        X_recon_curr, _, Z_curr = aanet(X_c)
        X_recon_curr = X_recon_curr.squeeze(0)             # (d_model,)
        Z_curr = Z_curr.squeeze(0)                          # (k-1,)
        bary_curr = aanet.euclidean_to_barycentric(Z_curr.unsqueeze(0)).squeeze(0)  # (k,)

        targets = {}
        for v_target in range(k):
            one_hot = torch.zeros(k, device=device, dtype=torch.float32)
            one_hot[v_target] = 1.0
            Z_target = one_hot @ aanet.archetypal_simplex  # (k-1,)
            X_recon_target = aanet.decode(Z_target.unsqueeze(0)).squeeze(0)  # (d_model,)
            targets[v_target] = X_recon_target

    return X_recon_curr, bary_curr, targets


# =============================================================================
# Per-example processing
# =============================================================================

def process_example(
    example, ex_idx, k, scales, model, sae, aanet, hook_name, W_c,
    cluster_indices_tensor, n_gen_tokens, tokenizer, device, cluster_key,
    k_sustain=16,
):
    """Process one vertex sample: compute steering deltas and run patched forwards.

    Runs generation for all three steering types per (target_vertex, scale).

    Returns:
        records: list of (record_id, result_dict) — one per target vertex
        logprobs_dict: dict of {key: np.array float16}
        pre_key: key for pre-patch logprobs in logprobs_dict
    """
    trigger_idx = example["trigger_idx"]
    source_vertex = example["vertex_id"]
    chunk_token_ids = example["chunk_token_ids"]
    chunk_tokens = torch.tensor(chunk_token_ids, dtype=torch.long, device=device)

    # --- Get SAE activations at trigger position ---
    with torch.no_grad():
        _, cache = model.run_with_cache(
            chunk_tokens.unsqueeze(0),
            return_type=None,
            names_filter=[hook_name],
        )
        resid_T = cache[hook_name][0, trigger_idx, :]        # (d_model,)
        del cache

        feature_acts, _, _ = sae_encode_features(sae, resid_T.unsqueeze(0))
        acts_c = feature_acts[0, cluster_indices_tensor]      # (n_latents,)

    # --- Compute steering vectors ---
    X_recon_curr, bary_curr, target_recons = compute_steering_vectors(
        aanet, acts_c, W_c, k, device
    )

    # --- Pre-patch logprobs (no hook, full context) ---
    pre_lp = run_patched_forward(model, chunk_tokens, hook_name, None, trigger_idx)
    pre_key = f"ex{ex_idx:04d}_V{source_vertex}_pre"

    # --- Unsteered generation (context = tokens up to and including trigger) ---
    context_tokens = chunk_tokens[:trigger_idx + 1]
    unsteered_ids = greedy_generate(
        model, hook_name, None, context_tokens, n_gen_tokens, device
    )
    unsteered_continuation = tokenizer.decode(unsteered_ids, skip_special_tokens=True)

    # Document continuation (original tokens after trigger within the 256-token window)
    doc_cont_ids = chunk_token_ids[trigger_idx + 1:]
    document_continuation = tokenizer.decode(doc_cont_ids, skip_special_tokens=True)

    # Pre-trigger text and trigger word (for autointerp prompt construction)
    trigger_word = tokenizer.decode([chunk_token_ids[trigger_idx]], skip_special_tokens=True)
    pre_trigger_text = tokenizer.decode(chunk_token_ids[:trigger_idx + 1], skip_special_tokens=True)

    # --- Process each target vertex ---
    records = []
    logprobs_dict = {pre_key: pre_lp.numpy()}

    for v_target in range(k):
        if v_target == source_vertex:
            continue

        record_id = f"{cluster_key}_ex{ex_idx:04d}_V{source_vertex}toV{v_target}"
        X_recon_target = target_recons[v_target]

        # steered_continuations: {steering_type: {scale_str: text}}
        steered_continuations = {stype: {} for stype in STEERING_TYPES}

        for scale in scales:
            # Delta: remove current reconstruction, add scaled target
            delta = -X_recon_curr + scale * X_recon_target  # (d_model,) float32

            # Post-patch logprobs use type1 (patch T only in the static context;
            # generation type doesn't apply here).
            hook_fn_t1 = make_patch_hook(trigger_idx, delta, "type1")
            post_lp = run_patched_forward(
                model, chunk_tokens, hook_name, hook_fn_t1, trigger_idx
            )
            scale_key = f"{record_id}_s{scale:g}"
            logprobs_dict[scale_key] = post_lp.numpy()

            # Generation for all three steering types
            for stype in STEERING_TYPES:
                hook_fn = make_patch_hook(trigger_idx, delta, stype, k_sustain)
                gen_ids = greedy_generate(
                    model, hook_name, hook_fn, context_tokens, n_gen_tokens, device
                )
                steered_continuations[stype][str(scale)] = tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                )

        result = {
            "record_id": record_id,
            "cluster_key": cluster_key,
            "source_vertex": source_vertex,
            "target_vertex": v_target,
            "trigger_token_index": trigger_idx,
            "trigger_word": trigger_word,
            "pre_trigger_text": pre_trigger_text,
            "source_barycentric_coords": bary_curr.tolist(),
            "original_text": example["full_text"],
            "document_continuation": document_continuation,
            "unsteered_continuation": unsteered_continuation,
            "steered_continuations": steered_continuations,
            "k_sustain": k_sustain,
        }
        records.append((record_id, result))

    return records, logprobs_dict, pre_key


# =============================================================================
# Cluster runner
# =============================================================================

def run_cluster(
    cluster_key, source_dir, csv_dir, output_dir, scales, n_gen_tokens,
    max_examples_per_vertex, model, sae, hook_name, tokenizer, args,
    k_sustain=16,
):
    """Run causal steering for one cluster and save results."""
    from AAnet.AAnet_torch.models.AAnet_vanilla import AAnet_vanilla

    n_clusters_str, cluster_id_str = cluster_key.split("_")
    n_clusters = int(n_clusters_str)
    cluster_id = int(cluster_id_str)

    print(f"\n{'=' * 60}")
    print(f"Cluster {cluster_key}")

    # Get k and latent indices from CSV
    k = get_k(csv_dir, n_clusters, cluster_id)
    if k is None:
        print(f"  Cannot determine k, skipping")
        return

    latent_indices = get_latent_indices(csv_dir, n_clusters, cluster_id)
    if latent_indices is None:
        print(f"  Cannot determine latent_indices, skipping")
        return

    print(f"  k={k}, n_latents={len(latent_indices)}")

    # Find and load AANet
    model_path = find_model_path(source_dir, n_clusters, cluster_id, k)
    if model_path is None:
        print(f"  AANet model not found in {source_dir}, skipping")
        return

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
    print(f"  AANet loaded from {model_path}")

    # Cluster indices and W_c
    cluster_indices_tensor = torch.tensor(latent_indices, device=args.device, dtype=torch.long)
    W_c = sae.W_dec[cluster_indices_tensor, :]  # (n_latents, d_model)

    # Load vertex samples
    examples = load_vertex_samples(
        source_dir, n_clusters, cluster_id, k, max_examples_per_vertex
    )
    print(f"  Loaded {len(examples)} examples "
          f"(max {max_examples_per_vertex} per vertex, {k} vertices)")

    if not examples:
        print(f"  No examples found, skipping")
        return

    # Output files
    out_dir = Path(output_dir) / f"cluster_{cluster_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "steering_results.jsonl"
    logprobs_path = out_dir / "logprobs.npz"

    all_logprobs = {}
    n_records = 0
    n_errors = 0

    with open(results_path, "w") as f_out:
        for ex_idx, example in enumerate(tqdm(examples, desc=f"  {cluster_key}")):
            try:
                records, logprobs_dict, pre_key = process_example(
                    example, ex_idx, k, scales, model, sae, aanet, hook_name, W_c,
                    cluster_indices_tensor, n_gen_tokens, tokenizer, args.device, cluster_key,
                    k_sustain=k_sustain,
                )
                for record_id, result in records:
                    f_out.write(json.dumps(result) + "\n")
                    n_records += 1

                all_logprobs.update(logprobs_dict)

            except Exception as e:
                import traceback
                print(f"\n  ERROR at example {ex_idx}: {e}")
                traceback.print_exc()
                n_errors += 1
                continue

    # Save logprobs
    if all_logprobs:
        np.savez(str(logprobs_path), **all_logprobs)

    print(f"\n  Written {n_records} records ({n_errors} errors)")
    print(f"  steering_results.jsonl → {results_path}")
    print(f"  logprobs.npz           → {logprobs_path}")
    print(f"  logprobs keys: {len(all_logprobs)}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    cluster_keys = [c.strip() for c in args.clusters.split(",") if c.strip()]

    if args.hf_token:
        login(token=args.hf_token)

    print(f"\nLoading model: {args.model_name}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        dtype=torch.float16,
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
    print(f"Hook name: {hook_name}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nClusters: {cluster_keys}")
    print(f"Scales:   {args.scales}")
    print(f"Steering types: {STEERING_TYPES}")
    print(f"k_sustain (type2): {args.k_sustain}")
    print(f"Gen tokens: {args.n_gen_tokens}")
    print(f"Max examples/vertex: {args.max_examples_per_vertex}")

    for cluster_key in cluster_keys:
        run_cluster(
            cluster_key=cluster_key,
            source_dir=args.source_dir,
            csv_dir=args.csv_dir,
            output_dir=args.output_dir,
            scales=args.scales,
            n_gen_tokens=args.n_gen_tokens,
            max_examples_per_vertex=args.max_examples_per_vertex,
            model=model,
            sae=sae,
            hook_name=hook_name,
            tokenizer=tokenizer,
            args=args,
            k_sustain=args.k_sustain,
        )

    print(f"\n{'=' * 60}")
    print(f"Done. Results in: {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
