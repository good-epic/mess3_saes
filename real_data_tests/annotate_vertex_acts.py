#!/usr/bin/env python3
"""
Annotate existing vertex_samples.jsonl with latent_acts and barycentric_coords.

Post-processing pass over vertex_samples.jsonl files (in selected_clusters_broad_2/ or
selected_null_clusters/). Re-runs each saved 256-token chunk through model + SAE + AANet
at each trigger position and appends two new parallel-to-trigger fields:

  latent_acts      list[list[float]]  cluster activation vector, one per trigger
  barycentric_coords  list[list[float]]  full k-dim simplex coords, one per trigger

Output: vertex_samples_with_acts.jsonl alongside the originals.

Usage:
    python real_data_tests/annotate_vertex_acts.py \\
        --source_dir /workspace/outputs/selected_clusters_broad_2 \\
        --csv_dir /workspace/outputs/real_data_analysis_canonical \\
        --output_dir /workspace/outputs/selected_clusters_broad_2 \\
        --clusters 512_17,512_22,768_140 \\
        --model_name gemma-2-9b \\
        --sae_release gemma-scope-9b-pt-res \\
        --sae_id layer_20/width_16k/average_l0_68 \\
        --device cuda \\
        --cache_dir /workspace/hf_cache \\
        --hf_token $HF_TOKEN
"""

import os
import sys
import ast
import re
import glob
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import login

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "AAnet"))

from mess3_gmg_analysis_utils import sae_encode_features


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate vertex_samples.jsonl with latent_acts + barycentric_coords"
    )
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Dir with n{N}/cluster_*_vertex_samples.jsonl")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Cluster CSVs dir (latent_indices + AANet model path)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write _with_acts.jsonl (default: same as source_dir)")
    parser.add_argument("--clusters", type=str, default=None,
                        help="Comma-separated cluster keys e.g. 512_17,768_140 "
                             "(default: process all found in source_dir)")
    parser.add_argument("--max_samples_per_vertex", type=int, default=None,
                        help="Cap per-vertex samples to avoid huge clusters (e.g. 500)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/average_l0_68")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--aanet_layer_widths", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--aanet_simplex_scale", type=float, default=1.0)
    parser.add_argument("--aanet_noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =============================================================================
# File-finding helpers (same patterns as collect_simplex_samples.py / causal_steering.py)
# =============================================================================

def find_vertex_samples_path(source_dir, n_clusters, cluster_id, k):
    pattern = str(Path(source_dir) / f"n{n_clusters}"
                  / f"cluster_{cluster_id}_k{k}_category*_vertex_samples.jsonl")
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def find_model_path(csv_dir, n_clusters, cluster_id, k):
    path = Path(csv_dir) / f"clusters_{n_clusters}" / f"aanet_cluster_{cluster_id}_k{k}.pt"
    return path if path.exists() else None


def get_k_from_source_dir(source_dir, n_clusters, cluster_id):
    pattern = str(Path(source_dir) / f"n{n_clusters}"
                  / f"cluster_{cluster_id}_k*_category*_vertex_samples.jsonl")
    matches = glob.glob(pattern)
    if not matches:
        return None
    m = re.search(r"_k(\d+)_", Path(matches[0]).name)
    return int(m.group(1)) if m else None


def get_latent_indices(csv_dir, n_clusters, cluster_id):
    import pandas as pd
    csv_path = Path(csv_dir) / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    rows = df[df["cluster_id"] == cluster_id]
    if rows.empty:
        return None
    return ast.literal_eval(rows.iloc[0]["latent_indices"])


def discover_clusters(source_dir):
    """Return list of (n_clusters, cluster_id) for all vertex_samples.jsonl found."""
    clusters = []
    for path in sorted(Path(source_dir).glob("n*/cluster_*_vertex_samples.jsonl")):
        # n512/cluster_17_k3_categoryM_vertex_samples.jsonl
        n_clusters = int(path.parent.name[1:])
        m = re.match(r"cluster_(\d+)_k\d+_", path.name)
        if m:
            clusters.append((n_clusters, int(m.group(1))))
    return clusters


# =============================================================================
# Core annotation function
# =============================================================================

def annotate_cluster(
    source_dir, csv_dir, output_dir,
    n_clusters, cluster_id, k, latent_indices,
    model, sae, hook_name,
    batch_size, max_samples_per_vertex, device, args,
):
    from AAnet.AAnet_torch.models.AAnet_vanilla import AAnet_vanilla

    cluster_key = f"{n_clusters}_{cluster_id}"
    print(f"\n{'=' * 60}")
    print(f"Cluster {cluster_key}  k={k}  n_latents={len(latent_indices)}")

    # Find input file
    vertex_samples_path = find_vertex_samples_path(source_dir, n_clusters, cluster_id, k)
    if vertex_samples_path is None:
        print(f"  No vertex_samples.jsonl found, skipping")
        return

    # Determine output filename (preserve original category tag)
    m_cat = re.search(r"_(category\w+)_vertex_samples", vertex_samples_path.name)
    category_tag = m_cat.group(1) if m_cat else "categoryM"
    out_dir = Path(output_dir) / f"n{n_clusters}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cluster_{cluster_id}_k{k}_{category_tag}_vertex_samples_with_acts.jsonl"

    # Load all records
    with open(vertex_samples_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(records)} records from {vertex_samples_path.name}")

    # Optional per-vertex cap
    if max_samples_per_vertex is not None:
        per_vertex = defaultdict(list)
        for r in records:
            per_vertex[r["vertex_id"]].append(r)
        rng = random.Random(args.seed)
        capped = []
        for v, recs in sorted(per_vertex.items()):
            if len(recs) > max_samples_per_vertex:
                recs = rng.sample(recs, max_samples_per_vertex)
            capped.extend(recs)
        print(f"  After cap ({max_samples_per_vertex}/vertex): {len(capped)} records")
        records = capped

    # Load AANet
    model_path = find_model_path(csv_dir, n_clusters, cluster_id, k)
    if model_path is None:
        print(f"  AANet model not found in {csv_dir}, skipping")
        return
    d_model = sae.W_dec.shape[1]
    aanet = AAnet_vanilla(
        input_shape=d_model,
        n_archetypes=k,
        noise=args.aanet_noise,
        layer_widths=args.aanet_layer_widths,
        activation_out="tanh",
        simplex_scale=args.aanet_simplex_scale,
        device=device,
    )
    aanet.load_state_dict(torch.load(model_path, map_location=device))
    aanet.eval()

    cluster_indices_tensor = torch.tensor(latent_indices, device=device, dtype=torch.long)
    W_c = sae.W_dec[cluster_indices_tensor, :]  # (n_latents, d_model)

    augmented = []
    n_batches = (len(records) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size),
                      total=n_batches, desc=f"  {cluster_key}"):
            batch = records[i : i + batch_size]

            # Stack chunk tokens — all are exactly activity_seq_len (256) tokens
            token_batch = torch.tensor(
                [r["chunk_token_ids"] for r in batch],
                device=device, dtype=torch.long,
            )  # (B, seq_len)

            _, cache = model.run_with_cache(
                token_batch, return_type=None, names_filter=[hook_name]
            )
            acts = cache[hook_name]  # (B, seq_len, d_model)
            del cache

            for j, record in enumerate(batch):
                latent_acts_per_trigger = []
                bary_per_trigger = []

                for trigger_idx in record["trigger_token_indices"]:
                    resid_t = acts[j, trigger_idx, :].unsqueeze(0)  # (1, d_model)

                    feature_acts, _, _ = sae_encode_features(sae, resid_t)
                    acts_c = feature_acts[:, cluster_indices_tensor]  # (1, n_latents)

                    X_c = acts_c @ W_c  # (1, d_model)
                    _, _, z = aanet(X_c)
                    bary = aanet.euclidean_to_barycentric(z)  # (1, k)

                    latent_acts_per_trigger.append(acts_c[0].cpu().float().tolist())
                    bary_per_trigger.append(bary[0].cpu().float().tolist())

                aug_record = dict(record)
                aug_record["latent_acts"] = latent_acts_per_trigger
                aug_record["barycentric_coords"] = bary_per_trigger
                augmented.append(aug_record)

            del acts

    with open(out_path, "w") as f:
        for rec in augmented:
            f.write(json.dumps(rec) + "\n")

    print(f"  Wrote {len(augmented)} records to {out_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    output_dir = args.output_dir or args.source_dir

    if args.hf_token:
        login(token=args.hf_token)

    print(f"\nLoading model: {args.model_name}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        dtype=torch.float16,
    )
    model.eval()

    print(f"Loading SAE: {args.sae_release} - {args.sae_id}")
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    sae.eval()

    layer_idx = int(args.sae_id.split("/")[0].split("_")[1])
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Build cluster list
    if args.clusters:
        cluster_keys = [c.strip() for c in args.clusters.split(",")]
        clusters = []
        for key in cluster_keys:
            n_str, id_str = key.split("_")
            clusters.append((int(n_str), int(id_str)))
    else:
        clusters = discover_clusters(args.source_dir)
        print(f"Auto-discovered {len(clusters)} clusters in {args.source_dir}")

    print(f"\nProcessing {len(clusters)} clusters")

    for n_clusters, cluster_id in clusters:
        k = get_k_from_source_dir(args.source_dir, n_clusters, cluster_id)
        if k is None:
            print(f"\nWARNING: Cannot determine k for {n_clusters}_{cluster_id}, skipping")
            continue

        latent_indices = get_latent_indices(args.csv_dir, n_clusters, cluster_id)
        if latent_indices is None:
            print(f"\nWARNING: Cannot find latent_indices for {n_clusters}_{cluster_id}, skipping")
            continue

        annotate_cluster(
            source_dir=args.source_dir,
            csv_dir=args.csv_dir,
            output_dir=output_dir,
            n_clusters=n_clusters,
            cluster_id=cluster_id,
            k=k,
            latent_indices=latent_indices,
            model=model,
            sae=sae,
            hook_name=hook_name,
            batch_size=args.batch_size,
            max_samples_per_vertex=args.max_samples_per_vertex,
            device=args.device,
            args=args,
        )

    print("\n\nDone.")


if __name__ == "__main__":
    main()
