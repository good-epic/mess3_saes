#!/usr/bin/env python3
"""
Collect random simplex samples for KL divergence analysis (2A).

Loads pre-trained AANet models and streams data through model + SAE + AANet,
collecting a stratified random sample of examples distributed across the simplex
(not filtered by vertex proximity). Position distribution is matched to that of
the cluster's near-vertex samples (decile buckets), so token-position bias is
controlled for.

Also supports collecting random control clusters (no prior selection) for
comparison against priority clusters.

Usage:
    python real_data_tests/collect_simplex_samples.py \\
        --n_clusters_list "512,768" \\
        --source_dir /workspace/outputs/selected_clusters_canonical \\
        --csv_dir /workspace/outputs/real_data_analysis_canonical \\
        --save_dir /workspace/outputs/simplex_samples \\
        --manual_cluster_ids "512:464,504,292;768:484" \\
        --manual_k "512:464=3,504=4,292=3;768:484=3" \\
        --n_random_controls 10 \\
        --n_simplex_samples 5000 \\
        --skip_docs 2000000 \\
        --model_name gemma-2-9b \\
        --sae_release gemma-scope-9b-pt-res \\
        --sae_id layer_20/width_16k/average_l0_68 \\
        --device cuda \\
        --cache_dir /workspace/hf_cache \\
        --hf_token $HF_TOKEN
"""

import os
import sys
import argparse
import json
import glob
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import login

sys.path.insert(0, str(Path(__file__).parent.parent))

from real_data_utils import RealDataSampler
from mess3_gmg_analysis_utils import sae_encode_features


# =============================================================================
# Argument parsing (mirrors refit_selected_clusters.py conventions)
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect random simplex samples for KL divergence analysis"
    )

    # Cluster selection
    parser.add_argument("--n_clusters_list", type=str, required=True,
                        help="Comma-separated list of n_clusters values (e.g., '512,768')")
    parser.add_argument("--manual_cluster_ids", type=str, default=None,
                        help="Priority clusters. Format: 'n_clusters1:id1,id2;n_clusters2:id3'")
    parser.add_argument("--manual_k", type=str, default=None,
                        help="k overrides. Format: 'n_clusters1:id1=k1,id2=k2;n_clusters2:id3=k3'")
    parser.add_argument("--n_random_controls", type=int, default=0,
                        help="Number of additional random non-priority clusters per n_clusters value")
    parser.add_argument("--controls_seed", type=int, default=99,
                        help="Seed for random control cluster selection")

    # Directories
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Dir containing refitted .pt models and vertex_samples.jsonl. "
                             "Structure: source_dir/n{N}/cluster_{id}_k{k}_category*.pt")
    parser.add_argument("--csv_dir", type=str,
                        default="outputs/real_data_analysis_canonical",
                        help="Dir containing CSV files. Also used to find Stage 1 models for controls: "
                             "csv_dir/clusters_{N}/aanet_cluster_{id}_k{k}.pt")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory. Structure: save_dir/n{N}/")

    # Collection parameters
    parser.add_argument("--n_simplex_samples", type=int, default=5000,
                        help="Target number of simplex samples per cluster")
    parser.add_argument("--n_position_buckets", type=int, default=10,
                        help="Number of decile buckets for token-position distribution matching")
    parser.add_argument("--skip_docs", type=int, default=2000000,
                        help="Skip this many docs before sampling (to avoid docs used for vertex collection)")
    parser.add_argument("--search_batch_size", type=int, default=32,
                        help="Batch size for streaming through model + SAE + AANet")
    parser.add_argument("--max_inputs_per_cluster", type=int, default=2000000,
                        help="Hard cap on inputs processed per cluster")

    # AANet architecture (must match training)
    parser.add_argument("--aanet_layer_widths", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--aanet_simplex_scale", type=float, default=1.0)
    parser.add_argument("--aanet_noise", type=float, default=0.05)

    # Model & SAE
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/average_l0_68")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="/workspace/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)

    # Data streaming
    parser.add_argument("--hf_dataset", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--hf_subset_name", type=str, default="sample-10BT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--activity_seq_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aanet_prefetch_size", type=int, default=1024)
    parser.add_argument("--shuffle_buffer_size", type=int, default=50000)
    parser.add_argument("--max_doc_tokens", type=int, default=3000)

    return parser.parse_args()


# =============================================================================
# Helpers for parsing manual cluster specs (from refit_selected_clusters.py)
# =============================================================================

def parse_manual_cluster_ids(manual_str):
    if not manual_str:
        return {}
    result = {}
    for group in manual_str.split(';'):
        group = group.strip()
        if not group:
            continue
        n_clusters_str, ids_str = group.split(':')
        result[int(n_clusters_str.strip())] = [int(x.strip()) for x in ids_str.split(',')]
    return result


def parse_manual_k(manual_str):
    if not manual_str:
        return {}
    result = {}
    for group in manual_str.split(';'):
        group = group.strip()
        if not group:
            continue
        n_clusters_str, assignments_str = group.split(':')
        n_clusters = int(n_clusters_str.strip())
        cluster_k_map = {}
        for assignment in assignments_str.split(','):
            assignment = assignment.strip()
            if not assignment:
                continue
            cluster_id_str, k_str = assignment.split('=')
            cluster_k_map[int(cluster_id_str.strip())] = int(k_str.strip())
        result[n_clusters] = cluster_k_map
    return result


# =============================================================================
# Position bucket distribution from near-vertex samples
# =============================================================================

def compute_bucket_targets(vertex_samples_path, n_simplex_samples, n_buckets, seq_len):
    """Compute target sample count per position bucket from near-vertex samples.

    Uses decile buckets of trigger token positions to match the position
    distribution observed in near-vertex examples.

    Returns list of length n_buckets with target counts summing to n_simplex_samples.
    """
    bucket_counts = [0] * n_buckets
    total = 0

    if vertex_samples_path is not None:
        try:
            with open(vertex_samples_path) as f:
                for line in f:
                    sample = json.loads(line)
                    for seq_idx in sample.get("trigger_token_indices", []):
                        bucket = min(int(seq_idx * n_buckets / seq_len), n_buckets - 1)
                        bucket_counts[bucket] += 1
                        total += 1
        except (FileNotFoundError, json.JSONDecodeError):
            total = 0

    if total == 0:
        print("    No near-vertex position data found; using uniform bucket distribution")
        per_bucket = n_simplex_samples // n_buckets
        targets = [per_bucket] * n_buckets
        # Distribute remainder
        for i in range(n_simplex_samples % n_buckets):
            targets[i] += 1
        return targets

    fracs = [c / total for c in bucket_counts]
    targets = [max(0, round(f * n_simplex_samples)) for f in fracs]

    # Ensure sum equals n_simplex_samples exactly
    diff = n_simplex_samples - sum(targets)
    if diff != 0:
        # Add/remove from the largest bucket
        largest_bucket = max(range(n_buckets), key=lambda i: targets[i])
        targets[largest_bucket] += diff

    return targets


# =============================================================================
# Find files for a cluster in source_dir
# =============================================================================

def find_model_path(csv_dir, n_clusters, cluster_id, k):
    """Find AANet .pt model in csv_dir (stage-1 naming: aanet_cluster_{id}_k{k}.pt)."""
    path = Path(csv_dir) / f"clusters_{n_clusters}" / f"aanet_cluster_{cluster_id}_k{k}.pt"
    return path if path.exists() else None


# Alias kept for control-cluster call sites
find_control_model_path = find_model_path


def find_vertex_samples_path(source_dir, n_clusters, cluster_id, k):
    """Find vertex_samples.jsonl for a cluster in source_dir."""
    pattern = str(Path(source_dir) / f"n{n_clusters}" / f"cluster_{cluster_id}_k{k}_category*_vertex_samples.jsonl")
    matches = glob.glob(pattern)
    if matches:
        return Path(matches[0])
    return None


# =============================================================================
# Select random control clusters from CSV
# =============================================================================

def select_random_controls(csv_dir, n_clusters, excluded_ids, n_controls, seed):
    """Randomly sample n_controls cluster IDs from the CSV, excluding priority clusters."""
    import pandas as pd

    csv_path = Path(csv_dir) / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
    if not csv_path.exists():
        print(f"    WARNING: CSV not found at {csv_path}, skipping controls")
        return []

    df = pd.read_csv(csv_path)
    all_ids = sorted(df['cluster_id'].unique().tolist())
    candidates = [cid for cid in all_ids if cid not in excluded_ids]

    if not candidates:
        print(f"    No control candidates available after excluding priority clusters")
        return []

    rng = np.random.RandomState(seed)
    n = min(n_controls, len(candidates))
    selected = sorted(rng.choice(candidates, n, replace=False).tolist())
    print(f"    Selected {n} random control clusters: {selected}")
    return selected


def get_control_k(csv_dir, n_clusters, cluster_id):
    """Get reconstruction elbow k for a control cluster from CSV."""
    import pandas as pd

    csv_path = Path(csv_dir) / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    rows = df[df['cluster_id'] == cluster_id]
    if rows.empty:
        return None
    return int(rows.iloc[0]['aanet_recon_loss_elbow_k'])


def get_latent_indices(csv_dir, n_clusters, cluster_id):
    """Get latent indices for a cluster from CSV."""
    import pandas as pd
    import ast

    csv_path = Path(csv_dir) / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    rows = df[df['cluster_id'] == cluster_id]
    if rows.empty:
        return None
    return ast.literal_eval(rows.iloc[0]['latent_indices'])


# =============================================================================
# Core collection function
# =============================================================================

def collect_for_cluster(
    n_clusters, cluster_id, k, latent_indices,
    model_path, vertex_samples_path,
    save_dir, args, model, sae, sampler, tokenizer,
    hook_name, is_control=False
):
    """Collect n_simplex_samples random simplex examples for one cluster."""
    from AAnet.AAnet_torch.models.AAnet_vanilla import AAnet_vanilla

    cluster_key = f"{n_clusters}_{cluster_id}"
    label = "control" if is_control else "priority"
    print(f"\n{'='*60}")
    print(f"Cluster {cluster_key} (k={k}, {label})")
    print(f"  Model: {model_path}")
    print(f"  Near-vertex samples: {vertex_samples_path}")
    print(f"{'='*60}")

    # Compute position bucket targets
    targets = compute_bucket_targets(
        vertex_samples_path,
        args.n_simplex_samples,
        args.n_position_buckets,
        args.activity_seq_len,
    )
    print(f"  Position bucket targets: {targets}")
    print(f"  Total target: {sum(targets)}")

    # Load AANet
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

    # Output paths
    out_dir = Path(save_dir) / f"n{n_clusters}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"cluster_{cluster_id}_k{k}_simplex_samples.jsonl"
    stats_path = out_dir / f"cluster_{cluster_id}_k{k}_simplex_stats.json"

    # Convert latent indices to tensor
    cluster_indices_tensor = torch.tensor(latent_indices, device=args.device, dtype=torch.long)

    # Stratified bucket tracking
    bucket_counts = [0] * args.n_position_buckets
    total_collected = 0
    total_inputs = 0

    with open(samples_path, 'w') as out_f:
        pbar = tqdm(total=args.n_simplex_samples, desc=f"  {cluster_key}")

        while total_inputs < args.max_inputs_per_cluster:
            if total_collected >= args.n_simplex_samples:
                break
            # Check if all buckets are full
            if all(bucket_counts[b] >= targets[b] for b in range(args.n_position_buckets)):
                break

            tokens = sampler.sample_tokens_batch(
                args.search_batch_size, args.activity_seq_len, args.device
            )

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, return_type=None, names_filter=[hook_name]
                )
                acts = cache[hook_name]
                acts_flat = acts.reshape(-1, acts.shape[-1])

                feature_acts, _, _ = sae_encode_features(sae, acts_flat)
                acts_c = feature_acts[:, cluster_indices_tensor]

                active_mask = (acts_c.abs().sum(dim=1) > 0)
                if not active_mask.any():
                    total_inputs += acts_flat.shape[0]
                    del tokens, cache, acts, acts_flat, feature_acts
                    continue

                acts_c_active = acts_c[active_mask]
                active_indices = torch.where(active_mask)[0]

                W_c = sae.W_dec[cluster_indices_tensor, :]
                X_recon = acts_c_active @ W_c

                _, _, embedding = aanet(X_recon)
                embedding = aanet.euclidean_to_barycentric(embedding)

                # Process each active (batch_idx, seq_idx) position
                acts_c_active_np = acts_c_active.cpu().numpy()
                for local_idx, (idx, bary) in enumerate(zip(active_indices.cpu().numpy(), embedding.cpu().numpy())):
                    batch_idx = int(idx // args.activity_seq_len)
                    seq_idx = int(idx % args.activity_seq_len)

                    # Skip BOS
                    if seq_idx == 0:
                        continue

                    # Determine position bucket
                    bucket = min(
                        int(seq_idx * args.n_position_buckets / args.activity_seq_len),
                        args.n_position_buckets - 1,
                    )

                    # Accept only if bucket not yet full
                    if bucket_counts[bucket] >= targets[bucket]:
                        continue

                    # Decode sample info
                    sequence_tokens = tokens[batch_idx].cpu().numpy()
                    full_text = tokenizer.decode(sequence_tokens, skip_special_tokens=True)

                    trigger_token_id = int(tokens[batch_idx, seq_idx].cpu().item())
                    trigger_word = tokenizer.decode([trigger_token_id], skip_special_tokens=True)

                    text_up_to_token = tokenizer.decode(
                        tokens[batch_idx, :seq_idx + 1].cpu().numpy(),
                        skip_special_tokens=True,
                    )
                    word_index = max(0, len(text_up_to_token.split()) - 1)

                    record = {
                        "barycentric_coords": bary.tolist(),
                        "latent_acts": acts_c_active_np[local_idx].tolist(),
                        "position_bucket": bucket,
                        "full_text": full_text,
                        "chunk_token_ids": sequence_tokens.tolist(),
                        "trigger_token_indices": [seq_idx],
                        "trigger_token_ids": [trigger_token_id],
                        "trigger_word_indices": [word_index],
                        "trigger_words": [trigger_word],
                    }

                    out_f.write(json.dumps(record) + '\n')
                    bucket_counts[bucket] += 1
                    total_collected += 1
                    pbar.update(1)

                    if total_collected >= args.n_simplex_samples:
                        break

                total_inputs += acts_flat.shape[0]
                del tokens, cache, acts, acts_flat, feature_acts

        pbar.close()

    print(f"\n  Collected {total_collected} simplex samples")
    print(f"  Bucket counts: {bucket_counts}")
    print(f"  Total inputs processed: {total_inputs:,}")

    # Save stats
    stats = {
        "cluster_key": cluster_key,
        "n_clusters": n_clusters,
        "cluster_id": cluster_id,
        "k": k,
        "n_latents": len(latent_indices),
        "latent_indices": latent_indices,
        "is_control": is_control,
        "n_collected": total_collected,
        "bucket_counts": bucket_counts,
        "bucket_targets": targets,
        "total_inputs_processed": total_inputs,
        "samples_path": str(samples_path),
        "near_vertex_samples_path": str(vertex_samples_path) if vertex_samples_path else None,
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return total_collected


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    n_clusters_list = [int(x.strip()) for x in args.n_clusters_list.split(',')]
    manual_cluster_ids = parse_manual_cluster_ids(args.manual_cluster_ids)
    manual_k = parse_manual_k(args.manual_k)

    # Login to HuggingFace
    if args.hf_token:
        login(token=args.hf_token)

    # Load model
    print(f"\nLoading Model: {args.model_name}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        dtype=torch.float16,
    )
    model.eval()
    tokenizer = model.tokenizer

    # Load SAE
    print(f"Loading SAE: {args.sae_release} - {args.sae_id}")
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    sae.eval()

    # Hook name
    layer_idx = int(args.sae_id.split("/")[0].split("_")[1])
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Initialize sampler
    sampler = RealDataSampler(
        model,
        hf_dataset=args.hf_dataset,
        hf_subset_name=args.hf_subset_name,
        split=args.dataset_split,
        seed=args.seed,
        prefetch_size=args.aanet_prefetch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_doc_tokens=args.max_doc_tokens,
    )

    # Skip docs to avoid overlap with vertex collection
    if args.skip_docs > 0:
        print(f"\nSkipping {args.skip_docs:,} documents to avoid vertex collection overlap...")
        docs_skipped = 0
        while docs_skipped < args.skip_docs:
            try:
                _ = next(sampler.iterator)
                docs_skipped += 1
                if docs_skipped % 100000 == 0:
                    print(f"  Skipped {docs_skipped:,}/{args.skip_docs:,}...")
            except StopIteration:
                sampler.iterator = iter(sampler.dataset)
                print("  Reached end of dataset, wrapped around")
        print(f"  Done skipping.")

    # Build cluster list
    clusters_to_process = []

    for n_clusters in n_clusters_list:
        # Priority clusters
        priority_ids = manual_cluster_ids.get(n_clusters, [])
        k_overrides = manual_k.get(n_clusters, {})

        for cluster_id in priority_ids:
            k = k_overrides.get(cluster_id)
            if k is None:
                k = get_control_k(args.csv_dir, n_clusters, cluster_id)
            if k is None:
                print(f"WARNING: Cannot determine k for cluster {n_clusters}_{cluster_id}, skipping")
                continue

            model_path = find_model_path(args.csv_dir, n_clusters, cluster_id, k)
            if model_path is None:
                print(f"WARNING: No model found for cluster {n_clusters}_{cluster_id} k={k}, skipping")
                continue

            vertex_samples_path = find_vertex_samples_path(args.source_dir, n_clusters, cluster_id, k)

            latent_indices = get_latent_indices(args.csv_dir, n_clusters, cluster_id)
            if latent_indices is None:
                print(f"WARNING: Cannot find latent indices for {n_clusters}_{cluster_id}, skipping")
                continue

            clusters_to_process.append({
                "n_clusters": n_clusters,
                "cluster_id": cluster_id,
                "k": k,
                "latent_indices": latent_indices,
                "model_path": model_path,
                "vertex_samples_path": vertex_samples_path,
                "is_control": False,
            })

        # Random control clusters
        if args.n_random_controls > 0:
            control_ids = select_random_controls(
                args.csv_dir, n_clusters, set(priority_ids),
                args.n_random_controls, args.controls_seed,
            )
            for cluster_id in control_ids:
                k = get_control_k(args.csv_dir, n_clusters, cluster_id)
                if k is None:
                    print(f"WARNING: Cannot determine k for control {n_clusters}_{cluster_id}, skipping")
                    continue

                model_path = find_control_model_path(args.csv_dir, n_clusters, cluster_id, k)
                if model_path is None:
                    print(f"WARNING: No Stage 1 model for control {n_clusters}_{cluster_id} k={k}, skipping")
                    continue

                latent_indices = get_latent_indices(args.csv_dir, n_clusters, cluster_id)
                if latent_indices is None:
                    print(f"WARNING: Cannot find latent indices for control {n_clusters}_{cluster_id}, skipping")
                    continue

                clusters_to_process.append({
                    "n_clusters": n_clusters,
                    "cluster_id": cluster_id,
                    "k": k,
                    "latent_indices": latent_indices,
                    "model_path": model_path,
                    "vertex_samples_path": None,  # No near-vertex data for controls
                    "is_control": True,
                })

    print(f"\nClusters to process: {len(clusters_to_process)}")
    for c in clusters_to_process:
        label = "control" if c["is_control"] else "priority"
        print(f"  {c['n_clusters']}_{c['cluster_id']} k={c['k']} ({label})")

    # Collect samples for each cluster
    for cluster in clusters_to_process:
        collect_for_cluster(
            n_clusters=cluster["n_clusters"],
            cluster_id=cluster["cluster_id"],
            k=cluster["k"],
            latent_indices=cluster["latent_indices"],
            model_path=cluster["model_path"],
            vertex_samples_path=cluster["vertex_samples_path"],
            save_dir=args.save_dir,
            args=args,
            model=model,
            sae=sae,
            sampler=sampler,
            tokenizer=tokenizer,
            hook_name=hook_name,
            is_control=cluster["is_control"],
        )

    print(f"\n{'='*60}")
    print(f"All done. Results saved to {args.save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
