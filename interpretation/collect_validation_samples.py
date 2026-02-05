#!/usr/bin/env python3
"""
Collect samples across the simplex for validation experiments.

Unlike vertex sample collection which filters for near-vertex samples,
this collects samples uniformly across the simplex (no distance threshold).

Each sample includes:
- Full text
- Trigger word and position
- Barycentric coordinates in the simplex
- Everything needed for next-token probability analysis

Usage:
    python collect_validation_samples.py \
        --n_clusters 512 \
        --cluster_id 261 \
        --aanet_checkpoint path/to/aanet.pt \
        --output_dir outputs/validation_samples \
        --target_samples 10000 \
        --skip_docs 400000 \
        --model_name gemma-2-9b \
        --device cuda
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import login
from real_data_utils import RealDataSampler
from mess3_gmg_analysis_utils import sae_encode_features


def parse_args():
    parser = argparse.ArgumentParser(description="Collect validation samples across simplex")

    # Cluster specification
    parser.add_argument("--n_clusters", type=int, required=True,
                       help="Number of clusters (for path construction)")
    parser.add_argument("--cluster_id", type=int, required=True,
                       help="Cluster ID")
    parser.add_argument("--aanet_checkpoint", type=str, required=True,
                       help="Path to AANet checkpoint (.pt file)")
    parser.add_argument("--k", type=int, default=None,
                       help="Number of archetypes (if not specified, inferred from checkpoint)")
    parser.add_argument("--latent_indices", type=str, default=None,
                       help="Comma-separated latent indices, or path to JSON file with cluster info")
    parser.add_argument("--cluster_manifest", type=str, default=None,
                       help="Path to manifest.json to get cluster info (alternative to --latent_indices)")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save validation samples")

    # Collection parameters
    parser.add_argument("--target_samples", type=int, default=10000,
                       help="Target number of samples to collect")
    parser.add_argument("--skip_docs", type=int, default=0,
                       help="Skip this many documents before collection (to avoid overlap with training data)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--seq_len", type=int, default=256,
                       help="Sequence length")
    parser.add_argument("--save_interval", type=int, default=1000,
                       help="Save checkpoint every N samples")

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefetch_size", type=int, default=1024)
    parser.add_argument("--shuffle_buffer_size", type=int, default=50000)
    parser.add_argument("--max_doc_tokens", type=int, default=3000)

    # AANet architecture (must match checkpoint)
    parser.add_argument("--aanet_layer_widths", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--aanet_simplex_scale", type=float, default=1.0)
    parser.add_argument("--aanet_noise", type=float, default=0.05)

    args = parser.parse_args()
    return args


def load_cluster_info(args):
    """Load cluster info (latent indices) from manifest or direct specification."""
    if args.latent_indices:
        if args.latent_indices.endswith('.json'):
            # Load from JSON file
            with open(args.latent_indices) as f:
                data = json.load(f)
                return data['latent_indices']
        else:
            # Parse comma-separated list
            return [int(x.strip()) for x in args.latent_indices.split(',')]

    elif args.cluster_manifest:
        # Load from manifest
        with open(args.cluster_manifest) as f:
            manifest = json.load(f)

        # Find the right cluster
        for cluster in manifest['clusters']:
            if cluster['n_clusters'] == args.n_clusters and cluster['cluster_id'] == args.cluster_id:
                return cluster['latent_indices']

        raise ValueError(f"Cluster {args.cluster_id} (n={args.n_clusters}) not found in manifest")

    else:
        raise ValueError("Must specify either --latent_indices or --cluster_manifest")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_key = f"{args.n_clusters}_{args.cluster_id}"
    samples_path = output_dir / f"validation_samples_{cluster_key}.jsonl"
    stats_path = output_dir / f"validation_stats_{cluster_key}.json"

    print("="*80)
    print("VALIDATION SAMPLE COLLECTION")
    print("="*80)
    print(f"Cluster: {cluster_key}")
    print(f"Target samples: {args.target_samples}")
    print(f"Skip docs: {args.skip_docs}")
    print(f"Output: {samples_path}")
    print("="*80)

    # HF login
    if args.hf_token:
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    # Load cluster info
    print("\nLoading cluster info...")
    latent_indices = load_cluster_info(args)
    print(f"  Cluster has {len(latent_indices)} latents")

    # Load model
    print("\nLoading model...")
    model_kwargs = {}
    if args.cache_dir:
        model_kwargs['cache_dir'] = args.cache_dir

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        center_unembed=False,
        center_writing_weights=False,
        dtype="bfloat16",
        **model_kwargs
    )
    model.eval()
    tokenizer = model.tokenizer

    # Load SAE
    print("Loading SAE...")
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device
    )
    sae.eval()

    # Load AANet
    print("Loading AANet...")
    from AAnet.AAnet_torch.models.AAnet_vanilla import AAnet_vanilla

    d_model = sae.W_dec.shape[1]

    # Infer k from checkpoint if not specified
    checkpoint = torch.load(args.aanet_checkpoint, map_location=args.device)

    if args.k is None:
        # Infer k from encoder bottleneck dimension
        # The last encoder layer maps to (k-1) Euclidean dimensions
        encoder_keys = sorted([k for k in checkpoint.keys() if k.startswith('encoder_layers.') and k.endswith('.weight')])
        bottleneck_key = encoder_keys[-1]  # Last encoder layer
        bottleneck_dim = checkpoint[bottleneck_key].shape[0]
        k = bottleneck_dim + 1  # k-1 dims -> k vertices
        print(f"  Inferred k={k} from checkpoint (bottleneck dim={bottleneck_dim})")
    else:
        k = args.k

    aanet = AAnet_vanilla(
        input_shape=d_model,
        n_archetypes=k,
        noise=args.aanet_noise,
        layer_widths=args.aanet_layer_widths,
        activation_out="tanh",
        simplex_scale=args.aanet_simplex_scale,
        device=args.device
    )
    aanet.load_state_dict(checkpoint)
    aanet.eval()
    print(f"  Loaded AANet with k={k}")

    # Get hook name
    layer_idx = int(args.sae_id.split("/")[0].split("_")[1])
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Convert latent indices to tensor
    cluster_indices = torch.tensor(latent_indices, device=args.device, dtype=torch.long)

    # Create sampler
    print("\nInitializing data sampler...")
    import random
    random.seed(args.seed)

    sampler = RealDataSampler(
        model,
        hf_dataset=args.hf_dataset,
        hf_subset_name=args.hf_subset_name,
        split=args.dataset_split,
        seed=args.seed,
        prefetch_size=args.prefetch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_doc_tokens=args.max_doc_tokens
    )

    # Skip documents
    if args.skip_docs > 0:
        print(f"\nSkipping {args.skip_docs} documents...")
        docs_skipped = 0
        while docs_skipped < args.skip_docs:
            try:
                _ = next(sampler.iterator)
                docs_skipped += 1
                if docs_skipped % 10000 == 0:
                    print(f"  Skipped {docs_skipped:,}/{args.skip_docs:,}...")
            except StopIteration:
                sampler.iterator = iter(sampler.dataset)
                print(f"  Reached end of dataset, wrapped around")
        print(f"  Done skipping {docs_skipped:,} documents")

    # Collection loop
    print(f"\nCollecting samples...")
    samples_collected = 0
    total_inputs_processed = 0

    # Open output file
    samples_file = open(samples_path, 'w')

    try:
        pbar = tqdm(total=args.target_samples, desc="Collecting")

        while samples_collected < args.target_samples:
            # Sample batch
            tokens = sampler.sample_tokens_batch(args.batch_size, args.seq_len, args.device)

            with torch.no_grad():
                # Forward through model
                _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                acts = cache[hook_name]
                acts_flat = acts.reshape(-1, acts.shape[-1])

                # Encode with SAE
                feature_acts, _, _ = sae_encode_features(sae, acts_flat)

                # Get cluster-specific activations
                acts_c = feature_acts[:, cluster_indices]

                # Filter for active samples (cluster must be active)
                active_mask = (acts_c.abs().sum(dim=1) > 0)

                if not active_mask.any():
                    total_inputs_processed += acts_flat.shape[0]
                    continue

                acts_c_active = acts_c[active_mask]
                active_indices = torch.where(active_mask)[0]

                # Compute cluster-specific reconstruction
                W_c = sae.W_dec[cluster_indices, :]
                X_recon = acts_c_active @ W_c

                # Forward through AANet to get barycentric coords
                _, _, embedding = aanet(X_recon)
                barycentric = aanet.euclidean_to_barycentric(embedding)  # Shape: (batch, k)

                # Compute distances to each vertex
                vertices = torch.eye(k, device=args.device)
                distances_to_vertices = torch.cdist(barycentric, vertices)  # Shape: (batch, k)

                # Process each active sample
                for i, (idx, bary, dists) in enumerate(zip(
                    active_indices.cpu().numpy(),
                    barycentric.cpu().numpy(),
                    distances_to_vertices.cpu().numpy()
                )):
                    batch_idx = idx // args.seq_len
                    seq_idx = idx % args.seq_len

                    # Skip position 0 (BOS for first chunks, unreliable for non-first)
                    if seq_idx == 0:
                        continue

                    # Get chunk tokens and decoded text
                    sequence_tokens = tokens[batch_idx].cpu().numpy()
                    full_text = tokenizer.decode(sequence_tokens, skip_special_tokens=True)

                    # Get trigger word
                    trigger_token_id = tokens[batch_idx, seq_idx].cpu().item()
                    trigger_word = tokenizer.decode([trigger_token_id], skip_special_tokens=True)

                    # Compute word index
                    text_up_to_token = tokenizer.decode(
                        tokens[batch_idx, :seq_idx+1].cpu().numpy(),
                        skip_special_tokens=True
                    )
                    word_index = len(text_up_to_token.split()) - 1
                    if word_index < 0:
                        word_index = 0

                    # Find nearest vertex
                    nearest_vertex = int(np.argmin(dists))
                    distance_to_nearest = float(dists[nearest_vertex])

                    # Create sample record
                    # chunk_token_ids: exact tokens the model saw (with/without BOS as-is)
                    # trigger_token_idx: raw seq_idx into chunk_token_ids
                    # token_id: the actual token at that position (for verification)
                    sample = {
                        "sample_id": samples_collected,
                        "barycentric_coords": bary.tolist(),
                        "distances_to_vertices": dists.tolist(),
                        "nearest_vertex": nearest_vertex,
                        "distance_to_nearest": distance_to_nearest,
                        "full_text": full_text,
                        "chunk_token_ids": sequence_tokens.tolist(),
                        "trigger_word": trigger_word,
                        "trigger_token_idx": int(seq_idx),
                        "trigger_word_idx": word_index,
                        "token_id": trigger_token_id,
                    }

                    samples_file.write(json.dumps(sample) + '\n')
                    samples_collected += 1
                    pbar.update(1)

                    if samples_collected >= args.target_samples:
                        break

                    # Checkpoint
                    if samples_collected % args.save_interval == 0:
                        samples_file.flush()

                total_inputs_processed += acts_flat.shape[0]

            del tokens, cache, acts, acts_flat, feature_acts

        pbar.close()

    finally:
        samples_file.close()

    # Save stats
    print(f"\nSaving stats...")
    stats = {
        "cluster_key": cluster_key,
        "n_clusters": args.n_clusters,
        "cluster_id": args.cluster_id,
        "k": k,
        "n_latents": len(latent_indices),
        "target_samples": args.target_samples,
        "samples_collected": samples_collected,
        "total_inputs_processed": total_inputs_processed,
        "skip_docs": args.skip_docs,
        "model_name": args.model_name,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "aanet_checkpoint": str(args.aanet_checkpoint),
        "samples_path": str(samples_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"Samples collected: {samples_collected:,}")
    print(f"Total inputs processed: {total_inputs_processed:,}")
    print(f"Output: {samples_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
