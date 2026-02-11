#!/usr/bin/env python3
"""
Refit AANet models for selected clusters and collect near-vertex samples.

This script implements a two-stage pipeline:

STAGE 1: AANet Training
1. Loads cluster selection results from the analysis
2. For each selected cluster, refits AANet at its archetypal elbow k value
3. Saves models, training curves, and metadata

STAGE 2: Near-Vertex Sample Collection (optional, via --collect_vertex_samples)
1. For each trained AANet, streams data through Gemma + SAE
2. Projects cluster-specific activations through AANet
3. Identifies samples near simplex vertices (potential pure belief states)
4. Saves text samples with trigger tokens for interpretation

Usage:
    # Stage 1 only (training):
    python refit_selected_clusters.py \
        --n_clusters_list 128,256,512,768 \
        --save_dir /workspace/outputs/selected_clusters_canonical \
        --model_name gemma-2-9b \
        --sae_release gemma-scope-9b-pt-res-canonical \
        --sae_id layer_20/width_16k/average_l0_68 \
        --aanet_streaming_steps 3000 \
        --device cuda

    # Stages 1 + 2 (training + vertex collection):
    python refit_selected_clusters.py \
        --n_clusters_list 128,256,512,768 \
        --save_dir /workspace/outputs/selected_clusters_canonical \
        --model_name gemma-2-9b \
        --sae_release gemma-scope-9b-pt-res-canonical \
        --sae_id layer_20/width_16k/average_l0_68 \
        --aanet_streaming_steps 3000 \
        --collect_vertex_samples \
        --samples_per_vertex 1000 \
        --vertex_distance_threshold 0.1 \
        --device cuda
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import login

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from real_data_utils import RealDataSampler
from aanet_pipeline.streaming_trainer import StreamingAAnetTrainer
from aanet_pipeline.training import TrainingConfig
from aanet_pipeline.cluster_summary import AAnetDescriptor
from aanet_pipeline.extrema import compute_diffusion_extrema, ExtremaConfig
from mess3_gmg_analysis_utils import sae_encode_features
from cluster_selection import select_promising_clusters, delete_special_tokens


def parse_manual_cluster_ids(manual_str):
    """Parse manual cluster IDs string into dict.

    Format: 'n_clusters1:id1,id2,id3;n_clusters2:id4,id5'
    Example: '512:23,45,67;768:100,200'

    Returns: {n_clusters1: [id1, id2, id3], n_clusters2: [id4, id5]}
    """
    if not manual_str:
        return None

    result = {}
    for group in manual_str.split(';'):
        group = group.strip()
        if not group:
            continue
        n_clusters_str, ids_str = group.split(':')
        n_clusters = int(n_clusters_str.strip())
        ids = [int(x.strip()) for x in ids_str.split(',')]
        result[n_clusters] = ids
    return result


def parse_manual_k(manual_str):
    """Parse manual k overrides string into nested dict.

    Format: 'n_clusters1:cluster_id1=k1,cluster_id2=k2;n_clusters2:cluster_id3=k3'
    Example: '512:261=3,504=4;768:100=5'

    Returns: {n_clusters1: {cluster_id1: k1, cluster_id2: k2}, ...}
    """
    if not manual_str:
        return None

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
            cluster_id = int(cluster_id_str.strip())
            k = int(k_str.strip())
            cluster_k_map[cluster_id] = k

        result[n_clusters] = cluster_k_map
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Refit AANet models for selected clusters")

    # Selection parameters
    parser.add_argument("--n_clusters_list", type=str, required=True,
                       help="Comma-separated list of n_clusters values to process (e.g., '128,256,512,768')")
    parser.add_argument("--csv_dir", type=str,
                       default="outputs/real_data_analysis_canonical",
                       help="Directory containing CSV files with AANet metrics")
    parser.add_argument("--manual_cluster_ids", type=str, default=None,
                       help="Manually specify cluster IDs instead of automatic selection. "
                            "Format: 'n_clusters1:id1,id2,id3;n_clusters2:id4,id5' "
                            "Example: '512:23,45,67;768:100,200'. "
                            "When specified, bypasses automatic cluster selection criteria.")
    parser.add_argument("--manual_k", type=str, default=None,
                       help="Manually override k (number of archetypes) for specific clusters. "
                            "Format: 'n_clusters1:cluster_id1=k1,cluster_id2=k2;n_clusters2:cluster_id3=k3' "
                            "Example: '512:261=3,504=4;768:100=5'. "
                            "Overrides the automatic arch_elbow_k detection for specified clusters.")

    # Output
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Base directory to save refitted models")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip Stage 1 (AANet training), load pre-trained models and only run Stage 2 (vertex collection). Requires --stage1_models_dir.")
    parser.add_argument("--stage1_models_dir", type=str, default=None,
                       help="Directory containing Stage 1 AANet models (e.g., outputs/real_data_analysis_canonical/clusters_N/). Required when --skip_training is set. Models should be named aanet_cluster_{cid}_k{k}.pt")
    parser.add_argument("--stage1_subfolder_pattern", type=str, default="clusters_{n_clusters}",
                       help="Subfolder pattern under stage1_models_dir where AANet .pt files live. "
                            "Use {n_clusters} as placeholder. "
                            "Default: 'clusters_{n_clusters}'. "
                            "Example for cooc: 'mutual_info_{n_clusters}/clusters_{n_clusters}'")

    # Model & SAE
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/average_l0_68")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="/workspace/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)

    # Data streaming
    parser.add_argument("--hf_dataset", type=str, default="HuggingFaceFW/fineweb",
                       help="HuggingFace dataset identifier")
    parser.add_argument("--hf_subset_name", type=str, default="sample-10BT",
                       help="Dataset subset/config name")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--activity_batch_size", type=int, default=32)
    parser.add_argument("--activity_seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aanet_prefetch_size", type=int, default=1024,
                       help="Prefetch buffer size for RealDataSampler")
    parser.add_argument("--shuffle_buffer_size", type=int, default=50000,
                       help="Shuffle buffer size for streaming dataset")
    parser.add_argument("--max_doc_tokens", type=int, default=2000,
                       help="Filter documents longer than this (approximate, uses ~4 chars/token)")

    # AANet training
    parser.add_argument("--aanet_streaming_steps", type=int, default=3000,
                       help="Number of training steps (increased from original 2000)")
    parser.add_argument("--aanet_batch_size", type=int, default=128)
    parser.add_argument("--aanet_lr", type=float, default=0.0025)
    parser.add_argument("--aanet_weight_decay", type=float, default=1e-5)
    parser.add_argument("--aanet_layer_widths", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--aanet_simplex_scale", type=float, default=1.0)
    parser.add_argument("--aanet_noise", type=float, default=0.05)
    parser.add_argument("--aanet_gamma_reconstruction", type=float, default=1.0)
    parser.add_argument("--aanet_gamma_archetypal", type=float, default=4.0)
    parser.add_argument("--aanet_gamma_extrema", type=float, default=2.0)
    parser.add_argument("--aanet_grad_clip", type=float, default=1.0)
    parser.add_argument("--aanet_lr_patience", type=int, default=50,
                       help="Reduce LR after this many steps without improvement")
    parser.add_argument("--aanet_lr_factor", type=float, default=0.5,
                       help="Factor to reduce LR by (new_lr = old_lr * factor)")
    parser.add_argument("--aanet_min_lr", type=float, default=1e-6,
                       help="Don't reduce LR below this value")
    parser.add_argument("--aanet_early_stop_patience", type=int, default=500,
                       help="Stop training after this many steps without improvement")
    parser.add_argument("--aanet_min_delta", type=float, default=1e-6,
                       help="Minimum change in loss to count as improvement")
    parser.add_argument("--aanet_loss_smoothing_window",
                       type=lambda x: int(x) if int(x) > 7 else parser.error("aanet_loss_smoothing_window must be at least 8"),
                       default=20, help="Window size for smoothing loss before early stopping comparison")
    parser.add_argument("--aanet_seed", type=int, default=43)
    parser.add_argument("--aanet_active_threshold", type=float, default=1e-6,
                       help="Threshold for active samples in AAnet training")
    parser.add_argument("--aanet_min_samples", type=int, default=32,
                       help="Minimum dataset size before training (not used in streaming)")

    # Extrema initialization
    parser.add_argument("--extrema_enabled", action="store_true", default=True)
    parser.add_argument("--extrema_knn", type=int, default=150)
    parser.add_argument("--extrema_max_points", type=int, default=30000)
    parser.add_argument("--extrema_pca", type=float, default=0.95)
    parser.add_argument("--extrema_seed", type=int, default=431)
    parser.add_argument("--extrema_warmup_samples", type=int, default=10000,
                       help="Number of samples to collect for extrema initialization")

    # Stage 2: Near-vertex sample collection
    parser.add_argument("--collect_vertex_samples", action="store_true",
                       help="Enable Stage 2: collect near-vertex samples after training")
    parser.add_argument("--samples_per_vertex", type=int, default=1000,
                       help="Target number of samples near each vertex")
    parser.add_argument("--vertex_distance_threshold", type=float, default=0.1,
                       help="Max L2 distance from vertex to count as 'near'")
    parser.add_argument("--min_vertex_ratio", type=float, default=0.1,
                       help="Fail-safe: quit if rarest vertex has <X fraction of target (0.1 = 10%%)")
    parser.add_argument("--vertex_search_batch_size", type=int, default=32,
                       help="Batch size for vertex sample collection")
    parser.add_argument("--concurrent_aanets", type=int, default=5,
                       help="Number of AANets to keep in memory simultaneously for batched inference")
    parser.add_argument("--max_inputs_per_cluster", type=int, default=1000000,
                       help="Hard cap on total inputs processed per cluster (fail-safe)")
    parser.add_argument("--vertex_save_interval", type=int, default=10000,
                       help="Save vertex samples every N inputs (for checkpointing)")
    parser.add_argument("--vertex_skip_docs", type=int, default=None,
                       help="Skip this many documents before vertex collection (required with --skip_training)")

    args = parser.parse_args()

    # Validation: skip_training requires stage1_models_dir
    if args.skip_training and args.stage1_models_dir is None:
        parser.error("--skip_training requires --stage1_models_dir to specify where to load pre-trained models from")

    # Validation: skip_training requires vertex_skip_docs
    if args.skip_training and args.collect_vertex_samples and args.vertex_skip_docs is None:
        parser.error("--skip_training with --collect_vertex_samples requires --vertex_skip_docs to specify where to start sampling")

    # Warning: vertex_skip_docs without skip_training won't be used
    if not args.skip_training and args.vertex_skip_docs is not None:
        print("\nWARNING: --vertex_skip_docs is set but --skip_training is not enabled.")
        print("         When training AANets, the sampler will naturally continue from where training ends.")
        print("         The --vertex_skip_docs parameter will be ignored.\n")

    # Warning: stage1_models_dir without skip_training won't be used
    if not args.skip_training and args.stage1_models_dir is not None:
        print("\nWARNING: --stage1_models_dir is set but --skip_training is not enabled.")
        print("         Models will be trained from scratch. The --stage1_models_dir parameter will be ignored.\n")

    return args


# select_promising_clusters imported from cluster_selection module


def load_selected_clusters(csv_dir, n_clusters_list, manual_cluster_ids=None, manual_k=None):
    """
    Load selected clusters using the same logic as analyze_aanet_results.py

    Args:
        csv_dir: Directory containing CSV files with AANet metrics
        n_clusters_list: List of n_clusters values to process
        manual_cluster_ids: Optional dict mapping n_clusters -> list of cluster_ids.
                           If provided, bypasses automatic selection and uses these instead.
                           Format: {512: [23, 45, 67], 768: [100, 200]}
        manual_k: Optional dict mapping n_clusters -> {cluster_id: k} for k overrides.
                  Format: {512: {261: 3, 504: 4}, 768: {100: 5}}

    Returns: List of dicts with cluster info:
        {
            'n_clusters': int,
            'cluster_id': int,
            'latent_indices': list[int],
            'arch_elbow_k': int,
            'category': str,
            'recon_elbow_strength': float,
            'arch_elbow_strength': float,
        }
    """
    import pandas as pd

    selected_clusters = []

    for n_clusters in n_clusters_list:
        print(f"\n{'='*80}")
        print(f"Loading selected clusters for n_clusters={n_clusters}")
        print(f"{'='*80}")

        # Load CSV with metrics
        csv_path = Path(csv_dir) / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
        if not csv_path.exists():
            print(f"WARNING: CSV not found at {csv_path}, skipping")
            continue

        df = pd.read_csv(csv_path)
        df['n_clusters_total'] = n_clusters

        # Read elbow metrics directly from CSV (calculated by analyze_real_saes.py)
        # Note: Elbow metrics are per-cluster, so they're identical across all k rows for a given cluster
        elbow_results = []
        for cluster_id, group in df.groupby('cluster_id'):
            # Take the first row for this cluster (all rows have same elbow metrics)
            row = group.iloc[0]

            elbow_results.append({
                'n_clusters_total': n_clusters,
                'cluster_id': cluster_id,
                'n_latents': row['n_latents'],
                'latent_indices': row['latent_indices'],
                'aanet_recon_loss_elbow_k': row['aanet_recon_loss_elbow_k'],
                'aanet_recon_loss_elbow_strength': row['aanet_recon_loss_elbow_strength'],
                'aanet_archetypal_loss_elbow_k': row['aanet_archetypal_loss_elbow_k'],
                'aanet_archetypal_loss_elbow_strength': row['aanet_archetypal_loss_elbow_strength'],
                'k_differential': row['k_differential'],
                'recon_is_monotonic': row['recon_is_monotonic'],
                'arch_is_monotonic': row['arch_is_monotonic'],
                'recon_pct_decrease': row['recon_pct_decrease'],
                'arch_pct_decrease': row['arch_pct_decrease'],
            })

        elbow_df = pd.DataFrame(elbow_results)

        # Use manual cluster IDs if provided, otherwise auto-select
        if manual_cluster_ids and n_clusters in manual_cluster_ids:
            selected_ids = set(manual_cluster_ids[n_clusters])
            # Validate that all specified cluster IDs exist in the data
            available_ids = set(elbow_df['cluster_id'].tolist())
            missing_ids = selected_ids - available_ids
            if missing_ids:
                print(f"  WARNING: Cluster IDs not found in data: {missing_ids}")
                selected_ids = selected_ids & available_ids

            # Create placeholder category stats - all marked as "M" for manual
            cat_stats = {
                'A_strong_both': [],
                'B_recon_outliers': [],
                'C_arch_outliers': [],
                'D_agreement': [],
                'M_manual': list(selected_ids)
            }
            print(f"  Using {len(selected_ids)} manually specified clusters")
        else:
            # Select clusters using automatic criteria
            selected_ids, cat_stats = select_promising_clusters(elbow_df, n_clusters)
            print(f"  Auto-selected {len(selected_ids)} clusters")

        # Extract cluster info
        for cluster_id in selected_ids:
            row = elbow_df[elbow_df['cluster_id'] == cluster_id].iloc[0]

            # Determine category (priority: M, A, D, B, C)
            category = None
            if 'M_manual' in cat_stats and cluster_id in cat_stats['M_manual']:
                category = 'M'
            elif cluster_id in cat_stats['A_strong_both']:
                category = 'A'
            elif cluster_id in cat_stats['D_agreement']:
                category = 'D'
            elif cluster_id in cat_stats['B_recon_outliers']:
                category = 'B'
            elif cluster_id in cat_stats['C_arch_outliers']:
                category = 'C'

            # Parse latent indices
            import ast
            latent_indices = ast.literal_eval(row['latent_indices'])

            # Determine k to use: manual override or arch_elbow_k
            arch_k = int(row['aanet_archetypal_loss_elbow_k'])
            recon_k = int(row['aanet_recon_loss_elbow_k'])

            # Check for manual k override
            if manual_k and n_clusters in manual_k and cluster_id in manual_k[n_clusters]:
                k_to_use = manual_k[n_clusters][cluster_id]
                print(f"    Cluster {cluster_id}: using manual k={k_to_use} (arch_elbow={arch_k}, recon_elbow={recon_k})")
            else:
                k_to_use = arch_k

            selected_clusters.append({
                'n_clusters': n_clusters,
                'cluster_id': int(cluster_id),
                'latent_indices': latent_indices,
                'arch_elbow_k': k_to_use,  # This is the k that will be used
                'original_arch_elbow_k': arch_k,  # Keep original for reference
                'recon_elbow_k': recon_k,
                'category': category,
                'recon_elbow_strength': float(row['aanet_recon_loss_elbow_strength']),
                'arch_elbow_strength': float(row['aanet_archetypal_loss_elbow_strength']),
                'n_latents': int(row['n_latents']),
            })

    return selected_clusters


# delete_special_tokens imported from cluster_selection module


def collect_warmup_data(sampler, model, sae, hook_name, cluster_indices, n_samples, batch_size, seq_len, device):
    """Collect samples for extrema initialization"""
    print(f"  Collecting {n_samples} warmup samples...")

    warmup_buffer = []
    samples_collected = 0

    with torch.no_grad():
        pbar = tqdm(total=n_samples, desc="Warmup")
        while samples_collected < n_samples:
            tokens = sampler.sample_tokens_batch(batch_size, seq_len, device)
            _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
            acts = cache[hook_name]
            acts_flat = acts.reshape(-1, acts.shape[-1])
            feature_acts, _, _ = sae_encode_features(sae, acts_flat)

            # Exclude BOS tokens (position 0) from training data
            feature_acts = delete_special_tokens(feature_acts, tokens.shape[0], seq_len, device)

            # Get cluster-specific activations
            acts_c = feature_acts[:, cluster_indices]

            # Filter for active samples
            active_mask = (acts_c.abs().sum(dim=1) > 0)
            acts_c_active = acts_c[active_mask]

            if acts_c_active.shape[0] > 0:
                # Compute cluster-specific reconstruction
                W_c = sae.W_dec[cluster_indices, :]
                X_recon = acts_c_active @ W_c

                warmup_buffer.append(X_recon.cpu())
                samples_collected += X_recon.shape[0]
                pbar.update(X_recon.shape[0])

            del tokens, cache, acts, acts_flat, feature_acts

            if samples_collected >= n_samples:
                break

        pbar.close()

    warmup_data = torch.cat(warmup_buffer, dim=0)[:n_samples]
    return warmup_data


def train_single_cluster(cluster_info, model, sae, sampler, args):
    """Train AANet for a single cluster"""

    n_clusters = cluster_info['n_clusters']
    cluster_id = cluster_info['cluster_id']
    k = cluster_info['arch_elbow_k']
    latent_indices = cluster_info['latent_indices']
    category = cluster_info['category']

    print(f"\n{'='*80}")
    print(f"Training cluster {cluster_id} (n={n_clusters}, k={k}, category={category})")
    print(f"  Latents: {len(latent_indices)}")
    print(f"{'='*80}")

    # Setup paths
    save_dir = Path(args.save_dir) / f"n{n_clusters}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}.pt"
    curves_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}_curves.json"
    metadata_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}_metadata.json"

    # Convert latent indices to tensor
    cluster_indices_tensor = torch.tensor(latent_indices, device=args.device, dtype=torch.long)

    # Get hook name
    layer_idx = int(args.sae_id.split("/")[0].split("_")[1])
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Get SAE decoder dimension
    d_model = sae.W_dec.shape[1]

    # Create descriptor
    descriptor = AAnetDescriptor(
        cluster_id=cluster_id,
        label=f"cluster_{cluster_id}",
        latent_indices=latent_indices,
        component_names=[],
        is_noise=False
    )

    # Create training config for StreamingAAnetTrainer
    config = TrainingConfig(
        k=k,
        epochs=1,  # Not used in streaming
        batch_size=args.aanet_batch_size,
        learning_rate=args.aanet_lr,
        weight_decay=args.aanet_weight_decay,
        gamma_reconstruction=args.aanet_gamma_reconstruction,
        gamma_archetypal=args.aanet_gamma_archetypal,
        gamma_extrema=args.aanet_gamma_extrema,
        simplex_scale=args.aanet_simplex_scale,
        noise=args.aanet_noise,
        layer_widths=args.aanet_layer_widths,
        min_samples=args.aanet_min_samples,
        lr_patience=args.aanet_lr_patience,
        lr_factor=args.aanet_lr_factor,
        grad_clip=args.aanet_grad_clip,
        active_threshold=args.aanet_active_threshold,
    )

    # Create trainer
    trainer = StreamingAAnetTrainer(
        descriptors=[descriptor],
        config=config,
        device=args.device,
        input_dim=d_model,
        sae_decoder_weight=sae.W_dec,
    )

    # Extrema initialization
    if args.extrema_enabled:
        warmup_data = collect_warmup_data(
            sampler, model, sae, hook_name,
            cluster_indices_tensor,
            args.extrema_warmup_samples,
            args.activity_batch_size,
            args.activity_seq_len,
            args.device
        )

        print(f"  Computing extrema from {warmup_data.shape[0]} samples...")
        extrema_np = warmup_data.cpu().numpy()
        extrema_config = ExtremaConfig(
            knn=args.extrema_knn,
            max_points=args.extrema_max_points,
            pca_components=args.extrema_pca,  # 0.95 = keep components explaining 95% variance
            random_seed=args.extrema_seed
        )
        extrema = compute_diffusion_extrema(extrema_np, max_k=k, config=extrema_config)

        if extrema is not None and extrema.shape[0] == k:
            trainer.models[cluster_id].set_archetypes(extrema.to(args.device))
            print(f"  Initialized {k} extrema")
        else:
            print(f"  WARNING: Failed to initialize extrema (got {extrema.shape[0] if extrema is not None else 0}/{k})")

        # CRITICAL: Free warmup data before training
        del warmup_data, extrema_np
        if extrema is not None:
            del extrema
        torch.cuda.empty_cache()

    # Training loop with early stopping
    metrics_history = {
        "loss": [],
        "reconstruction_loss": [],
        "archetypal_loss": [],
        "extrema_loss": []
    }

    # Early stopping tracking
    best_loss = float('inf')
    steps_since_improvement = 0
    best_step = 0
    check_interval = (args.aanet_loss_smoothing_window // 2) + (args.aanet_loss_smoothing_window // 4) # Check every 3/4-window

    print(f"  Starting training (max {args.aanet_streaming_steps} steps, early stop patience={args.aanet_early_stop_patience})...")
    print(f"  Early stopping will check every {check_interval} steps using {args.aanet_loss_smoothing_window}-step smoothed loss")
    for step in tqdm(range(args.aanet_streaming_steps), desc=f"Training"):
        # Sample batch
        tokens = sampler.sample_tokens_batch(args.aanet_batch_size, args.activity_seq_len, args.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
            acts = cache[hook_name]
            acts_flat = acts.reshape(-1, acts.shape[-1])
            feature_acts, _, _ = sae_encode_features(sae, acts_flat)

            # Exclude BOS tokens (position 0) from training data
            feature_acts = delete_special_tokens(feature_acts, tokens.shape[0], args.activity_seq_len, args.device)

        # Train step
        step_losses = trainer.train_step(feature_acts)

        # Record metrics and update scheduler
        if cluster_id in step_losses:
            current_loss = step_losses[cluster_id]['loss']

            for key, value in step_losses[cluster_id].items():
                metrics_history[key].append(value)

            # Step LR scheduler
            current_lr = trainer.optimizers[cluster_id].param_groups[0]['lr']
            if current_lr > args.aanet_min_lr:
                trainer.schedulers[cluster_id].step(current_loss)
                new_lr = trainer.optimizers[cluster_id].param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"\n  Step {step}: LR reduced from {current_lr:.2e} to {new_lr:.2e}")

            # Early stopping check - only compare at intervals to avoid overlapping windows
            # Check every half-window to ensure meaningful separation between comparisons
            if step % check_interval == 0 and len(metrics_history['loss']) >= args.aanet_loss_smoothing_window:
                # Use smoothed loss (mean over window)
                smoothed_loss = np.mean(metrics_history['loss'][-args.aanet_loss_smoothing_window:])

                if smoothed_loss < best_loss - args.aanet_min_delta:
                    best_loss = smoothed_loss
                    steps_since_improvement = -1  # Will become 0 after increment below
                    best_step = step
                    print(f"\n  Step {step}: New best smoothed loss: {best_loss:.6f}")

            # Always increment step counter
            steps_since_improvement += 1

            # Check for early stopping
            if steps_since_improvement >= args.aanet_early_stop_patience:
                print(f"\n  Early stopping at step {step} (best smoothed loss {best_loss:.6f} at step {best_step})")
                print(f"  No improvement for {steps_since_improvement} steps")
                break

        # Cleanup
        del tokens, cache, acts, acts_flat, feature_acts

    # Calculate final metrics (median of last 20)
    final_metrics = {}
    for key, values in metrics_history.items():
        if values and len(values) >= 20:
            final_metrics[key] = float(np.median(values[-20:]))
        elif values:
            final_metrics[key] = float(values[-1])
        else:
            final_metrics[key] = float('nan')

    print(f"  Final metrics:")
    print(f"    Loss: {final_metrics['loss']:.6f}")
    print(f"    Reconstruction: {final_metrics['reconstruction_loss']:.6f}")
    print(f"    Archetypal: {final_metrics['archetypal_loss']:.6f}")
    print(f"    Extrema: {final_metrics['extrema_loss']:.6f}")

    # Save model
    print(f"  Saving model to {model_path}")
    torch.save(trainer.models[cluster_id].state_dict(), model_path)

    # Save training curves
    print(f"  Saving curves to {curves_path}")
    with open(curves_path, 'w') as f:
        json.dump(metrics_history, f)

    # Save metadata
    metadata = {
        **cluster_info,
        'final_metrics': final_metrics,
        'training_config': {
            'streaming_steps': args.aanet_streaming_steps,
            'batch_size': args.aanet_batch_size,
            'learning_rate': args.aanet_lr,
            'extrema_enabled': args.extrema_enabled,
        },
        'model_path': str(model_path),
        'curves_path': str(curves_path),
    }

    print(f"  Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Cluster {cluster_id} complete!")

    # CRITICAL: Free GPU memory
    print(f"  Cleaning up GPU memory...")
    del trainer
    if 'warmup_data' in locals():
        del warmup_data
    torch.cuda.empty_cache()

    return metadata


def collect_vertex_samples_for_cluster(cluster_metadata, model, sae, sampler, tokenizer, args):
    """
    Collect near-vertex samples for a single trained AANet cluster.

    Returns: vertex_stats dict with collection results
    """
    n_clusters = cluster_metadata['n_clusters']
    cluster_id = cluster_metadata['cluster_id']
    k = cluster_metadata['arch_elbow_k']
    latent_indices = cluster_metadata['latent_indices']
    category = cluster_metadata['category']

    print(f"\n{'='*80}")
    print(f"Collecting vertex samples for cluster {cluster_id} (n={n_clusters}, k={k}, category={category})")
    print(f"{'='*80}")

    # Setup paths
    save_dir = Path(args.save_dir) / f"n{n_clusters}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}.pt"
    samples_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}_vertex_samples.jsonl"
    stats_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}_vertex_stats.json"
    all_coords_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}_all_barycentric_coords.jsonl"

    # Load trained AANet
    from AAnet.AAnet_torch.models.AAnet_vanilla import AAnet_vanilla
    d_model = sae.W_dec.shape[1]
    aanet = AAnet_vanilla(
        input_shape=d_model,
        n_archetypes=k,
        noise=args.aanet_noise,
        layer_widths=args.aanet_layer_widths,
        activation_out="tanh",
        simplex_scale=args.aanet_simplex_scale,
        device=args.device
    )

    # Load model from Stage 1 if skipping training, otherwise from refit output
    if 'stage1_model_path' in cluster_metadata:
        load_path = Path(cluster_metadata['stage1_model_path'])
        print(f"  Loading pre-trained model from Stage 1: {load_path}")
    else:
        load_path = model_path
        print(f"  Loading model: {load_path}")

    aanet.load_state_dict(torch.load(load_path))
    aanet.eval()

    # Get hook name
    layer_idx = int(args.sae_id.split("/")[0].split("_")[1])
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Convert latent indices to tensor
    cluster_indices_tensor = torch.tensor(latent_indices, device=args.device, dtype=torch.long)

    # Initialize vertex sample collections
    vertex_samples = {i: [] for i in range(k)}
    vertex_stats = {i: {"samples": 0, "reached_target": False} for i in range(k)}

    # Create one-hot vertex targets for distance calculation (k-dimensional barycentric)
    vertices = torch.eye(k, device=args.device)  # Shape: (k, k)

    total_inputs_processed = 0
    samples_saved = 0
    coords_saved = 0

    # Milestone tracking for reduced printing
    sample_milestone = args.samples_per_vertex // 10  # Print every 10% of target
    inputs_milestone = args.max_inputs_per_cluster // 10  # Print every 10% of max inputs
    last_vertex_milestone = {i: 0 for i in range(k)}
    last_inputs_milestone = 0

    # Open JSONL files for incremental writing
    samples_file = open(samples_path, 'w')
    all_coords_file = open(all_coords_path, 'w')

    print(f"  Target: {args.samples_per_vertex} samples per vertex")
    print(f"  Distance threshold: {args.vertex_distance_threshold}")
    print(f"  Starting collection...")
    print(f"  Saving all barycentric coordinates to: {all_coords_path.name}")

    try:
        while total_inputs_processed < args.max_inputs_per_cluster:
            # Check early stopping: stop if ALL vertices have enough samples
            all_reached = all(v["samples"] >= args.samples_per_vertex for v in vertex_stats.values())
            if all_reached:
                samples_list = [v["samples"] for v in vertex_stats.values()]
                print(f"\n  ✓ All vertices reached target: {samples_list}")
                break

            # Sample batch
            tokens = sampler.sample_tokens_batch(args.vertex_search_batch_size, args.activity_seq_len, args.device)

            with torch.no_grad():
                # Forward through model
                _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                acts = cache[hook_name]
                acts_flat = acts.reshape(-1, acts.shape[-1])

                # Encode with SAE
                feature_acts, _, _ = sae_encode_features(sae, acts_flat)

                # Get cluster-specific activations
                acts_c = feature_acts[:, cluster_indices_tensor]

                # Filter for active samples
                active_mask = (acts_c.abs().sum(dim=1) > 0)

                if not active_mask.any():
                    total_inputs_processed += acts_flat.shape[0]
                    pbar.update(acts_flat.shape[0])
                    continue

                acts_c_active = acts_c[active_mask]
                active_indices = torch.where(active_mask)[0]

                # Compute cluster-specific reconstruction
                W_c = sae.W_dec[cluster_indices_tensor, :]
                X_recon = acts_c_active @ W_c

                # Forward through AANet to get bottleneck coords
                _, _, embedding = aanet(X_recon)  # embedding is (k-1)-dim Euclidean coords

                # Convert from Euclidean to k-dimensional barycentric coordinates
                # Using AANet's built-in conversion function
                embedding = aanet.euclidean_to_barycentric(embedding)  # Now shape: (batch, k)

                # Save all barycentric coordinates for distribution analysis
                for idx, coord in zip(active_indices.cpu().numpy(), embedding.cpu().numpy()):
                    batch_idx = idx // args.activity_seq_len
                    seq_idx = idx % args.activity_seq_len

                    # Skip BOS token (position 0) - not meaningful for analysis
                    if seq_idx == 0:
                        continue

                    coord_record = {
                        "barycentric_coords": coord.tolist(),
                        "batch_idx": int(batch_idx),
                        "seq_idx": int(seq_idx),
                    }
                    all_coords_file.write(json.dumps(coord_record) + '\n')
                    coords_saved += 1

                # Flush periodically
                if coords_saved % 10000 == 0:
                    all_coords_file.flush()

                # Accumulate samples per (batch_idx, vertex_id) to avoid duplicates
                # Key: (batch_idx, vertex_id), Value: sample dict with lists
                batch_samples = {}

                # Calculate distances to each vertex
                for i in range(k):
                    vertex = vertices[i]
                    distances = torch.norm(embedding - vertex, dim=1)
                    near_vertex_mask = distances < args.vertex_distance_threshold

                    if not near_vertex_mask.any():
                        continue

                    # Get indices of samples near this vertex
                    near_indices = active_indices[near_vertex_mask]
                    near_distances = distances[near_vertex_mask]

                    # For each near-vertex sample
                    for idx, dist in zip(near_indices.cpu().numpy(), near_distances.cpu().numpy()):
                        # Calculate batch and sequence position
                        batch_idx = idx // args.activity_seq_len
                        seq_idx = idx % args.activity_seq_len

                        # Skip BOS token (position 0) - not meaningful for interpretation
                        if seq_idx == 0:
                            continue

                        # Create unique key for this (sequence, vertex) pair
                        key = (int(batch_idx), int(i))

                        # If first time seeing this sequence for this vertex, initialize
                        if key not in batch_samples:
                            # Get full sequence tokens (skip special tokens for clean text)
                            sequence_tokens = tokens[batch_idx].cpu().numpy()
                            full_text = tokenizer.decode(sequence_tokens, skip_special_tokens=True)

                            batch_samples[key] = {
                                "vertex_id": int(i),
                                "distances_to_vertex": [],
                                "full_text": full_text,
                                "chunk_token_ids": sequence_tokens.tolist(),
                                "trigger_token_indices": [],
                                "trigger_token_ids": [],
                                "trigger_word_indices": [],
                                "trigger_words": [],
                            }

                        # Add this token's information to the accumulated sample
                        trigger_token_id = tokens[batch_idx, seq_idx].cpu().item()
                        trigger_word = tokenizer.decode([trigger_token_id], skip_special_tokens=True)

                        # Compute word index by decoding up to this token and counting words
                        # Use skip_special_tokens=True to avoid <bos>/<eos> being counted as words
                        text_up_to_token = tokenizer.decode(
                            tokens[batch_idx, :seq_idx+1].cpu().numpy(),
                            skip_special_tokens=True
                        )
                        # Count words (split by whitespace), subtract 1 for 0-indexing
                        word_index = len(text_up_to_token.split()) - 1
                        if word_index < 0:
                            word_index = 0

                        batch_samples[key]["distances_to_vertex"].append(float(dist))
                        batch_samples[key]["trigger_token_indices"].append(int(seq_idx))
                        batch_samples[key]["trigger_token_ids"].append(int(trigger_token_id))
                        batch_samples[key]["trigger_word_indices"].append(int(word_index))
                        batch_samples[key]["trigger_words"].append(trigger_word)

                # Write all accumulated samples to file
                for sample in batch_samples.values():
                    samples_file.write(json.dumps(sample) + '\n')
                    vertex_stats[sample["vertex_id"]]["samples"] += 1
                    samples_saved += 1

                total_inputs_processed += acts_flat.shape[0]

            # Check if any milestone crossed (vertex samples or inputs processed)
            # This happens AFTER processing all vertices in the batch
            vertex_milestone_crossed = False
            for i in range(k):
                current_milestone = (vertex_stats[i]["samples"] // sample_milestone) * sample_milestone
                if current_milestone > last_vertex_milestone[i] and current_milestone > 0:
                    last_vertex_milestone[i] = current_milestone
                    vertex_milestone_crossed = True

            current_inputs_milestone = (total_inputs_processed // inputs_milestone) * inputs_milestone
            inputs_milestone_crossed = (current_inputs_milestone > last_inputs_milestone and current_inputs_milestone > 0)
            if inputs_milestone_crossed:
                last_inputs_milestone = current_inputs_milestone

            # Print combined milestone update
            if vertex_milestone_crossed or inputs_milestone_crossed:
                vertex_counts = ", ".join([f"v{i}={vertex_stats[i]['samples']}" for i in range(k)])
                print(f"  Iteration {total_inputs_processed:,}: {vertex_counts}")

            del tokens, cache, acts, acts_flat, feature_acts

    finally:
        samples_file.close()
        all_coords_file.close()

    # Print final summary
    print(f"\n  Collection complete!")
    print(f"  Total inputs processed: {total_inputs_processed:,}")
    print(f"  Total near-vertex samples: {samples_saved:,}")
    print(f"  Total barycentric coords saved: {coords_saved:,}")
    print(f"  Samples per vertex:")
    for i in range(k):
        status = "✓" if vertex_stats[i]["samples"] >= args.samples_per_vertex else "✗"
        print(f"    Vertex {i}: {vertex_stats[i]['samples']:,} {status}")

    # Update reached_target flags
    for i in range(k):
        vertex_stats[i]["reached_target"] = vertex_stats[i]["samples"] >= args.samples_per_vertex

    all_reached = all(v["reached_target"] for v in vertex_stats.values())
    min_samples = min(v["samples"] for v in vertex_stats.values())

    # Determine collection status
    if all_reached:
        collection_status = "complete"
    else:
        incomplete_vertices = [i for i in range(k) if not vertex_stats[i]["reached_target"]]
        collection_status = f"incomplete_vertices_{','.join(map(str, incomplete_vertices))}"

    # Save stats
    stats = {
        "cluster_id": cluster_id,
        "n_clusters": n_clusters,
        "k": k,
        "category": category,
        "target_samples_per_vertex": args.samples_per_vertex,
        "total_inputs_processed": total_inputs_processed,
        "total_samples_collected": samples_saved,
        "total_barycentric_coords_saved": coords_saved,
        "vertex_stats": vertex_stats,
        "all_vertices_reached_target": all_reached,
        "collection_status": collection_status,
        "model_name": args.model_name,
        "tokenizer": str(type(tokenizer).__name__),
        "distance_threshold": args.vertex_distance_threshold,
        "all_coords_path": str(all_coords_path),
    }

    print(f"\n  Saving stats to {stats_path}")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  ✓ Collected {samples_saved} vertex samples across {k} vertices")
    print(f"     Samples per vertex: {[vertex_stats[i]['samples'] for i in range(k)]}")
    print(f"  ✓ Saved {coords_saved} total barycentric coordinates for distribution analysis")
    print(f"     Status: {collection_status}")

    return stats


def main():
    args = parse_args()

    # Parse n_clusters_list
    n_clusters_list = [int(x.strip()) for x in args.n_clusters_list.split(',')]
    print(f"Processing n_clusters: {n_clusters_list}")

    # HF login
    if args.hf_token:
        login(token=args.hf_token)

    # Parse manual cluster IDs if provided
    manual_cluster_ids = parse_manual_cluster_ids(args.manual_cluster_ids)
    if manual_cluster_ids:
        print(f"\nManual cluster IDs specified: {manual_cluster_ids}")

    # Parse manual k overrides if provided
    manual_k = parse_manual_k(args.manual_k)
    if manual_k:
        print(f"\nManual k overrides specified: {manual_k}")

    # Load selected clusters
    print("\n" + "="*80)
    print("LOADING SELECTED CLUSTERS")
    print("="*80)
    selected_clusters = load_selected_clusters(args.csv_dir, n_clusters_list, manual_cluster_ids, manual_k)
    print(f"\nTotal selected clusters: {len(selected_clusters)}")

    # Check if we found any clusters
    if len(selected_clusters) == 0:
        print("\n" + "="*80)
        print("ERROR: No clusters selected! Check that CSV files exist.")
        print("="*80)
        print(f"Expected location: {args.csv_dir}/clusters_*/consolidated_metrics_n*.csv")
        return

    # Load model and SAE (shared across all clusters)
    print("\n" + "="*80)
    print("LOADING MODEL AND SAE")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"SAE: {args.sae_release}/{args.sae_id}")

    # Login to Hugging Face if token provided
    if args.hf_token:
        print("Logging in to Hugging Face with provided token...")
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        print("Logging in to Hugging Face with HF_TOKEN environment variable...")
        login(token=os.environ["HF_TOKEN"])

    # Build model kwargs
    model_kwargs = {}
    if args.cache_dir:
        model_kwargs['cache_dir'] = args.cache_dir

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        center_unembed=False,  # Required for Gemma-2 models with logit softcap
        center_writing_weights=False,
        dtype="bfloat16",  # CRITICAL: Use bfloat16 to save ~18GB memory!
        **model_kwargs
    )
    model.eval()

    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device
    )
    sae.eval()

    # Create data sampler
    print("\n" + "="*80)
    print("INITIALIZING DATA SAMPLER")
    print("="*80)

    # Seed Python's random for reproducibility
    import random
    random.seed(args.seed)
    print(f"Set Python random seed to {args.seed}")

    sampler = RealDataSampler(
        model,
        hf_dataset=args.hf_dataset,
        hf_subset_name=args.hf_subset_name,
        split=args.dataset_split,
        seed=args.seed,
        prefetch_size=args.aanet_prefetch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_doc_tokens=args.max_doc_tokens
    )

    # Stage 1: Train clusters (or load existing)
    manifest_path = Path(args.save_dir) / "manifest.json"

    if args.skip_training:
        # Load pre-trained models from Stage 1
        print("\n" + "="*80)
        print("SKIPPING STAGE 1: Loading pre-trained models from Stage 1 outputs")
        print("="*80)
        print(f"Stage 1 models directory: {args.stage1_models_dir}")

        # Build metadata from selected clusters
        all_metadata = []
        for cluster_info in selected_clusters:
            n_clusters = cluster_info['n_clusters']
            cluster_id = cluster_info['cluster_id']
            k = cluster_info['arch_elbow_k']  # Use archetypal elbow k
            category = cluster_info['category']
            latent_indices = cluster_info['latent_indices']

            # Check if model exists in Stage 1 output
            subfolder = args.stage1_subfolder_pattern.format(n_clusters=n_clusters)
            stage1_model_path = Path(args.stage1_models_dir) / subfolder / f"aanet_cluster_{cluster_id}_k{k}.pt"

            if not stage1_model_path.exists():
                print(f"\nWARNING: Model not found for cluster {cluster_id} k={k}")
                print(f"  Expected: {stage1_model_path}")
                print(f"  Skipping this cluster...")
                continue

            # Metadata structure matching what training produces
            # Use **cluster_info to include all keys (like training branch does)
            all_metadata.append({
                **cluster_info,
                'stage1_model_path': str(stage1_model_path),  # Add path to pre-trained model
            })

        print(f"Found {len(all_metadata)} pre-trained models from Stage 1")

        if len(all_metadata) == 0:
            print("\nERROR: No pre-trained models found!")
            print(f"Check that Stage 1 models exist in: {args.stage1_models_dir}")
            return

        # Create manifest for skip_training mode
        manifest = {
            'total_clusters': len(all_metadata),
            'n_clusters_list': n_clusters_list,
            'model_name': args.model_name,
            'tokenizer': str(type(model.tokenizer).__name__),
            'sae_release': args.sae_release,
            'sae_id': args.sae_id,
            'stage1_models_dir': args.stage1_models_dir,
            'clusters': all_metadata
        }

    else:
        # Train each cluster
        print("\n" + "="*80)
        print("STAGE 1: TRAINING CLUSTERS")
        print("="*80)

        all_metadata = []
        for i, cluster_info in enumerate(selected_clusters):
            print(f"\n[{i+1}/{len(selected_clusters)}]")
            metadata = train_single_cluster(cluster_info, model, sae, sampler, args)
            all_metadata.append(metadata)

        # Save training manifest
        print(f"\n" + "="*80)
        print(f"Saving training manifest to {manifest_path}")
        print("="*80)

        manifest = {
            'total_clusters': len(all_metadata),
            'n_clusters_list': n_clusters_list,
            'model_name': args.model_name,
            'tokenizer': str(type(model.tokenizer).__name__),
            'sae_release': args.sae_release,
            'sae_id': args.sae_id,
            'clusters': all_metadata
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{'='*80}")
        print("STAGE 1 COMPLETE: AANet Training")
        print(f"{'='*80}")
        print(f"Trained {len(all_metadata)} clusters")
        print(f"Models saved to: {args.save_dir}")

    # Stage 2: Collect near-vertex samples
    if args.collect_vertex_samples:
        print(f"\n{'='*80}")
        print("STAGE 2: COLLECTING NEAR-VERTEX SAMPLES")
        print(f"{'='*80}")
        print(f"Target: {args.samples_per_vertex} samples per vertex")
        print(f"Distance threshold: {args.vertex_distance_threshold}")

        # Skip ahead in the dataset if using --skip_training
        # (When training, sampler naturally continues from where training ended)
        if args.skip_training and args.vertex_skip_docs and args.vertex_skip_docs > 0:
            print(f"\n  Skipping ahead {args.vertex_skip_docs:,} documents to avoid training data...")
            docs_skipped = 0
            while docs_skipped < args.vertex_skip_docs:
                try:
                    _ = next(sampler.iterator)
                    docs_skipped += 1
                    if docs_skipped % 10000 == 0:
                        print(f"    Skipped {docs_skipped:,}/{args.vertex_skip_docs:,} documents...")
                except StopIteration:
                    # Reached end of dataset, reset iterator
                    sampler.iterator = iter(sampler.dataset)
                    print(f"    Reached end of dataset, wrapped around")
            print(f"  ✓ Skipped {docs_skipped:,} documents, now sampling from fresh data")
        elif not args.skip_training:
            print(f"\n  Sampler will continue from where AANet training finished (no skip needed)")

        print(f"Processing clusters sequentially...")

        vertex_collection_results = []
        for i, cluster_metadata in enumerate(all_metadata):
            print(f"\n[{i+1}/{len(all_metadata)}]")
            try:
                vertex_stats = collect_vertex_samples_for_cluster(
                    cluster_metadata, model, sae, sampler, model.tokenizer, args
                )
                vertex_collection_results.append(vertex_stats)
            except Exception as e:
                print(f"  ✗ Error collecting samples for cluster {cluster_metadata['cluster_id']}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next cluster
                continue

        # Update manifest with vertex collection results
        manifest['vertex_collection'] = {
            'enabled': True,
            'samples_per_vertex_target': args.samples_per_vertex,
            'distance_threshold': args.vertex_distance_threshold,
            'results': vertex_collection_results
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{'='*80}")
        print("STAGE 2 COMPLETE: Vertex Sample Collection")
        print(f"{'='*80}")
        print(f"Collected samples for {len(vertex_collection_results)} clusters")

        # Summary statistics
        complete_clusters = sum(1 for r in vertex_collection_results if r['all_vertices_reached_target'])
        print(f"  Complete: {complete_clusters}/{len(vertex_collection_results)}")
        print(f"  Incomplete: {len(vertex_collection_results) - complete_clusters}/{len(vertex_collection_results)}")

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"Stage 1: Trained {len(all_metadata)} clusters")
    if args.collect_vertex_samples:
        print(f"Stage 2: Collected vertex samples for {len(vertex_collection_results)} clusters")
    print(f"Output directory: {args.save_dir}")


if __name__ == "__main__":
    main()
