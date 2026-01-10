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


def parse_args():
    parser = argparse.ArgumentParser(description="Refit AANet models for selected clusters")

    # Selection parameters
    parser.add_argument("--n_clusters_list", type=str, required=True,
                       help="Comma-separated list of n_clusters values to process (e.g., '128,256,512,768')")
    parser.add_argument("--corrected_csv_dir", type=str,
                       default="outputs/real_data_analysis_canonical",
                       help="Directory containing corrected CSV files")

    # Output
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Base directory to save refitted models")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip Stage 1 (training), load existing manifest and only run Stage 2 (vertex collection)")

    # Model & SAE
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/average_l0_68")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="/workspace/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)

    # Data streaming
    parser.add_argument("--dataset_name", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--activity_batch_size", type=int, default=32)
    parser.add_argument("--activity_seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aanet_prefetch_size", type=int, default=1024,
                       help="Prefetch buffer size for RealDataSampler")

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
    parser.add_argument("--aanet_lr_patience", type=int, default=5)
    parser.add_argument("--aanet_lr_factor", type=float, default=0.5)
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

    return parser.parse_args()


def select_promising_clusters(df, n_clusters_val, delta_k_threshold=1, sd_outlier=3, sd_strong=1):
    """
    Select promising clusters - copied from analyze_aanet_results.py
    """
    import pandas as pd

    # Filter for this n_clusters value
    group = df[df['n_clusters_total'] == n_clusters_val].copy()

    # Apply quality filters
    group = group[group['n_latents'] >= 2].copy()
    group = group[group['recon_is_monotonic'] == True].copy()
    group = group[group['arch_is_monotonic'] == True].copy()
    group = group[group['recon_pct_decrease'] >= 20].copy()
    group = group[group['arch_pct_decrease'] >= 20].copy()

    # Filter: Delta K constraint
    if 'k_differential' not in group.columns:
        group['k_differential'] = (group['aanet_recon_loss_elbow_k'] -
                                    group['aanet_archetypal_loss_elbow_k'])
    group = group[group['k_differential'].abs() <= delta_k_threshold].copy()

    if len(group) == 0:
        return set(), {}

    # Calculate distance from origin for ranking
    group['distance_from_origin'] = np.sqrt(
        group['aanet_recon_loss_elbow_strength']**2 +
        group['aanet_archetypal_loss_elbow_strength']**2
    )

    # Calculate SD-based thresholds
    recon_mean = group['aanet_recon_loss_elbow_strength'].mean()
    recon_std = group['aanet_recon_loss_elbow_strength'].std()
    arch_mean = group['aanet_archetypal_loss_elbow_strength'].mean()
    arch_std = group['aanet_archetypal_loss_elbow_strength'].std()

    # Thresholds for outliers (categories B & C)
    recon_outlier_threshold = recon_mean + sd_outlier * recon_std
    arch_outlier_threshold = arch_mean + sd_outlier * arch_std

    # Thresholds for strong values (categories A & D)
    recon_strong_threshold = recon_mean + sd_strong * recon_std
    arch_strong_threshold = arch_mean + sd_strong * arch_std

    selected_clusters = set()
    category_stats = {
        'A_strong_both': [],
        'B_recon_outliers': [],
        'C_arch_outliers': [],
        'D_agreement': []
    }

    # Category A: Strong on Both Axes
    cat_a = group[
        (group['aanet_recon_loss_elbow_strength'] > recon_strong_threshold) &
        (group['aanet_archetypal_loss_elbow_strength'] > arch_strong_threshold)
    ].copy()
    category_stats['A_strong_both'] = cat_a['cluster_id'].tolist()
    selected_clusters.update(cat_a['cluster_id'])

    # Category B: Reconstruction Outliers
    cat_b = group[
        group['aanet_recon_loss_elbow_strength'] > recon_outlier_threshold
    ].copy()
    category_stats['B_recon_outliers'] = cat_b['cluster_id'].tolist()
    selected_clusters.update(cat_b['cluster_id'])

    # Category C: Archetypal Outliers
    cat_c = group[
        group['aanet_archetypal_loss_elbow_strength'] > arch_outlier_threshold
    ].copy()
    category_stats['C_arch_outliers'] = cat_c['cluster_id'].tolist()
    selected_clusters.update(cat_c['cluster_id'])

    # Category D: Perfect Agreement Standouts
    # Delta k = 0 AND both metrics above their means
    cat_d = group[
        (group['k_differential'] == 0) &
        (((group['aanet_recon_loss_elbow_strength'] > recon_mean) & (group['aanet_archetypal_loss_elbow_strength'] > arch_mean)) |
         ((group['aanet_archetypal_loss_elbow_strength'] > arch_mean) & (group['aanet_recon_loss_elbow_strength'] > recon_mean)))
    ].copy()
    category_stats['D_agreement'] = cat_d['cluster_id'].tolist()
    selected_clusters.update(cat_d['cluster_id'])

    return selected_clusters, category_stats


def load_selected_clusters(corrected_csv_dir, n_clusters_list):
    """
    Load selected clusters using the same logic as analyze_aanet_results.py

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

        # Load corrected CSV
        csv_path = Path(corrected_csv_dir) / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}_corrected.csv"
        if not csv_path.exists():
            print(f"WARNING: CSV not found at {csv_path}, skipping")
            continue

        df = pd.read_csv(csv_path)
        df['n_clusters_total'] = n_clusters

        # Calculate elbow metrics (same as in plot_cluster_selection.py)
        def calculate_elbow_score(x, y):
            if len(x) < 3:
                return None, 0.0
            x_norm = (np.array(x) - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
            y_norm = (np.array(y) - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else np.zeros_like(y)
            distances = np.abs(x_norm + y_norm - 1) / np.sqrt(2)
            elbow_idx = np.argmax(distances)
            return x[elbow_idx], distances[elbow_idx]

        elbow_results = []
        for cluster_id, group in df.groupby('cluster_id'):
            group = group.sort_values('aanet_k')
            if len(group) < 3:
                continue

            k_vals = group['aanet_k'].values
            recon_losses = group['aanet_recon_loss'].values
            arch_losses = group['aanet_archetypal_loss'].values

            recon_k, recon_strength = calculate_elbow_score(k_vals, recon_losses)
            arch_k, arch_strength = calculate_elbow_score(k_vals, arch_losses)
            k_diff = abs(recon_k - arch_k) if (recon_k and arch_k) else np.nan

            elbow_results.append({
                'n_clusters_total': n_clusters,
                'cluster_id': cluster_id,
                'n_latents': group['n_latents'].iloc[0],
                'latent_indices': group['latent_indices'].iloc[0],
                'aanet_recon_loss_elbow_k': recon_k,
                'aanet_recon_loss_elbow_strength': recon_strength,
                'aanet_archetypal_loss_elbow_k': arch_k,
                'aanet_archetypal_loss_elbow_strength': arch_strength,
                'k_differential': k_diff,
                'recon_is_monotonic': group['recon_is_monotonic'].iloc[0],
                'arch_is_monotonic': group['arch_is_monotonic'].iloc[0],
                'recon_pct_decrease': group['recon_pct_decrease'].iloc[0],
                'arch_pct_decrease': group['arch_pct_decrease'].iloc[0],
            })

        elbow_df = pd.DataFrame(elbow_results)

        # Select clusters using same criteria
        selected_ids, cat_stats = select_promising_clusters(elbow_df, n_clusters)

        print(f"  Selected {len(selected_ids)} clusters")

        # Extract cluster info
        for cluster_id in selected_ids:
            row = elbow_df[elbow_df['cluster_id'] == cluster_id].iloc[0]

            # Determine category (priority: A, D, B, C)
            category = None
            if cluster_id in cat_stats['A_strong_both']:
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

            selected_clusters.append({
                'n_clusters': n_clusters,
                'cluster_id': int(cluster_id),
                'latent_indices': latent_indices,
                'arch_elbow_k': int(row['aanet_archetypal_loss_elbow_k']),
                'recon_elbow_k': int(row['aanet_recon_loss_elbow_k']),
                'category': category,
                'recon_elbow_strength': float(row['aanet_recon_loss_elbow_strength']),
                'arch_elbow_strength': float(row['aanet_archetypal_loss_elbow_strength']),
                'n_latents': int(row['n_latents']),
            })

    return selected_clusters


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

    # Training loop
    metrics_history = {
        "loss": [],
        "reconstruction_loss": [],
        "archetypal_loss": [],
        "extrema_loss": []
    }

    print(f"  Starting training ({args.aanet_streaming_steps} steps)...")
    for step in tqdm(range(args.aanet_streaming_steps), desc=f"Training"):
        # Sample batch
        tokens = sampler.sample_tokens_batch(args.aanet_batch_size, args.activity_seq_len, args.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
            acts = cache[hook_name]
            acts_flat = acts.reshape(-1, acts.shape[-1])
            feature_acts, _, _ = sae_encode_features(sae, acts_flat)

        # Train step
        step_losses = trainer.train_step(feature_acts)

        # Record metrics
        if cluster_id in step_losses:
            for key, value in step_losses[cluster_id].items():
                metrics_history[key].append(value)

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
    model_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}.pt"
    samples_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}_vertex_samples.jsonl"
    stats_path = save_dir / f"cluster_{cluster_id}_k{k}_category{category}_vertex_stats.json"

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
    aanet.load_state_dict(torch.load(model_path))
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

    # Open JSONL file for incremental writing
    samples_file = open(samples_path, 'w')

    print(f"  Target: {args.samples_per_vertex} samples per vertex")
    print(f"  Distance threshold: {args.vertex_distance_threshold}")
    print(f"  Starting collection...")

    pbar = tqdm(desc=f"Collecting samples")

    try:
        while total_inputs_processed < args.max_inputs_per_cluster:
            # Check termination conditions
            min_samples = min(v["samples"] for v in vertex_stats.values())

            if min_samples >= args.samples_per_vertex:
                print(f"\n  ✓ All vertices reached target ({min_samples}/{args.samples_per_vertex})")
                break

            # Fail-safe: check if rarest vertex is too rare
            if total_inputs_processed > 0 and total_inputs_processed % args.vertex_save_interval == 0:
                min_ratio = min_samples / args.samples_per_vertex if args.samples_per_vertex > 0 else 0
                if min_ratio < args.min_vertex_ratio:
                    max_samples = max(v["samples"] for v in vertex_stats.values())
                    if max_samples >= args.samples_per_vertex:
                        print(f"\n  ⚠ Fail-safe triggered: rarest vertex has {min_samples} samples ({min_ratio:.1%} of target)")
                        print(f"     Other vertices have reached target. Stopping collection.")
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
                        # Stop collecting for this vertex if target reached
                        if vertex_stats[i]["samples"] >= args.samples_per_vertex:
                            continue

                        # Calculate batch and sequence position
                        batch_idx = idx // args.activity_seq_len
                        seq_idx = idx % args.activity_seq_len

                        # Get full sequence tokens
                        sequence_tokens = tokens[batch_idx].cpu().numpy()

                        # Decode full text
                        full_text = tokenizer.decode(sequence_tokens)

                        # Decode trigger token
                        trigger_token_id = sequence_tokens[seq_idx]
                        trigger_word = tokenizer.decode([trigger_token_id])

                        # Create sample record
                        sample = {
                            "vertex_id": int(i),
                            "distance_to_vertex": float(dist),
                            "full_text": full_text,
                            "trigger_token_index": int(seq_idx),
                            "trigger_word": trigger_word,
                            "sequence_position": f"token {seq_idx} of {args.activity_seq_len}"
                        }

                        # Write to file immediately
                        samples_file.write(json.dumps(sample) + '\n')

                        vertex_stats[i]["samples"] += 1
                        samples_saved += 1

                total_inputs_processed += acts_flat.shape[0]
                pbar.update(acts_flat.shape[0])
                pbar.set_postfix({
                    f"v{i}": vertex_stats[i]["samples"] for i in range(min(k, 5))
                })

            del tokens, cache, acts, acts_flat, feature_acts

    finally:
        samples_file.close()
        pbar.close()

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
        "vertex_stats": vertex_stats,
        "all_vertices_reached_target": all_reached,
        "collection_status": collection_status,
        "model_name": args.model_name,
        "tokenizer": str(type(tokenizer).__name__),
        "distance_threshold": args.vertex_distance_threshold,
    }

    print(f"\n  Saving stats to {stats_path}")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  ✓ Collected {samples_saved} samples across {k} vertices")
    print(f"     Samples per vertex: {[vertex_stats[i]['samples'] for i in range(k)]}")
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

    # Load selected clusters
    print("\n" + "="*80)
    print("LOADING SELECTED CLUSTERS")
    print("="*80)
    selected_clusters = load_selected_clusters(args.corrected_csv_dir, n_clusters_list)
    print(f"\nTotal selected clusters: {len(selected_clusters)}")

    # Check if we found any clusters
    if len(selected_clusters) == 0:
        print("\n" + "="*80)
        print("ERROR: No clusters selected! Check that corrected CSV files exist.")
        print("="*80)
        print(f"Expected location: {args.corrected_csv_dir}/clusters_*/consolidated_metrics_n*_corrected.csv")
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
    sampler = RealDataSampler(
        model,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        seed=args.seed,
        prefetch_size=args.aanet_prefetch_size
    )

    # Stage 1: Train clusters (or load existing)
    manifest_path = Path(args.save_dir) / "manifest.json"

    if args.skip_training:
        # Load existing manifest
        print("\n" + "="*80)
        print("SKIPPING STAGE 1: Loading existing manifest")
        print("="*80)

        if not manifest_path.exists():
            print(f"\nERROR: Cannot skip training - manifest not found at {manifest_path}")
            print("Run without --skip_training first to train the models.")
            return

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        all_metadata = manifest['clusters']
        print(f"Loaded {len(all_metadata)} trained clusters from manifest")

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
