#!/usr/bin/env python3
"""
Null baseline validation for AANet simplex fits.

Tests whether geometrically-clustered SAE latents produce better AANet fits
than random latent subsets of the same size. If real clusters significantly
beat random baselines, the geometric clustering is finding real structure.

Two modes:
  --build_buffer: Stream data through model + SAE, save activations to disk
  --run_baselines: Load buffer, train AANets on random subsets, compare to real

Usage:
    # Build activation buffer (requires GPU + model)
    python validation/null_baselines.py --build_buffer \
        --model_name gemma-2-9b \
        --sae_release gemma-scope-9b-pt-res-canonical \
        --sae_id layer_20/width_16k/average_l0_68 \
        --buffer_size 100000 \
        --output_dir outputs/validation/null_baselines \
        --device cuda

    # Run baselines (requires buffer on disk, GPU for AANet training)
    python validation/null_baselines.py --run_baselines \
        --buffer_path outputs/validation/null_baselines/activation_buffer.pt \
        --csv_dir outputs/real_data_analysis_canonical \
        --n_baselines 20 \
        --output_dir outputs/validation/null_baselines \
        --device cuda
"""

import os
import sys
import argparse
import json
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone

# Add project root and AAnet/ to path (AAnet_torch uses absolute imports internally)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "AAnet"))

from AAnet.AAnet_torch.models.AAnet_vanilla import AAnet_vanilla


# =============================================================================
# 1. Activation Buffer
# =============================================================================

def build_activation_buffer(args):
    """Stream data through model + SAE and save activations to disk."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    from real_data_utils import RealDataSampler
    from mess3_gmg_analysis_utils import sae_encode_features
    from cluster_selection import delete_special_tokens

    print("=" * 80)
    print("BUILDING ACTIVATION BUFFER")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model_name}")
    if args.hf_token:
        import huggingface_hub
        huggingface_hub.login(token=args.hf_token, add_to_git_credential=False)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        dtype=torch.float16,
    )

    # Load SAE
    print(f"Loading SAE: {args.sae_release}/{args.sae_id}")
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )[0]

    # Extract layer index for hook
    layer_idx = int(args.sae_id.split("/")[0].split("_")[1])
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    d_sae = sae.W_dec.shape[0]
    d_model = sae.W_dec.shape[1]
    print(f"  Layer: {layer_idx}, d_sae: {d_sae}, d_model: {d_model}")

    # Save decoder weights (needed for partial reconstruction during baseline training)
    sae_W_dec = sae.W_dec.detach().cpu()

    # Set up data sampler
    print(f"\nSetting up data sampler...")
    sampler = RealDataSampler(
        model,
        hf_dataset=args.hf_dataset,
        hf_subset_name=args.hf_subset_name,
        split=args.dataset_split,
        streaming=True,
        seed=args.seed,
        prefetch_size=args.prefetch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_doc_tokens=args.max_doc_tokens,
    )

    # Stream and collect activations
    print(f"\nCollecting {args.buffer_size} active token positions...")
    all_acts = []
    total_active = 0
    batch_count = 0

    with torch.no_grad():
        pbar = tqdm(total=args.buffer_size, desc="Active positions")
        while total_active < args.buffer_size:
            # Sample batch
            tokens = sampler.sample_tokens_batch(
                args.batch_size, args.seq_len, args.device
            )

            # Forward pass
            _, cache = model.run_with_cache(
                tokens, return_type=None, names_filter=[hook_name]
            )
            acts = cache[hook_name]
            acts_flat = acts.reshape(-1, acts.shape[-1])

            # SAE encode
            feature_acts, _, _ = sae_encode_features(sae, acts_flat)

            # Remove BOS tokens
            feature_acts = delete_special_tokens(
                feature_acts, tokens.shape[0], args.seq_len, args.device
            )

            # Filter to active positions (any latent fires)
            active_mask = feature_acts.abs().sum(dim=1) > 1e-6
            active_acts = feature_acts[active_mask]

            if active_acts.shape[0] > 0:
                # Store as float16 to save memory/disk
                all_acts.append(active_acts.cpu().half())
                total_active += active_acts.shape[0]
                pbar.update(active_acts.shape[0])

            batch_count += 1

            # Cleanup
            del tokens, cache, acts, acts_flat, feature_acts

        pbar.close()

    # Concatenate and trim to exact size
    buffer = torch.cat(all_acts, dim=0)[:args.buffer_size]
    print(f"\nBuffer shape: {buffer.shape}")
    print(f"  Batches processed: {batch_count}")
    print(f"  Sparsity: {(buffer == 0).float().mean():.4f}")

    # Save
    buffer_path = output_dir / "activation_buffer.pt"
    save_data = {
        "activations": buffer,
        "sae_W_dec": sae_W_dec,
        "metadata": {
            "buffer_size": buffer.shape[0],
            "d_sae": d_sae,
            "d_model": d_model,
            "model_name": args.model_name,
            "sae_release": args.sae_release,
            "sae_id": args.sae_id,
            "layer_idx": layer_idx,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "created": datetime.now(timezone.utc).isoformat(),
        },
    }
    print(f"\nSaving buffer to {buffer_path}")
    torch.save(save_data, buffer_path)

    file_size_mb = buffer_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # Cleanup model from GPU
    del model, sae, sampler
    torch.cuda.empty_cache()

    return buffer_path


def load_activation_buffer(buffer_path, device="cpu"):
    """Load a saved activation buffer."""
    print(f"Loading activation buffer from {buffer_path}")
    data = torch.load(buffer_path, map_location="cpu", weights_only=False)
    print(f"  Buffer shape: {data['activations'].shape}")
    print(f"  Created: {data['metadata']['created']}")
    return data


# =============================================================================
# 2. Real Cluster Metrics
# =============================================================================

def load_real_cluster_metrics(csv_dir, cluster_configs):
    """Load real cluster metrics from consolidated CSVs.

    Args:
        csv_dir: Directory containing consolidated_metrics_n{N}.csv files
        cluster_configs: List of dicts with keys: n_clusters, cluster_id, k

    Returns:
        Dict mapping cluster_key -> {recon_loss, archetypal_loss, latent_indices, ...}
    """
    import pandas as pd

    csv_dir = Path(csv_dir)
    results = {}

    for config in cluster_configs:
        n_clusters = config["n_clusters"]
        cluster_id = config["cluster_id"]
        k = config["k"]
        cluster_key = f"{n_clusters}_{cluster_id}"

        csv_path = csv_dir / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
        if not csv_path.exists():
            print(f"  WARNING: CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Filter to this cluster at the target k
        mask = (df["cluster_id"] == cluster_id) & (df["aanet_k"] == k)
        rows = df[mask]

        if len(rows) == 0:
            print(f"  WARNING: No row for cluster {cluster_key} at k={k}")
            continue

        row = rows.iloc[0]
        latent_indices = json.loads(row["latent_indices"])

        results[cluster_key] = {
            "n_clusters": n_clusters,
            "cluster_id": cluster_id,
            "k": k,
            "n_latents": len(latent_indices),
            "latent_indices": latent_indices,
            "recon_loss": row["aanet_recon_loss"],
            "archetypal_loss": row["aanet_archetypal_loss"],
            "total_loss": row["aanet_loss"],
        }
        print(f"  {cluster_key}: k={k}, n_latents={len(latent_indices)}, "
              f"recon={row['aanet_recon_loss']:.6f}, arch={row['aanet_archetypal_loss']:.6f}")

    return results


# =============================================================================
# 3. Random Index Sampling
# =============================================================================

def sample_random_indices_uniform(n_latents, d_sae, rng):
    """Sample n_latents random indices uniformly from [0, d_sae)."""
    return sorted(rng.sample(range(d_sae), n_latents))


def sample_random_indices_norm_matched(n_latents, d_sae, decoder_norms, real_indices, rng, n_bins=20):
    """Sample random indices matching the decoder norm distribution of the real cluster.

    Uses histogram matching: bin all decoder norms, then sample from each bin
    proportionally to the real cluster's bin counts.
    """
    real_norms = decoder_norms[real_indices]

    # Create bins from the full range of decoder norms
    all_norms_np = decoder_norms.numpy()
    bin_edges = np.histogram_bin_edges(all_norms_np, bins=n_bins)

    # Bin the real cluster's norms
    real_bin_counts, _ = np.histogram(real_norms.numpy(), bins=bin_edges)

    # For each bin, get the available indices (excluding real cluster indices)
    real_set = set(real_indices)
    bin_indices = []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            # Last bin includes right edge
            mask = (all_norms_np >= low) & (all_norms_np <= high)
        else:
            mask = (all_norms_np >= low) & (all_norms_np < high)
        available = [j for j in np.where(mask)[0] if j not in real_set]
        bin_indices.append(available)

    # Sample from each bin according to real cluster's distribution
    sampled = []
    for i in range(n_bins):
        count = real_bin_counts[i]
        available = bin_indices[i]
        if count == 0 or len(available) == 0:
            continue
        # If not enough indices in this bin, take all available
        n_sample = min(count, len(available))
        sampled.extend(rng.sample(available, n_sample))

    # If we got fewer than n_latents (due to sparse bins), fill uniformly
    if len(sampled) < n_latents:
        remaining = [j for j in range(d_sae) if j not in real_set and j not in set(sampled)]
        n_extra = n_latents - len(sampled)
        sampled.extend(rng.sample(remaining, min(n_extra, len(remaining))))

    # If we got more than n_latents (due to rounding), trim
    if len(sampled) > n_latents:
        sampled = rng.sample(sampled, n_latents)

    return sorted(sampled)


# =============================================================================
# 4. AANet Training from Buffer
# =============================================================================

def train_aanet_from_buffer(
    buffer_acts,
    sae_W_dec,
    indices,
    k,
    device,
    max_steps=3000,
    batch_size=128,
    lr=0.0025,
    weight_decay=1e-5,
    noise=0.05,
    layer_widths=(64, 32),
    simplex_scale=1.0,
    gamma_recon=1.0,
    gamma_arch=4.0,
    grad_clip=1.0,
    lr_patience=50,
    lr_factor=0.5,
    min_lr=1e-6,
    early_stop_patience=500,
    min_delta=1e-6,
    smoothing_window=20,
    active_threshold=1e-6,
    verbose=False,
):
    """Train an AANet on pre-computed activations for a given set of latent indices.

    Returns dict with final metrics.
    """
    indices_t = torch.tensor(indices, dtype=torch.long)
    d_model = sae_W_dec.shape[1]

    # Slice cluster activations and compute partial reconstruction
    acts_c = buffer_acts[:, indices_t]  # (N, n_latents)
    active_mask = acts_c.abs().sum(dim=1) > active_threshold
    acts_c_active = acts_c[active_mask]  # (N_active, n_latents)

    n_active = acts_c_active.shape[0]
    if n_active < batch_size:
        return {
            "recon_loss": float("nan"),
            "archetypal_loss": float("nan"),
            "total_loss": float("nan"),
            "steps_trained": 0,
            "n_active": n_active,
            "status": "insufficient_samples",
        }

    # Partial reconstruction
    W_c = sae_W_dec[indices_t, :]  # (n_latents, d_model)
    X_recon = (acts_c_active.float() @ W_c.float()).to(device)  # (N_active, d_model)

    # Create model
    model = AAnet_vanilla(
        input_shape=d_model,
        n_archetypes=k,
        noise=noise,
        layer_widths=list(layer_widths),
        simplex_scale=simplex_scale,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    # Training loop
    loss_history = []
    best_loss = float("inf")
    steps_since_improvement = 0
    check_interval = (smoothing_window // 2) + (smoothing_window // 4)

    n_samples = X_recon.shape[0]
    perm = torch.randperm(n_samples, device=device)
    ptr = 0

    for step in range(max_steps):
        # Get batch (cycle through shuffled buffer)
        if ptr + batch_size > n_samples:
            perm = torch.randperm(n_samples, device=device)
            ptr = 0

        batch_idx = perm[ptr : ptr + batch_size]
        batch = X_recon[batch_idx]
        ptr += batch_size

        # Forward + backward
        optimizer.zero_grad()
        recon, _, embedding = model(batch)
        loss, metrics = model.loss_function(
            batch,
            recon,
            embedding,
            gamma_reconstruction=gamma_recon,
            gamma_archetypal=gamma_arch,
            gamma_extrema=0.0,
        )
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        loss_val = loss.item()
        loss_history.append({
            "loss": loss_val,
            "recon": metrics["reconstruction_loss"],
            "arch": metrics["archetypal_loss"],
        })

        # LR scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr > min_lr:
            scheduler.step(loss_val)

        # Early stopping check
        if step % check_interval == 0 and len(loss_history) >= smoothing_window:
            smoothed = np.mean([h["loss"] for h in loss_history[-smoothing_window:]])
            if smoothed < best_loss - min_delta:
                best_loss = smoothed
                steps_since_improvement = -1

        steps_since_improvement += 1
        if steps_since_improvement >= early_stop_patience:
            if verbose:
                print(f"    Early stop at step {step}")
            break

    # Final metrics (median of last 20)
    n_final = min(20, len(loss_history))
    final_recon = float(np.median([h["recon"] for h in loss_history[-n_final:]]))
    final_arch = float(np.median([h["arch"] for h in loss_history[-n_final:]]))
    final_loss = float(np.median([h["loss"] for h in loss_history[-n_final:]]))

    # Cleanup
    del model, optimizer, scheduler, X_recon
    torch.cuda.empty_cache()

    return {
        "recon_loss": final_recon,
        "archetypal_loss": final_arch,
        "total_loss": final_loss,
        "steps_trained": step + 1,
        "n_active": n_active,
        "status": "ok",
    }


# =============================================================================
# 5. Run Null Baselines
# =============================================================================

def run_null_baselines(buffer_data, cluster_configs, real_metrics, args):
    """Run null baseline comparisons for all clusters."""
    import random

    buffer_acts = buffer_data["activations"]
    sae_W_dec = buffer_data["sae_W_dec"]
    d_sae = buffer_acts.shape[1]

    # Compute decoder norms for norm-matched sampling
    decoder_norms = sae_W_dec.norm(dim=1)  # (d_sae,)

    results = {}
    rng = random.Random(args.seed)

    for config in cluster_configs:
        cluster_key = f"{config['n_clusters']}_{config['cluster_id']}"
        if cluster_key not in real_metrics:
            print(f"\nSkipping {cluster_key}: no real metrics found")
            continue

        real = real_metrics[cluster_key]
        n_latents = real["n_latents"]
        k = real["k"]
        real_indices = real["latent_indices"]

        print(f"\n{'=' * 60}")
        print(f"CLUSTER {cluster_key} (n_latents={n_latents}, k={k})")
        print(f"  Real recon_loss={real['recon_loss']:.6f}, arch_loss={real['archetypal_loss']:.6f}")
        print(f"{'=' * 60}")

        cluster_results = {
            "cluster_key": cluster_key,
            "n_latents": n_latents,
            "k": k,
            "real_metrics": {
                "recon_loss": real["recon_loss"],
                "archetypal_loss": real["archetypal_loss"],
                "total_loss": real["total_loss"],
            },
            "baselines": {},
        }

        for strategy in ["uniform", "norm_matched"]:
            print(f"\n  Strategy: {strategy}")
            baseline_metrics = []

            for i in tqdm(range(args.n_baselines), desc=f"  {strategy}"):
                # Generate random indices
                if strategy == "uniform":
                    rand_indices = sample_random_indices_uniform(n_latents, d_sae, rng)
                else:
                    rand_indices = sample_random_indices_norm_matched(
                        n_latents, d_sae, decoder_norms, real_indices, rng
                    )

                # Train AANet
                metrics = train_aanet_from_buffer(
                    buffer_acts,
                    sae_W_dec,
                    rand_indices,
                    k,
                    args.device,
                    max_steps=args.max_steps,
                    batch_size=args.train_batch_size,
                )

                baseline_metrics.append(metrics)

                if metrics["status"] != "ok":
                    tqdm.write(f"    Baseline {i}: {metrics['status']}")

            # Compute statistics
            ok_metrics = [m for m in baseline_metrics if m["status"] == "ok"]
            if ok_metrics:
                recon_losses = [m["recon_loss"] for m in ok_metrics]
                arch_losses = [m["archetypal_loss"] for m in ok_metrics]

                recon_mean = np.mean(recon_losses)
                recon_std = np.std(recon_losses)
                arch_mean = np.mean(arch_losses)
                arch_std = np.std(arch_losses)

                # Z-scores (negative = real is better)
                recon_z = (real["recon_loss"] - recon_mean) / recon_std if recon_std > 0 else 0
                arch_z = (real["archetypal_loss"] - arch_mean) / arch_std if arch_std > 0 else 0

                # Percentile (what fraction of baselines did the real cluster beat)
                recon_pct = np.mean([r > real["recon_loss"] for r in recon_losses]) * 100
                arch_pct = np.mean([a > real["archetypal_loss"] for a in arch_losses]) * 100

                cluster_results["baselines"][strategy] = {
                    "n_ok": len(ok_metrics),
                    "n_failed": len(baseline_metrics) - len(ok_metrics),
                    "recon_losses": recon_losses,
                    "arch_losses": arch_losses,
                    "recon_mean": recon_mean,
                    "recon_std": recon_std,
                    "recon_z_score": recon_z,
                    "recon_percentile_beaten": recon_pct,
                    "arch_mean": arch_mean,
                    "arch_std": arch_std,
                    "arch_z_score": arch_z,
                    "arch_percentile_beaten": arch_pct,
                }

                print(f"    Recon: real={real['recon_loss']:.6f} vs null={recon_mean:.6f}±{recon_std:.6f} "
                      f"(z={recon_z:.2f}, beats {recon_pct:.0f}%)")
                print(f"    Arch:  real={real['archetypal_loss']:.6f} vs null={arch_mean:.6f}±{arch_std:.6f} "
                      f"(z={arch_z:.2f}, beats {arch_pct:.0f}%)")
            else:
                print(f"    All baselines failed!")
                cluster_results["baselines"][strategy] = {"n_ok": 0, "n_failed": len(baseline_metrics)}

        results[cluster_key] = cluster_results

    return results


# =============================================================================
# 6. Summary and Output
# =============================================================================

def summarize_results(results, output_dir):
    """Print summary table and save detailed results."""
    print("\n" + "=" * 100)
    print("NULL BASELINE RESULTS")
    print("=" * 100)

    for strategy in ["uniform", "norm_matched"]:
        print(f"\n--- Strategy: {strategy} ---")
        print(f"{'Cluster':<12} {'n_lat':>5} {'k':>3} {'Real Recon':>11} "
              f"{'Null Recon (mean±std)':>22} {'Z-score':>8} {'Beat %':>7} | "
              f"{'Real Arch':>10} {'Null Arch (mean±std)':>22} {'Z-score':>8} {'Beat %':>7}")
        print("-" * 130)

        for cluster_key, data in sorted(results.items()):
            if strategy not in data["baselines"]:
                continue
            b = data["baselines"][strategy]
            if b["n_ok"] == 0:
                continue

            real_r = data["real_metrics"]["recon_loss"]
            real_a = data["real_metrics"]["archetypal_loss"]

            print(f"{cluster_key:<12} {data['n_latents']:>5} {data['k']:>3} "
                  f"{real_r:>11.6f} {b['recon_mean']:>10.6f}±{b['recon_std']:<9.6f} "
                  f"{b['recon_z_score']:>8.2f} {b['recon_percentile_beaten']:>6.0f}% | "
                  f"{real_a:>10.6f} {b['arch_mean']:>10.6f}±{b['arch_std']:<9.6f} "
                  f"{b['arch_z_score']:>8.2f} {b['arch_percentile_beaten']:>6.0f}%")

    # Save detailed JSON
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "null_baseline_results.json"

    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


# =============================================================================
# 7. Main
# =============================================================================

# Default cluster configurations (our 4 priority + 3 additional clusters)
DEFAULT_CLUSTER_CONFIGS = [
    # Priority clusters
    {"n_clusters": 512, "cluster_id": 292, "k": 3},
    {"n_clusters": 512, "cluster_id": 464, "k": 5},
    {"n_clusters": 512, "cluster_id": 504, "k": 5},
    {"n_clusters": 768, "cluster_id": 484, "k": 3},
    # Additional clusters
    {"n_clusters": 512, "cluster_id": 261, "k": 3},
    {"n_clusters": 768, "cluster_id": 210, "k": 5},
    {"n_clusters": 768, "cluster_id": 455, "k": 4},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Null baseline validation for AANet simplex fits")

    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build_buffer", action="store_true", help="Build and save activation buffer")
    mode.add_argument("--run_baselines", action="store_true", help="Run baseline comparisons")

    # Buffer building args
    parser.add_argument("--model_name", type=str, default="gemma-2-9b")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_16k/canonical")  # formerly "layer_20/width_16k/average_l0_68"
    parser.add_argument("--buffer_size", type=int, default=100_000,
                        help="Number of active token positions to collect")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for model forward passes")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--hf_dataset", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--hf_subset_name", type=str, default="sample-10BT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--prefetch_size", type=int, default=1024)
    parser.add_argument("--shuffle_buffer_size", type=int, default=50000)
    parser.add_argument("--max_doc_tokens", type=int, default=3000)

    # Baseline running args
    parser.add_argument("--buffer_path", type=str, default=None,
                        help="Path to saved activation buffer")
    parser.add_argument("--csv_dir", type=str, default=None,
                        help="Directory with consolidated_metrics CSVs")
    parser.add_argument("--n_baselines", type=int, default=20,
                        help="Number of random baselines per cluster per strategy")
    parser.add_argument("--cluster_configs", type=str, default=None,
                        help="JSON string of cluster configs (overrides defaults)")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Max training steps per AANet")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Batch size for AANet training")

    # Common
    parser.add_argument("--output_dir", type=str, default="outputs/validation/null_baselines")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Default: run both stages
    if not args.build_buffer and not args.run_baselines:
        args.build_buffer = True
        args.run_baselines = True

    # Parse cluster configs
    if args.cluster_configs:
        cluster_configs = json.loads(args.cluster_configs)
    else:
        cluster_configs = DEFAULT_CLUSTER_CONFIGS

    buffer_path = args.buffer_path

    # Stage 1: Build buffer
    if args.build_buffer:
        buffer_path = str(build_activation_buffer(args))

    # Stage 2: Run baselines
    if args.run_baselines:
        if buffer_path is None:
            buffer_path = str(Path(args.output_dir) / "activation_buffer.pt")

        if not Path(buffer_path).exists():
            print(f"ERROR: Buffer not found at {buffer_path}")
            print("Run with --build_buffer first, or provide --buffer_path")
            sys.exit(1)

        if args.csv_dir is None:
            print("ERROR: Must provide --csv_dir for baseline comparisons")
            sys.exit(1)

        # Load buffer
        buffer_data = load_activation_buffer(buffer_path)

        # Load real cluster metrics
        print("\nLoading real cluster metrics...")
        real_metrics = load_real_cluster_metrics(args.csv_dir, cluster_configs)

        if not real_metrics:
            print("ERROR: No real cluster metrics loaded")
            sys.exit(1)

        # Run baselines
        results = run_null_baselines(buffer_data, cluster_configs, real_metrics, args)

        # Summarize
        summarize_results(results, args.output_dir)


if __name__ == "__main__":
    main()
