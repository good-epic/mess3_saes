
import os
import gc
from tqdm import tqdm
import argparse
import json
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from clustering.config import ClusteringConfig, CooccurrenceConfig
from clustering.config import GeometryFittingConfig, SubspaceParams, SpectralParams, SamplingConfig
from aanet_pipeline.extrema import compute_diffusion_extrema, ExtremaConfig
from aanet_pipeline.training import train_aanet_model, TrainingConfig
from aanet_pipeline.cluster_summary import AAnetDescriptor
import jax
from huggingface_hub import login

import sys
sys.stdout.reconfigure(line_buffering=True)

from real_data_utils import RealDataSampler, build_real_aanet_datasets
from real_data_tests.real_pipeline import RealDataClusteringPipeline
from aanet_pipeline.streaming_trainer import StreamingAAnetTrainer
from mess3_gmg_analysis_utils import sae_encode_features
from cluster_selection import delete_special_tokens


def compute_cluster_activation_pca_ranks(
    model,
    sae,
    sampler,
    clustering_result,
    hook_name,
    num_cycles=80,
    batch_size=256,
    seq_len=128,
    max_samples=100000,
    variance_threshold=0.95,
    device='cuda'
):
    """
    For each cluster, sample activations across diverse batches and compute practical PCA rank.

    Args:
        model: HookedTransformer model
        sae: Trained SAE
        sampler: RealDataSampler for getting diverse batches
        clustering_result: ClusteringResult with cluster_labels
        hook_name: Hook name for model.run_with_cache
        num_cycles: Number of sampling cycles (each cycle samples batch_size sequences). Default 80 → ~5M token positions before filtering
        batch_size: Batch size for forward passes (hardcoded to 256 for memory safety)
        seq_len: Sequence length (default 128)
        max_samples: Maximum samples for PCA (randomly subsample if exceeded). Limits computation/memory.
        variance_threshold: Variance explained threshold for practical rank (default 0.95)
        device: Device to run on

    Returns:
        dict mapping cluster_id -> {"avg_rank": float, "avg_variance": float}
    """
    from sklearn.decomposition import PCA

    cluster_labels = clustering_result.cluster_labels
    n_clusters = clustering_result.n_clusters

    # Group latent indices by cluster
    cluster_to_latents = {}
    for latent_idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in cluster_to_latents:
            cluster_to_latents[cluster_id] = []
        cluster_to_latents[cluster_id].append(latent_idx)

    pca_results = {}

    for cluster_id in tqdm(range(n_clusters), desc="Computing PCA ranks"):
        if cluster_id not in cluster_to_latents:
            continue

        latent_indices = cluster_to_latents[cluster_id]
        if len(latent_indices) < 2:
            # Can't do PCA on single latent
            pca_results[cluster_id] = {"avg_rank": 1, "avg_variance": 1.0}
            continue

        # Collect activations from num_cycles diverse batches
        all_activations = []

        for cycle_idx in range(num_cycles):
            # Sample a batch of tokens
            tokens = sampler.sample_tokens_batch(batch_size, seq_len, device)

            with torch.no_grad():
                # Forward pass through model
                _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                acts = cache[hook_name]  # (batch, seq, d_model)
                acts_flat = acts.reshape(-1, acts.shape[-1])  # (batch*seq, d_model)

                # Encode with SAE
                feature_acts, _, _ = sae_encode_features(sae, acts_flat)  # (batch*seq, d_sae)

                # Exclude BOS tokens (position 0) - not meaningful for analysis
                feature_acts = delete_special_tokens(feature_acts, batch_size, seq_len, device)

                # Filter to only this cluster's latents
                cluster_acts = feature_acts[:, latent_indices]  # (batch*seq-batch, n_cluster_latents)

                # Only keep rows where at least one latent is active
                active_mask = (cluster_acts.abs().sum(dim=1) > 1e-5)
                if active_mask.any():
                    cluster_acts_active = cluster_acts[active_mask].cpu().numpy()
                    all_activations.append(cluster_acts_active)

            # Clear CUDA cache and force garbage collection after each cycle to avoid memory accumulation
            torch.cuda.empty_cache()
            gc.collect()

        if not all_activations:
            # No active samples found
            print(f"  WARNING: Cluster {cluster_id} - No active samples found after {num_cycles} cycles!")
            pca_results[cluster_id] = {"avg_rank": 0, "avg_variance": 0.0}
            continue

        # Concatenate all samples
        X = np.vstack(all_activations)  # (n_total_samples, n_cluster_latents)
        n_samples_collected = X.shape[0]
        n_latents = X.shape[1]

        # Calculate filtering efficiency
        total_possible = num_cycles * batch_size * seq_len
        efficiency = 100 * n_samples_collected / total_possible

        # Subsample if we collected more than max_samples
        if n_samples_collected > max_samples:
            # Randomly subsample to max_samples
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            subsample_indices = rng.choice(n_samples_collected, size=max_samples, replace=False)
            X = X[subsample_indices]
            print(f"  Cluster {cluster_id}: Subsampled {n_samples_collected:,} → {max_samples:,} samples for PCA")
            n_samples_used = max_samples
        else:
            n_samples_used = n_samples_collected

        # Warn if we got very few samples
        if n_samples_collected < 1000:
            print(f"  WARNING: Cluster {cluster_id} - Only {n_samples_collected:,} samples collected "
                  f"({efficiency:.1f}% hit rate) for {n_latents} latents. Consider increasing --pca_num_cycles")
        elif n_samples_collected < n_latents * 2:
            print(f"  WARNING: Cluster {cluster_id} - {n_samples_collected:,} samples < 2× latents ({n_latents}). "
                  f"PCA may be unreliable.")

        if X.shape[0] < 2:
            pca_results[cluster_id] = {"avg_rank": 1, "avg_variance": 1.0}
            continue

        # Fit PCA
        n_components = min(X.shape[0], X.shape[1])  # Can't have more components than samples or features
        pca = PCA(n_components=n_components)
        pca.fit(X)

        # Find practical rank: number of components to explain variance_threshold of variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        practical_rank = np.searchsorted(cumsum_variance, variance_threshold) + 1
        practical_rank = min(practical_rank, n_components)  # Cap at max components

        variance_explained = cumsum_variance[practical_rank - 1] if practical_rank > 0 else 0.0

        pca_results[cluster_id] = {
            "avg_rank": int(practical_rank),
            "avg_variance": float(variance_explained),
            "n_samples_collected": n_samples_collected,
            "n_samples_used": n_samples_used,
            "n_latents": n_latents
        }
        torch.cuda.empty_cache()
        gc.collect()

    # Print summary statistics
    if pca_results:
        ranks = [r["avg_rank"] for r in pca_results.values() if r["avg_rank"] > 0]
        samples_collected = [r.get("n_samples_collected", 0) for r in pca_results.values()]
        samples_used = [r.get("n_samples_used", 0) for r in pca_results.values()]
        if ranks and samples_collected:
            print(f"\n  PCA Summary: {len(pca_results)} clusters")
            print(f"    Rank range: {min(ranks)} - {max(ranks)}, median: {np.median(ranks):.0f}")
            print(f"    Samples collected: {min(samples_collected):,} - {max(samples_collected):,}, median: {np.median(samples_collected):,.0f}")
            if any(s > max_samples for s in samples_collected):
                print(f"    Samples used (after subsampling): {min(samples_used):,} - {max(samples_used):,}, median: {np.median(samples_used):,.0f}")

    return pca_results


def main():
    parser = argparse.ArgumentParser(description="Analyze Real SAEs with Clustering and Simplex Fitting")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-9b-pt-res", help="SAE Lens release name")
    parser.add_argument("--sae_id", type=str, default="layer_20/width_32k/average_l0_57", help="SAE ID within release")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for authentication")


        # Options for layer 20 for Gemma 2 9b:
        #
        # width_16k
        # average_l0_11
        # average_l0_138
        # average_l0_20
        # average_l0_310
        # average_l0_36
        # average_l0_408
        # average_l0_58
        # average_l0_68

        # width_32k
        # average_l0_11
        # average_l0_344
        # average_l0_57

        # width_65k
        # average_l0_11
        # average_l0_298
        # average_l0_55

        # width_131k
        # average_l0_11
        # average_l0_114
        # average_l0_19
        # average_l0_221
        # average_l0_276
        # average_l0_34
        # average_l0_53
        # average_l0_62

        # width_262k
        # average_l0_11
        # average_l0_259
        # average_l0_50

        # width_524k
        # average_l0_10
        # average_l0_241
        # average_l0_48

        # width_1m
        # average_l0_101
        # average_l0_19
        # average_l0_193
        # average_l0_34
        # average_l0_57
        # average_l0_57
    parser.add_argument("--model_name", type=str, default="gemma-2-9b", help="TransformerLens model name")
    parser.add_argument("--output_dir", type=str, default="outputs/real_data_analysis", help="Output directory")
    parser.add_argument("--n_clusters_list", type=int, nargs="+", default=[128, 256, 512, 645], help="List of cluster counts to try")
    parser.add_argument("--total_samples", type=int, default=25000, help="Total samples for clustering")
    parser.add_argument("--latent_activity_threshold", type=float, default=1e-5, help="Minimum activation rate for latents")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for Hugging Face cache")

    # Data streaming
    parser.add_argument("--hf_dataset", type=str, default="HuggingFaceFW/fineweb", help="HuggingFace dataset identifier")
    parser.add_argument("--hf_subset_name", type=str, default="sample-10BT", help="Dataset subset/config name")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--aanet_prefetch_size", type=int, default=1024, help="Prefetch buffer size for RealDataSampler")
    parser.add_argument("--shuffle_buffer_size", type=int, default=50000, help="Shuffle buffer size for streaming dataset")
    parser.add_argument("--max_doc_tokens", type=int, default=3000, help="Filter documents longer than this (approximate, uses ~4 chars/token)")
    parser.add_argument("--activity_batch_size", type=int, default=16, help="Batch size for activity stats")
    parser.add_argument("--activity_batches", type=int, default=1024, help="Number of batches for activity stats")
    parser.add_argument("--activity_seq_len", type=int, default=128, help="Sequence length for activity stats")

    # Clustering method and similarity metric
    parser.add_argument("--method", type=str, default="k_subspaces",
                        choices=["k_subspaces", "spectral"],
                        help="Clustering method. k_subspaces uses decoder geometry. "
                             "spectral uses an affinity matrix (from --sim_metric).")
    parser.add_argument("--sim_metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "jaccard", "dice", "overlap", "phi", "mutual_info", "ami"],
                        help="Similarity metric for spectral clustering. Geometry-based: cosine, euclidean. "
                             "Co-occurrence-based: jaccard, dice, overlap, phi (|phi|), mutual_info, ami (|PMI|)")

    # Co-occurrence statistics collection (for jaccard, dice, overlap, phi, mutual_info metrics)
    parser.add_argument("--cooc_n_batches", type=int, default=1000,
                        help="Number of batches for co-occurrence collection. Total tokens = n_batches * batch_size * seq_len")
    parser.add_argument("--cooc_batch_size", type=int, default=32,
                        help="Sequences per batch for co-occurrence collection")
    parser.add_argument("--cooc_seq_len", type=int, default=256,
                        help="Tokens per sequence for co-occurrence collection")
    parser.add_argument("--cooc_activation_threshold", type=float, default=1e-6,
                        help="Feature counts as firing if |activation| > threshold")
    parser.add_argument("--cooc_skip_special_tokens", action="store_true", default=True,
                        help="Skip BOS token (position 0) when collecting co-occurrence")
    parser.add_argument("--cooc_cache_path", type=str, default=None,
                        help="Path to save/load co-occurrence stats (avoids recomputation when trying different metrics)")

    # PCA rank estimation parameters
    parser.add_argument("--pca_num_cycles", type=int, default=80,
                       help="Number of data sampling cycles for PCA rank estimation. "
                            "Each cycle: 256 sequences × seq_len tokens = 256 × 256 = 65,536 token positions. "
                            "Default 80 cycles = 5.24M token positions sampled (before sparsity filtering). "
                            "After filtering (~30-50%% hit rate) → ~2M samples per cluster.")
    parser.add_argument("--pca_max_samples", type=int, default=100000,
                       help="Maximum number of samples to use for PCA (randomly subsampled if more collected). "
                            "Limits computational cost and memory usage. Default 100K is good for up to ~1000-D.")
    parser.add_argument("--pca_variance_threshold", type=float, default=0.95,
                       help="Variance threshold for PCA rank estimation (fraction of variance to explain)")
    parser.add_argument("--skip_pca", action="store_true",
                       help="Skip PCA rank estimation entirely (saves significant time)")
    parser.add_argument("--clustering_only", action="store_true",
                       help="Only run clustering (and PCA if enabled), skip AANet fitting. "
                            "Useful for comparing clustering methods before committing to expensive AANet training.")
    parser.add_argument("--subspace_variance_threshold", type=float, default=0.95, help="Variance threshold for rank estimation")
    parser.add_argument("--subspace_gap_threshold", type=float, default=2.0, help="Eigengap threshold for rank estimation")

    # AAnet Arguments
    parser.add_argument("--aanet_epochs", type=int, default=100, help="Training epochs per model.")
    parser.add_argument("--aanet_batch_size", type=int, default=256, help="Batch size for AAnet training.")
    parser.add_argument("--aanet_lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--aanet_weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--aanet_layer_widths", type=int, nargs="+", default=[256, 128], help="Hidden layer widths.")
    parser.add_argument("--aanet_simplex_scale", type=float, default=1.0, help="Simplex scale.")
    parser.add_argument("--aanet_noise", type=float, default=0.05, help="Latent noise value or scale.")
    parser.add_argument("--aanet_noise_relative", action="store_true", help="Interpret --aanet-noise as a multiple of the dataset std.")
    parser.add_argument("--aanet_gamma_reconstruction", type=float, default=1.0)
    parser.add_argument("--aanet_gamma_archetypal", type=float, default=1.0)
    parser.add_argument("--aanet_gamma_extrema", type=float, default=1.0)
    parser.add_argument("--aanet_min_samples", type=int, default=32, help="Minimum dataset size before training.")
    parser.add_argument("--aanet_num_workers", type=int, default=0, help="DataLoader workers for AAnet training.")
    parser.add_argument("--aanet_seed", type=int, default=43, help="Base seed for AAnet training.")
    parser.add_argument("--aanet_val_fraction", type=float, default=0.1, help="Fraction of samples reserved for validation per cluster.")
    parser.add_argument("--aanet_val_min_size", type=int, default=256, help="Minimum number of samples required for a validation split.")

    # Early stopping and LR scheduling parameters
    parser.add_argument("--aanet_lr_patience", type=int, default=30, help="Reduce LR after this many steps without improvement")
    parser.add_argument("--aanet_lr_factor", type=float, default=0.5, help="Factor to reduce learning rate when plateau is detected.")
    parser.add_argument("--aanet_min_lr", type=float, default=1e-6, help="Don't reduce LR below this value")
    parser.add_argument("--aanet_early_stop_patience", type=int, default=250, help="Stop training after this many steps without improvement")
    parser.add_argument("--aanet_min_delta", type=float, default=1e-6, help="Minimum change in loss to count as improvement")
    parser.add_argument("--aanet_loss_smoothing_window",
                       type=lambda x: int(x) if int(x) > 7 else parser.error("aanet_loss_smoothing_window must be at least 8"),
                       default=20, help="Window size for smoothing loss before early stopping comparison")
    parser.add_argument("--aanet_grad_clip", type=float, default=1.0, help="Gradient clipping norm (set <=0 to disable).")
    parser.add_argument("--aanet_restarts_no_extrema", type=int, default=3, help="Number of random restarts when no warm-start extrema are available.")
    parser.add_argument("--aanet_streaming_steps", type=int, default=1000, help="Number of streaming steps (batches) for AAnet training.")
    parser.add_argument("--aanet_warmup_steps", type=int, default=50, help="Number of batches to collect for extrema initialization.")
    parser.add_argument("--aanet_k_min", type=int, default=2, help="Minimum k for AAnet.")
    parser.add_argument("--aanet_k_max", type=int, default=8, help="Maximum k for AAnet.")
    parser.add_argument("--aanet_warmup_max_per_cluster", type=int, default=1000, help="Max samples per cluster for extrema finding.")
    parser.add_argument("--aanet_warmup_cluster_chunk_size", type=int, default=16, help="Process clusters in chunks of this size during warmup to save memory.")
    parser.add_argument("--aanet_active_threshold", type=float, default=1e-6, help="Threshold for active samples in AAnet training.")
    parser.add_argument("--aanet_sequential_k", action="store_true", help="Train AAnet for each k sequentially (saves VRAM) instead of concurrently.")

    # Extrema Arguments
    parser.add_argument("--extrema_enabled", dest="extrema_enabled", action="store_true", default=True, help="Enable Laplacian extrema warm start.")
    parser.add_argument("--no_extrema", dest="extrema_enabled", action="store_false", help="Disable Laplacian extrema warm start.")
    parser.add_argument("--extrema_knn", type=int, default=10, help="kNN value for Laplacian extrema.")
    parser.add_argument("--extrema_disable_subsample", action="store_true", help="Disable internal subsampling.")
    parser.add_argument("--extrema_max_points", type=int, default=10000, help="Maximum samples used for extrema computation.")
    parser.add_argument("--extrema_pca", type=float, default=None, help="PCA components (int > 1) or variance (float < 1) for extrema graph.")
    parser.add_argument("--extrema_seed", type=int, default=0, help="Seed for extrema subsampling.")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume analysis from. Skips clustering if results exist.")
    parser.add_argument("--cluster_labels_file", type=str, default=None, help="Path to a .npy file containing external cluster labels (shape: sae_width, dtype int, -1 for unassigned). If provided, skips co-occurrence collection and clustering entirely.")

    args = parser.parse_args()

    if args.hf_token:
        print("Logging in to Hugging Face with provided token...")
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        print("Logging in to Hugging Face with HF_TOKEN environment variable...")
        login(token=os.environ["HF_TOKEN"])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading Model: {args.model_name}")
    model_kwargs = {}
    if args.cache_dir:
        model_kwargs["cache_dir"] = args.cache_dir
        print(f"Using cache directory: {args.cache_dir}")
        
    model = HookedTransformer.from_pretrained_no_processing(args.model_name, device=args.device, center_unembed=False, center_writing_weights=False, dtype="bfloat16", **model_kwargs)
    
    print(f"Loading SAE: {args.sae_release} - {args.sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(args.sae_release, args.sae_id, device=args.device)
    print(f"SAE Config: {cfg_dict}")
    print(f"SAE Sparsity: {sparsity}")
    
    # Handle hook_name access
    hook_name = None
    if hasattr(sae.cfg, "hook_name"):
        hook_name = sae.cfg.hook_name
    elif hasattr(sae.cfg, "metadata") and "hook_name" in sae.cfg.metadata:
        hook_name = sae.cfg.metadata["hook_name"]
    elif cfg_dict and "hook_name" in cfg_dict:
        hook_name = cfg_dict["hook_name"]
        
    print(f"SAE Hook Name: {hook_name}")

    print("Initializing RealDataSampler...")
    # Seed Python's random for reproducibility
    import random
    random.seed(args.seed)
    print(f"Set Python random seed to {args.seed}")

    # Handle dataset config (None means no subset)
    hf_subset = None if args.hf_subset_name.lower() == "none" else args.hf_subset_name
    print(f"Using dataset: {args.hf_dataset}, subset: {hf_subset}, split: {args.dataset_split}")
    sampler = RealDataSampler(
        model,
        hf_dataset=args.hf_dataset,
        hf_subset_name=hf_subset,
        split=args.dataset_split,
        seed=args.seed,
        prefetch_size=args.aanet_prefetch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_doc_tokens=args.max_doc_tokens
    )
    

    # Determine site and hook_name from SAE config
    if hook_name:
        print(f"Using hook_name from SAE config: {hook_name}")
        
        # Derive site name for file paths (e.g. layer_20)
        if "blocks." in hook_name:
            try:
                layer = hook_name.split(".")[1]
                site = f"layer_{layer}"
            except IndexError:
                site = hook_name.replace(".", "_")
        else:
            site = hook_name.replace(".", "_")
    else:
        # Fallback to parsing sae_id if config doesn't have hook_name (unlikely for sae_lens)
        print("Warning: hook_name not found in SAE config. Attempting to parse from sae_id.")
        if "layer_" in args.sae_id:
            try:
                layer_str = args.sae_id.split("layer_")[1].split("/")[0]
                site = f"layer_{layer_str}"
                hook_name = f"blocks.{layer_str}.hook_resid_post"
            except:
                raise ValueError(f"Could not derive site/hook from sae_id: {args.sae_id}")
        else:
             raise ValueError(f"Could not derive site/hook from sae_id: {args.sae_id}")

    print(f"Site: {site}, Hook: {hook_name}")
    
    # Ensure SAE config has hook_name (if we derived it manually)
    if not hasattr(sae.cfg, "hook_name"):
        # We can't easily set attributes on frozen dataclasses or Pydantic models sometimes,
        # but let's try or just rely on our local 'hook_name' variable.
        pass

    for n_clusters in args.n_clusters_list:
        print(f"\n--- Running Analysis for n_clusters={n_clusters} ---")
        
        # Build co-occurrence config (used when method=spectral with cooc metrics)
        cooc_config = CooccurrenceConfig(
            n_batches=args.cooc_n_batches,
            batch_size=args.cooc_batch_size,
            seq_len=args.cooc_seq_len,
            activation_threshold=args.cooc_activation_threshold,
            skip_special_tokens=args.cooc_skip_special_tokens,
            cache_path=args.cooc_cache_path,
        )

        clustering_config = ClusteringConfig(
            site=site,
            method=args.method,
            selected_k=0,
            spectral_params=SpectralParams(
                sim_metric=args.sim_metric,
                n_clusters=n_clusters,
                cooccurrence_config=cooc_config,
            ),
            subspace_params=SubspaceParams(
                n_clusters=n_clusters,
                variance_threshold=args.subspace_variance_threshold,
                gap_threshold=args.subspace_gap_threshold,
            ),
            sampling_config=SamplingConfig(
                latent_activity_threshold=args.latent_activity_threshold,
                activation_batches=args.activity_batches,
                sample_sequences=args.activity_batch_size,
                sample_seq_len=args.activity_seq_len,
                max_activations=args.total_samples,
            ),
            geometry_fitting_config=GeometryFittingConfig(
                enabled=True,
                simplex_k_range=(2, 8),
                include_circle=False,
                include_hypersphere=False,
                cost_fn="cosine",
            ),
            sae_type="jumprelu",
            sae_param=0,
            seed=args.seed
        )
        
        # Instantiate pipeline with SAE and Config
        pipeline = RealDataClusteringPipeline(sae=sae, config=clustering_config)
        
        # Determine output directory
        if args.resume_from:
            output_dir = os.path.join(args.resume_from, f"clusters_{n_clusters}")
            print(f"Resuming from: {output_dir}")
        else:
            output_dir = os.path.join(args.output_dir, f"clusters_{n_clusters}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        clustering_pkl_path = os.path.join(output_dir, "clustering_result.pkl")
        result = None
        
        # Try to load existing clustering result if resuming (skipped when using external labels)
        if not args.cluster_labels_file and args.resume_from and os.path.exists(clustering_pkl_path):
            print(f"Loading existing clustering result from {clustering_pkl_path}")
            try:
                import pickle
                with open(clustering_pkl_path, "rb") as f:
                    result = pickle.load(f)
                print(f"Loaded clustering result with {result.n_clusters} clusters")
            except Exception as e:
                print(f"Failed to load clustering result: {e}. Re-running clustering.")
                result = None

        if args.cluster_labels_file:
            # Bypass co-occurrence collection and clustering — use external labels directly
            from types import SimpleNamespace
            print(f"Loading external cluster labels from {args.cluster_labels_file}")
            labels = np.load(args.cluster_labels_file)
            n_c = int(np.max(labels[labels != -1])) + 1
            assert n_c == n_clusters, (
                f"Labels file encodes {n_c} clusters but --n_clusters_list specifies {n_clusters}"
            )
            print(f"  Loaded {n_c} clusters, {int((labels != -1).sum())} assigned latents")
            result = SimpleNamespace(
                cluster_labels=labels,
                n_clusters=n_c,
                cluster_stats={cid: {} for cid in range(n_c)},
                subspace_diagnostics=None,
            )
        elif result is None:
            result = pipeline.run(
                model=model,
                cache={},
                data_source=sampler,
                site_dir=output_dir,
                component_beliefs_flat=None,
                device=args.device,
            )
            # Save clustering result
            import pickle
            with open(clustering_pkl_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Saved clustering result to {clustering_pkl_path}")

        # Compute activation PCA ranks if not already present
        # Initialize cluster_stats if it's empty (happens when acts_flat=None in clustering)
        if result.cluster_stats is None or len(result.cluster_stats) == 0:
            print("Initializing cluster_stats for PCA computation...")
            result.cluster_stats = {cid: {} for cid in range(result.n_clusters)}

        # Check if we need to compute PCA
        has_pca = any("activation_pca_rank" in stats for stats in result.cluster_stats.values())

        if args.skip_pca:
            print(f"\nSkipping PCA rank estimation (--skip_pca flag set)")
        elif not has_pca:
            # Clear memory before PCA computation
            torch.cuda.empty_cache()
            gc.collect()

            print(f"\nComputing activation PCA ranks for {result.n_clusters} clusters...")
            total_tokens = args.pca_num_cycles * 256 * args.activity_seq_len
            print(f"  Using {args.pca_num_cycles} cycles × 256 sequences × {args.activity_seq_len} tokens = {total_tokens:,} token positions sampled")
            print(f"  Max samples per cluster: {args.pca_max_samples:,} (will subsample if exceeded)")
            pca_ranks = compute_cluster_activation_pca_ranks(
                model=model,
                sae=sae,
                sampler=sampler,
                clustering_result=result,
                hook_name=hook_name,
                num_cycles=args.pca_num_cycles,
                batch_size=256,  # Hardcoded for memory safety
                seq_len=args.activity_seq_len,
                max_samples=args.pca_max_samples,
                variance_threshold=args.pca_variance_threshold,
                device=args.device
            )

            # Add to cluster_stats
            for cid, rank_info in pca_ranks.items():
                result.cluster_stats[cid]["activation_pca_rank"] = rank_info["avg_rank"]
                result.cluster_stats[cid]["activation_pca_variance_explained"] = rank_info["avg_variance"]

            # Re-save clustering result with PCA ranks
            with open(clustering_pkl_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Added PCA ranks to clustering result and saved to {clustering_pkl_path}")
        else:
            print(f"Note: PCA ranks already present in cluster_stats, skipping computation")

        if args.clustering_only:
            print(f"\n--clustering_only set, skipping AANet fitting for n_clusters={n_clusters}")
            continue

        print(f"Running AAnet fitting for n_clusters={n_clusters}")
        
        # Load existing metrics if resuming
        csv_path = os.path.join(output_dir, f"consolidated_metrics_n{n_clusters}.csv")
        completed_clusters = set()
        
        if args.resume_from and os.path.exists(csv_path):
            import pandas as pd
            try:
                existing_df = pd.read_csv(csv_path)
                # We need to know which (cluster_id, k) pairs are done
                # But we iterate by cluster_id, then k.
                # If we want to skip entire clusters, we should check if ALL k are done?
                # Or just append rows?
                # Safest is to append rows. But we don't want duplicates.
                # Let's track completed (cluster_id, k) tuples.
                for _, row in existing_df.iterrows():
                    completed_clusters.add((int(row["cluster_id"]), int(row["aanet_k"])))
                print(f"Found {len(completed_clusters)} completed AAnet runs in {csv_path}")
            except Exception as e:
                print(f"Failed to read existing CSV: {e}. Starting fresh.")
        
        consolidated_results = []
        descriptors = []
        unique_labels = np.unique(result.cluster_labels)
        for label in unique_labels:
            if label == -1: continue
            latent_indices = np.where(result.cluster_labels == label)[0].tolist()
            descriptors.append(AAnetDescriptor(
                cluster_id=int(label),
                label=f"cluster_{label}",
                latent_indices=latent_indices,
                component_names=[],
                is_noise=False
            ))
            
        # --- Streaming Training ---
        
        # Helper to create trainer for a specific k
        def create_trainer(k, descriptors_list):
            active_descriptors = []
            for desc in descriptors_list:
                if (desc.cluster_id, k) not in completed_clusters:
                    active_descriptors.append(desc)
            
            if not active_descriptors:
                return None, 0

            aanet_config = TrainingConfig(
                k=k,
                epochs=1,
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
            
            # Note: AAnet trains on reconstructions (d_model), not sparse latents (d_sae).
            # d_model is the output dimension of the SAE decoder (W_dec has shape [d_sae, d_model])
            d_model = sae.W_dec.shape[1]
            trainer = StreamingAAnetTrainer(
                descriptors=active_descriptors,
                config=aanet_config,
                device=args.device,
                input_dim=d_model,
                sae_decoder_weight=sae.W_dec
            )
            return trainer, len(active_descriptors)

        # --- Warmup Phase (Extrema) ---
        # Process clusters in chunks to reduce memory usage

        all_extrema = {}

        if args.extrema_enabled:
            # Split descriptors into chunks
            chunk_size = args.aanet_warmup_cluster_chunk_size
            n_chunks = (len(descriptors) + chunk_size - 1) // chunk_size

            print(f"Processing {len(descriptors)} clusters in {n_chunks} chunks of size {chunk_size}")
            print(f"Collecting up to {args.aanet_warmup_max_per_cluster} samples/cluster over {args.aanet_warmup_steps} steps")

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(descriptors))
                chunk_descriptors = descriptors[start_idx:end_idx]

                print(f"\n--- Warmup Chunk {chunk_idx+1}/{n_chunks} (clusters {start_idx}-{end_idx-1}) ---")

                # Initialize buffers for this chunk only
                warmup_buffer = {desc.cluster_id: [] for desc in chunk_descriptors}
                cluster_indices_map = {desc.cluster_id: torch.tensor(desc.latent_indices, device=args.device) for desc in chunk_descriptors}

                # Streaming Loop for this chunk
                for step in tqdm(range(args.aanet_warmup_steps), desc=f"Warmup Chunk {chunk_idx+1}/{n_chunks}"):
                    # 1. Stream ONE batch
                    tokens = sampler.sample_tokens_batch(args.aanet_batch_size, args.activity_seq_len, args.device)

                    with torch.no_grad():
                        _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                        acts = cache[hook_name]
                        acts_flat = acts.reshape(-1, acts.shape[-1])
                        feature_acts, _, _ = sae_encode_features(sae, acts_flat)

                        # Exclude BOS tokens (position 0) - not meaningful for training
                        feature_acts = delete_special_tokens(feature_acts, args.aanet_batch_size, args.activity_seq_len, args.device)

                        # 2. Distribute to clusters in this chunk
                        for cid, indices in cluster_indices_map.items():
                            if indices.numel() == 0: continue

                            # Check if full
                            current_count = sum(t.shape[0] for t in warmup_buffer[cid])
                            if current_count >= args.aanet_warmup_max_per_cluster:
                                continue

                            subset = feature_acts[:, indices]
                            active_mask = (subset.abs().sum(dim=1) > args.aanet_active_threshold)

                            if active_mask.any():
                                W_c = sae.W_dec[indices, :]
                                recon = subset[active_mask] @ W_c

                                # Truncate if adding this batch exceeds max
                                remaining = args.aanet_warmup_max_per_cluster - current_count
                                if recon.shape[0] > remaining:
                                    recon = recon[:remaining]

                                warmup_buffer[cid].append(recon.cpu())

                    # 3. Discard batch
                    del tokens, cache, acts, acts_flat, feature_acts

                # Compute extrema for this chunk
                print(f"Computing extrema for chunk {chunk_idx+1}/{n_chunks}...")
                for cid, tensors in tqdm(warmup_buffer.items(), desc=f"Extrema Chunk {chunk_idx+1}"):
                    if not tensors:
                        continue
                    data = torch.cat(tensors, dim=0)
                    if data.shape[0] < args.aanet_k_max:
                        print(f"Cluster {cid}: insufficient samples ({data.shape[0]} < {args.aanet_k_max}), skipping extrema")
                        continue

                    try:
                        data_np = data.numpy()
                        extrema = compute_diffusion_extrema(
                            data_np,
                            max_k=args.aanet_k_max,
                            config=ExtremaConfig(knn=args.extrema_knn)
                        )
                        if extrema is not None:
                            all_extrema[cid] = extrema.to(args.device)
                    except Exception as e:
                        print(f"Failed to compute extrema for cluster {cid}: {e}")

                # Clear chunk buffers before next chunk
                del warmup_buffer, cluster_indices_map
                torch.cuda.empty_cache()
                gc.collect(); 

            print(f"\nCompleted warmup for all {len(descriptors)} clusters. Computed extrema for {len(all_extrema)} clusters.")

        # --- Training Loop ---
        
        if args.aanet_sequential_k:
            print(f"Running Sequential Training (one k at a time) for k={args.aanet_k_min} to {args.aanet_k_max}...")
            
            for k in range(args.aanet_k_min, args.aanet_k_max + 1):
                print(f"\n==========================================")
                print(f"--- STARTING TRAINING FOR k={k} ---")
                print(f"==========================================")
                
                trainer, n_active = create_trainer(k, descriptors)
                if trainer is None:
                    print(f"All clusters completed for k={k}. Skipping.")
                    continue
                
                print(f"Initializing Trainer for k={k} with {n_active} clusters")
                
                # Initialize Extrema
                if args.extrema_enabled:
                    count = 0
                    skipped = 0
                    for cid, model_inst in trainer.models.items():
                        if cid in all_extrema:
                            k_extrema = all_extrema[cid][:k]
                            # Only set extrema if we have exactly k extrema (safety check)
                            if k_extrema.shape[0] == k:
                                model_inst.set_archetypes(k_extrema)
                                count += 1
                            else:
                                skipped += 1
                                # Don't set extrema - model will train without extrema loss
                    if skipped > 0:
                        print(f"Initialized extrema for {count} models (skipped {skipped} clusters with only {k_extrema.shape[0]}/{k} extrema).")
                    else:
                        print(f"Initialized extrema for {count} models.")

                # Streaming Loop for this k
                metrics_history = {cid: {"loss": [], "reconstruction_loss": [], "archetypal_loss": [], "extrema_loss": []} for cid in trainer.models.keys()}

                # Early stopping tracking (per cluster)
                best_losses = {cid: float('inf') for cid in trainer.models.keys()}
                steps_since_improvement = {cid: 0 for cid in trainer.models.keys()}
                best_steps = {cid: 0 for cid in trainer.models.keys()}
                stopped_clusters = set()
                check_interval = (args.aanet_loss_smoothing_window // 2) + (args.aanet_loss_smoothing_window // 4)  # Check every 3/4-window

                print(f"Starting Streaming Training for k={k} ({args.aanet_streaming_steps} steps)...")
                for step in tqdm(range(args.aanet_streaming_steps), desc=f"Streaming k={k}"):
                    # Skip if all clusters stopped
                    if len(stopped_clusters) == len(trainer.models):
                        print(f"\nAll clusters have stopped early at step {step}")
                        break

                    # 1. Stream ONE batch
                    tokens = sampler.sample_tokens_batch(args.aanet_batch_size, args.activity_seq_len, args.device)

                    with torch.no_grad():
                        _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                        acts = cache[hook_name]
                        acts_flat = acts.reshape(-1, acts.shape[-1])
                        feature_acts, _, _ = sae_encode_features(sae, acts_flat)

                        # Exclude BOS tokens (position 0) - not meaningful for training
                        feature_acts = delete_special_tokens(feature_acts, args.aanet_batch_size, args.activity_seq_len, args.device)

                    # 2. Train (Distribute to all clusters in this trainer)
                    step_losses = trainer.train_step(feature_acts)
                    for cid, metrics in step_losses.items():
                        if cid in stopped_clusters:
                            continue
                        for key, value in metrics.items():
                            metrics_history[cid][key].append(value)

                        # Early stopping and LR scheduling for this cluster
                        current_loss = metrics['loss']

                        # Step LR scheduler
                        current_lr = trainer.optimizers[cid].param_groups[0]['lr']
                        if current_lr > args.aanet_min_lr:
                            trainer.schedulers[cid].step(current_loss)
                            new_lr = trainer.optimizers[cid].param_groups[0]['lr']
                            if new_lr != current_lr:
                                print(f"\n  Cluster {cid}, Step {step}: LR reduced from {current_lr:.2e} to {new_lr:.2e}")

                        # Early stopping check - only at intervals
                        if step % check_interval == 0 and len(metrics_history[cid]['loss']) >= args.aanet_loss_smoothing_window:
                            smoothed_loss = np.mean(metrics_history[cid]['loss'][-args.aanet_loss_smoothing_window:])

                            if smoothed_loss < best_losses[cid] - args.aanet_min_delta:
                                best_losses[cid] = smoothed_loss
                                steps_since_improvement[cid] = -1  # Will become 0 after increment below
                                best_steps[cid] = step
                                print(f"\n  Cluster {cid}, Step {step}: New best smoothed loss: {best_losses[cid]:.6f}")

                        steps_since_improvement[cid] += 1

                        if steps_since_improvement[cid] >= args.aanet_early_stop_patience:
                            print(f"\n  Cluster {cid}: Early stopping at step {step} (best smoothed loss {best_losses[cid]:.6f} at step {best_steps[cid]})")
                            stopped_clusters.add(cid)

                    # 3. Discard batch
                    del tokens, cache, acts, acts_flat, feature_acts
                
                # Save results for this k
                print(f"Saving results for k={k}...")
                for cid, model_inst in trainer.models.items():
                    history = metrics_history[cid]
                    final_metrics = {}
                    for key, values in history.items():
                        final_metrics[key] = np.median(values[-20:]) if values and len(values) >= 20 else (values[-1] if values else float("nan"))
                    
                    row = {
                        "n_clusters_total": n_clusters,
                        "cluster_id": cid,
                        "n_latents": len(trainer.cluster_indices[cid]),
                        "latent_indices": str(trainer.cluster_indices[cid].tolist()),
                        "aanet_k": k,
                        "aanet_status": "ok",
                        "aanet_loss": final_metrics["loss"],
                        "aanet_recon_loss": final_metrics["reconstruction_loss"],
                        "aanet_archetypal_loss": final_metrics["archetypal_loss"],
                        "aanet_extrema_loss": final_metrics["extrema_loss"],
                        "sae_id": args.sae_id,
                        "average_l0": sparsity,
                    }
                    
                    # Add clustering metrics
                    if result.subspace_diagnostics:
                        recon_errors = result.subspace_diagnostics.get("reconstruction_errors", {})
                        row["cluster_recon_error"] = recon_errors.get(cid, float("nan"))
                        cluster_ranks = result.subspace_diagnostics.get("cluster_ranks", {})
                        row["decoder_dir_rank"] = cluster_ranks.get(cid, float("nan"))
                        
                    # Add activation PCA rank
                    if result.cluster_stats and cid in result.cluster_stats:
                        stats = result.cluster_stats[cid]
                        # First check for pre-computed rank (new format)
                        if "activation_pca_rank" in stats:
                            row["activation_pca_rank"] = stats["activation_pca_rank"]
                        # Fall back to computing from variance ratios (old format)
                        elif "explained_variance_ratio" in stats:
                            var_ratios = np.array(stats["explained_variance_ratio"])
                            cumsum = np.cumsum(var_ratios)
                            rank_95 = np.argmax(cumsum >= 0.95) + 1 if np.any(cumsum >= 0.95) else len(var_ratios)
                            row["activation_pca_rank"] = rank_95
                        else:
                            row["activation_pca_rank"] = float("nan")
                    else:
                        row["activation_pca_rank"] = float("nan")

                    # Incremental save
                    import pandas as pd
                    row_df = pd.DataFrame([row])
                    write_header = not os.path.exists(csv_path)
                    row_df.to_csv(csv_path, mode='a', header=write_header, index=False)
                    
                    # Save training curves
                    jsonl_path = os.path.join(output_dir, f"training_curves_n{n_clusters}.jsonl")
                    curve_data = {
                        "cluster_id": cid,
                        "aanet_k": k,
                        "metrics_history": history
                    }
                    import json
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(curve_data) + "\n")

                    # Save model weights
                    model_path = os.path.join(output_dir, f"aanet_cluster_{cid}_k{k}.pt")
                    torch.save(model_inst.state_dict(), model_path)

                # Cleanup trainer
                del trainer
                torch.cuda.empty_cache()
                gc.collect()

            # Calculate elbow quality metrics after all k values complete
            print(f"\nCalculating elbow quality metrics for n_clusters={n_clusters}...")
            calculate_elbow_quality_metrics(csv_path, n_clusters)
            print(f"Elbow quality metrics saved to {csv_path}")

            # Generate analysis plots
            print(f"\nGenerating analysis plots for n_clusters={n_clusters}...")
            generate_analysis_plots(csv_path, n_clusters, output_dir)
            print(f"Analysis plots generated for n_clusters={n_clusters}")

        else:
            print(f"Running Concurrent Training (all k at once) for k={args.aanet_k_min} to {args.aanet_k_max}...")
            print(f"==========================================")
            print(f"--- INITIALIZING ALL TRAINERS ---")
            print(f"==========================================")
            
            trainers = {}
            for k in range(args.aanet_k_min, args.aanet_k_max + 1):
                trainer, n_active = create_trainer(k, descriptors)
                if trainer:
                    print(f"Initializing Trainer for k={k} with {n_active} clusters")
                    trainers[k] = trainer
            
            if not trainers:
                print("No training needed (all completed).")
                continue
            
            # Initialize Extrema
            if args.extrema_enabled:
                for k, trainer in trainers.items():
                    count = 0
                    skipped = 0
                    for cid, model_inst in trainer.models.items():
                        if cid in all_extrema:
                            k_extrema = all_extrema[cid][:k]
                            # Only set extrema if we have exactly k extrema (safety check)
                            if k_extrema.shape[0] == k:
                                model_inst.set_archetypes(k_extrema)
                                count += 1
                            else:
                                skipped += 1
                                # Don't set extrema - model will train without extrema loss
                    if skipped > 0:
                        print(f"Initialized extrema for {count} models (k={k}, skipped {skipped} clusters with insufficient extrema).")
                    else:
                        print(f"Initialized extrema for {count} models (k={k}).")
            
            # Streaming Loop
            print(f"Starting Streaming Training ({args.aanet_streaming_steps} steps)...")
            training_history = {k: {cid: {"loss": [], "reconstruction_loss": [], "archetypal_loss": [], "extrema_loss": []} for cid in trainer.models.keys()} for k, trainer in trainers.items()}

            # Early stopping tracking (per k, per cluster)
            best_losses = {k: {cid: float('inf') for cid in trainer.models.keys()} for k, trainer in trainers.items()}
            steps_since_improvement = {k: {cid: 0 for cid in trainer.models.keys()} for k, trainer in trainers.items()}
            best_steps = {k: {cid: 0 for cid in trainer.models.keys()} for k, trainer in trainers.items()}
            stopped_clusters = {k: set() for k in trainers.keys()}
            check_interval = (args.aanet_loss_smoothing_window // 2) + (args.aanet_loss_smoothing_window // 4)  # Check every 3/4-window

            for step in tqdm(range(args.aanet_streaming_steps), desc="Streaming"):
                # Check if all clusters across all k values have stopped
                all_stopped = all(len(stopped_clusters[k]) == len(trainers[k].models) for k in trainers.keys())
                if all_stopped:
                    print(f"\nAll clusters across all k values have stopped early at step {step}")
                    break

                # 1. Stream ONE batch
                tokens = sampler.sample_tokens_batch(args.aanet_batch_size, args.activity_seq_len, args.device)

                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                    acts = cache[hook_name]
                    acts_flat = acts.reshape(-1, acts.shape[-1])
                    feature_acts, _, _ = sae_encode_features(sae, acts_flat)

                    # Exclude BOS tokens (position 0) - not meaningful for training
                    feature_acts = delete_special_tokens(feature_acts, args.aanet_batch_size, args.activity_seq_len, args.device)

                # 2. Train (Distribute to ALL trainers)
                for k, trainer in trainers.items():
                    step_losses = trainer.train_step(feature_acts)
                    for cid, metrics in step_losses.items():
                        if cid in stopped_clusters[k]:
                            continue

                        for key, value in metrics.items():
                            training_history[k][cid][key].append(value)

                        # Early stopping and LR scheduling for this cluster
                        current_loss = metrics['loss']

                        # Step LR scheduler
                        current_lr = trainer.optimizers[cid].param_groups[0]['lr']
                        if current_lr > args.aanet_min_lr:
                            trainer.schedulers[cid].step(current_loss)
                            new_lr = trainer.optimizers[cid].param_groups[0]['lr']
                            if new_lr != current_lr:
                                print(f"\n  k={k}, Cluster {cid}, Step {step}: LR reduced from {current_lr:.2e} to {new_lr:.2e}")

                        # Early stopping check - only at intervals
                        if step % check_interval == 0 and len(training_history[k][cid]['loss']) >= args.aanet_loss_smoothing_window:
                            smoothed_loss = np.mean(training_history[k][cid]['loss'][-args.aanet_loss_smoothing_window:])

                            if smoothed_loss < best_losses[k][cid] - args.aanet_min_delta:
                                best_losses[k][cid] = smoothed_loss
                                steps_since_improvement[k][cid] = -1  # Will become 0 after increment below
                                best_steps[k][cid] = step
                                print(f"\n  k={k}, Cluster {cid}, Step {step}: New best smoothed loss: {best_losses[k][cid]:.6f}")

                        steps_since_improvement[k][cid] += 1

                        if steps_since_improvement[k][cid] >= args.aanet_early_stop_patience:
                            print(f"\n  k={k}, Cluster {cid}: Early stopping at step {step} (best smoothed loss {best_losses[k][cid]:.6f} at step {best_steps[k][cid]})")
                            stopped_clusters[k].add(cid)

                # 3. Discard batch
                del tokens, cache, acts, acts_flat, feature_acts
            
            # Save Results (Concurrent)
            print("Saving results...")
            for k, trainer in trainers.items():
                for cid, model_inst in trainer.models.items():
                    history = training_history[k][cid]
                    final_metrics = {}
                    for key, values in history.items():
                        final_metrics[key] = np.median(values[-20:]) if values and len(values) >= 20 else (values[-1] if values else float("nan"))
                    
                    row = {
                        "n_clusters_total": n_clusters,
                        "cluster_id": cid,
                        "n_latents": len(trainer.cluster_indices[cid]),
                        "latent_indices": str(trainer.cluster_indices[cid].tolist()),
                        "aanet_k": k,
                        "aanet_status": "ok",
                        "aanet_loss": final_metrics["loss"],
                        "aanet_recon_loss": final_metrics["reconstruction_loss"],
                        "aanet_archetypal_loss": final_metrics["archetypal_loss"],
                        "aanet_extrema_loss": final_metrics["extrema_loss"],
                        "sae_id": args.sae_id,
                        "average_l0": sparsity,
                    }
                    
                    # Add clustering metrics
                    if result.subspace_diagnostics:
                        recon_errors = result.subspace_diagnostics.get("reconstruction_errors", {})
                        row["cluster_recon_error"] = recon_errors.get(cid, float("nan"))
                        cluster_ranks = result.subspace_diagnostics.get("cluster_ranks", {})
                        row["decoder_dir_rank"] = cluster_ranks.get(cid, float("nan"))
                        
                    # Add activation PCA rank
                    if result.cluster_stats and cid in result.cluster_stats:
                        stats = result.cluster_stats[cid]
                        # First check for pre-computed rank (new format)
                        if "activation_pca_rank" in stats:
                            row["activation_pca_rank"] = stats["activation_pca_rank"]
                        # Fall back to computing from variance ratios (old format)
                        elif "explained_variance_ratio" in stats:
                            var_ratios = np.array(stats["explained_variance_ratio"])
                            cumsum = np.cumsum(var_ratios)
                            rank_95 = np.argmax(cumsum >= 0.95) + 1 if np.any(cumsum >= 0.95) else len(var_ratios)
                            row["activation_pca_rank"] = rank_95
                        else:
                            row["activation_pca_rank"] = float("nan")
                    else:
                        row["activation_pca_rank"] = float("nan")

                    # Incremental save
                    import pandas as pd
                    row_df = pd.DataFrame([row])
                    write_header = not os.path.exists(csv_path)
                    row_df.to_csv(csv_path, mode='a', header=write_header, index=False)
                    
                    # Save training curves
                    jsonl_path = os.path.join(output_dir, f"training_curves_n{n_clusters}.jsonl")
                    curve_data = {
                        "cluster_id": cid,
                        "aanet_k": k,
                        "metrics_history": history
                    }
                    import json
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(curve_data) + "\n")

                    # Save model weights
                    model_path = os.path.join(output_dir, f"aanet_cluster_{cid}_k{k}.pt")
                    torch.save(model_inst.state_dict(), model_path)

            # Calculate elbow quality metrics
            print(f"\nCalculating elbow quality metrics for n_clusters={n_clusters}...")
            calculate_elbow_quality_metrics(csv_path, n_clusters)
            print(f"Elbow quality metrics saved to {csv_path}")

            # Generate analysis plots
            print(f"\nGenerating analysis plots for n_clusters={n_clusters}...")
            generate_analysis_plots(csv_path, n_clusters, output_dir)
            print(f"Analysis plots generated for n_clusters={n_clusters}")


def generate_analysis_plots(csv_path, n_clusters, output_dir):
    """
    Generate key analysis plots for cluster quality assessment.

    Plots generated:
    1. PCA rank vs Geometric rank scatter
    2. Recon vs Arch elbow strength (unzoomed)
    3. Recon vs Arch elbow strength (zoomed, excluding top 2 outliers)
    4. Decoder rank distribution histogram
    5. Elbow strength distributions (recon & arch)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter to one row per cluster (all k have same clustering metrics)
    df_k2 = df[df['aanet_k'] == df['aanet_k'].min()].copy()

    print(f"Generating analysis plots for n_clusters={n_clusters}...")

    # ============================================================================
    # Plot 1: PCA Rank vs Geometric Rank
    # ============================================================================
    if 'activation_pca_rank' in df_k2.columns and 'decoder_dir_rank' in df_k2.columns:
        valid_data = df_k2.dropna(subset=['activation_pca_rank', 'decoder_dir_rank'])

        if len(valid_data) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            scatter = ax.scatter(
                valid_data['decoder_dir_rank'],
                valid_data['activation_pca_rank'],
                alpha=0.6,
                s=50,
                c=valid_data['n_latents'],
                cmap='viridis'
            )

            # Add diagonal line (perfect agreement)
            max_rank = max(valid_data['decoder_dir_rank'].max(), valid_data['activation_pca_rank'].max())
            ax.plot([0, max_rank], [0, max_rank], 'r--', alpha=0.5, linewidth=2, label='Perfect Agreement')

            ax.set_xlabel('Geometric Rank (from Clustering)', fontsize=12)
            ax.set_ylabel('PCA Rank (from Activations)', fontsize=12)
            ax.set_title(f'PCA Rank vs Geometric Rank (n_clusters={n_clusters})', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.colorbar(scatter, ax=ax, label='Number of Latents in Cluster')
            plt.tight_layout()
            plt.savefig(plots_dir / f'pca_vs_geometric_rank_n{n_clusters}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved PCA vs Geometric rank plot")
        else:
            print(f"  ⚠ Skipping PCA vs Geometric rank plot (no valid data)")

    # ============================================================================
    # Plot 2: Decoder Rank Distribution
    # ============================================================================
    if 'decoder_dir_rank' in df_k2.columns:
        valid_data = df_k2.dropna(subset=['decoder_dir_rank'])

        if len(valid_data) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            ax.hist(valid_data['decoder_dir_rank'], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
            ax.set_xlabel('Decoder Direction Rank', fontsize=12)
            ax.set_ylabel('Number of Clusters', fontsize=12)
            ax.set_title(f'Decoder Direction Rank Distribution (n_clusters={n_clusters})', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistics text
            mean_rank = valid_data['decoder_dir_rank'].mean()
            median_rank = valid_data['decoder_dir_rank'].median()
            ax.axvline(mean_rank, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rank:.1f}')
            ax.axvline(median_rank, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_rank:.1f}')
            ax.legend()

            plt.tight_layout()
            plt.savefig(plots_dir / f'decoder_rank_dist_n{n_clusters}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved decoder rank distribution")

    # ============================================================================
    # Plot 3: Elbow Strength Distributions
    # ============================================================================
    if 'aanet_recon_loss_elbow_strength' in df_k2.columns and 'aanet_archetypal_loss_elbow_strength' in df_k2.columns:
        valid_data = df_k2.dropna(subset=['aanet_recon_loss_elbow_strength', 'aanet_archetypal_loss_elbow_strength'])

        if len(valid_data) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Reconstruction elbow strength
            ax = axes[0]
            ax.hist(valid_data['aanet_recon_loss_elbow_strength'], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
            ax.set_xlabel('Reconstruction Elbow Strength', fontsize=12)
            ax.set_ylabel('Number of Clusters', fontsize=12)
            ax.set_title(f'Reconstruction Elbow Strength (n_clusters={n_clusters})', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')

            mean_val = valid_data['aanet_recon_loss_elbow_strength'].mean()
            median_val = valid_data['aanet_recon_loss_elbow_strength'].median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax.axvline(median_val, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')
            ax.legend()

            # Archetypal elbow strength
            ax = axes[1]
            ax.hist(valid_data['aanet_archetypal_loss_elbow_strength'], bins=30, alpha=0.7, edgecolor='black', color='coral')
            ax.set_xlabel('Archetypal Elbow Strength', fontsize=12)
            ax.set_ylabel('Number of Clusters', fontsize=12)
            ax.set_title(f'Archetypal Elbow Strength (n_clusters={n_clusters})', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')

            mean_val = valid_data['aanet_archetypal_loss_elbow_strength'].mean()
            median_val = valid_data['aanet_archetypal_loss_elbow_strength'].median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax.axvline(median_val, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')
            ax.legend()

            plt.tight_layout()
            plt.savefig(plots_dir / f'elbow_strength_dists_n{n_clusters}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved elbow strength distributions")

    # ============================================================================
    # Plot 4: Recon vs Arch Elbow Strength (UNZOOMED)
    # ============================================================================
    if 'aanet_recon_loss_elbow_strength' in df_k2.columns and 'aanet_archetypal_loss_elbow_strength' in df_k2.columns and 'k_differential' in df_k2.columns:
        valid_data = df_k2.dropna(subset=['aanet_recon_loss_elbow_strength', 'aanet_archetypal_loss_elbow_strength', 'k_differential']).copy()

        if len(valid_data) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Color by k_differential
            unique_diffs = sorted(valid_data['k_differential'].unique())

            if len(unique_diffs) <= 20:
                color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
            else:
                tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
                set3 = plt.cm.Set3(np.linspace(0, 1, 12))
                color_palette = np.vstack([tab20, set3])

            diff_to_color = {}
            color_idx = 0
            for diff in unique_diffs:
                if diff == 0:
                    diff_to_color[diff] = 'black'
                else:
                    diff_to_color[diff] = color_palette[color_idx % len(color_palette)]
                    color_idx += 1

            # Sort so differential == 0 is plotted last (on top)
            valid_data['plot_order'] = (valid_data['k_differential'] == 0).astype(int)
            valid_data = valid_data.sort_values('plot_order')
            colors = [diff_to_color[diff] for diff in valid_data['k_differential']]

            ax.scatter(
                valid_data['aanet_recon_loss_elbow_strength'],
                valid_data['aanet_archetypal_loss_elbow_strength'],
                c=colors,
                s=40,
                alpha=0.6
            )

            ax.set_xlabel('Reconstruction Loss Elbow Strength', fontsize=12)
            ax.set_ylabel('Archetypal Loss Elbow Strength', fontsize=12)
            ax.set_title(f'Elbow Strength: Recon vs Arch (Unzoomed, n_clusters={n_clusters})', fontsize=14)
            ax.grid(True, alpha=0.3)

            # Add legend (limit to first 10 for readability)
            from matplotlib.patches import Patch
            legend_elements = []
            for diff in sorted(unique_diffs)[:10]:
                label = f'Δk={int(diff)}' if diff != 0 else 'Δk=0 (same)'
                legend_elements.append(Patch(facecolor=diff_to_color[diff], label=label, alpha=0.6))
            ax.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.9, title='K Differential')

            plt.tight_layout()
            plt.savefig(plots_dir / f'recon_vs_arch_unzoomed_n{n_clusters}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved recon vs arch (unzoomed)")

    # ============================================================================
    # Plot 5: Recon vs Arch Elbow Strength (ZOOMED - exclude top 2 outliers)
    # ============================================================================
    if 'aanet_recon_loss_elbow_strength' in df_k2.columns and 'aanet_archetypal_loss_elbow_strength' in df_k2.columns and 'k_differential' in df_k2.columns:
        # Apply quality filters first
        quality_filtered = df_k2.copy()
        if 'n_latents' in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered['n_latents'] >= 2].copy()
        if 'recon_is_monotonic' in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered['recon_is_monotonic'] == True].copy()
        if 'arch_is_monotonic' in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered['arch_is_monotonic'] == True].copy()
        if 'recon_pct_decrease' in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered['recon_pct_decrease'] >= 20].copy()
        if 'arch_pct_decrease' in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered['arch_pct_decrease'] >= 20].copy()
        if 'k_differential' in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered['k_differential'].abs() <= 1].copy()

        valid_data = quality_filtered.dropna(subset=['aanet_recon_loss_elbow_strength', 'aanet_archetypal_loss_elbow_strength', 'k_differential']).copy()

        if len(valid_data) > 0:
            # Calculate zoom limits from quality-filtered data
            recon_mean = valid_data['aanet_recon_loss_elbow_strength'].mean()
            recon_std = valid_data['aanet_recon_loss_elbow_strength'].std()
            arch_mean = valid_data['aanet_archetypal_loss_elbow_strength'].mean()
            arch_std = valid_data['aanet_archetypal_loss_elbow_strength'].std()

            xlim_max = recon_mean + 10 * recon_std
            ylim_max = arch_mean + 10 * arch_std

            # For display, exclude top 2 outliers
            recon_threshold = df_k2['aanet_recon_loss_elbow_strength'].nlargest(2).min()
            arch_threshold = df_k2['aanet_archetypal_loss_elbow_strength'].nlargest(2).min()

            plot_data = valid_data[
                (valid_data['aanet_recon_loss_elbow_strength'] < recon_threshold) &
                (valid_data['aanet_archetypal_loss_elbow_strength'] < arch_threshold)
            ].copy()

            if len(plot_data) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))

                # Color by k_differential
                unique_diffs = sorted(plot_data['k_differential'].unique())

                if len(unique_diffs) <= 20:
                    color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
                else:
                    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
                    set3 = plt.cm.Set3(np.linspace(0, 1, 12))
                    color_palette = np.vstack([tab20, set3])

                diff_to_color = {}
                color_idx = 0
                for diff in unique_diffs:
                    if diff == 0:
                        diff_to_color[diff] = 'black'
                    else:
                        diff_to_color[diff] = color_palette[color_idx % len(color_palette)]
                        color_idx += 1

                # Sort so differential == 0 is plotted last (on top)
                plot_data['plot_order'] = (plot_data['k_differential'] == 0).astype(int)
                plot_data = plot_data.sort_values('plot_order')
                colors = [diff_to_color[diff] for diff in plot_data['k_differential']]

                ax.scatter(
                    plot_data['aanet_recon_loss_elbow_strength'],
                    plot_data['aanet_archetypal_loss_elbow_strength'],
                    c=colors,
                    s=40,
                    alpha=0.6
                )

                ax.set_xlabel('Reconstruction Loss Elbow Strength', fontsize=12)
                ax.set_ylabel('Archetypal Loss Elbow Strength', fontsize=12)
                ax.set_title(f'Elbow Strength: Recon vs Arch (Zoomed, n_clusters={n_clusters})\nExcluding top 2 outliers, showing quality-filtered clusters', fontsize=13)
                ax.grid(True, alpha=0.3)

                # Set zoom limits
                ax.set_xlim(0, xlim_max)
                ax.set_ylim(0, ylim_max)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = []
                for diff in sorted(unique_diffs)[:10]:
                    label = f'Δk={int(diff)}' if diff != 0 else 'Δk=0 (same)'
                    legend_elements.append(Patch(facecolor=diff_to_color[diff], label=label, alpha=0.6))
                ax.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.9, title='K Differential')

                plt.tight_layout()
                plt.savefig(plots_dir / f'recon_vs_arch_zoomed_n{n_clusters}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved recon vs arch (zoomed)")
            else:
                print(f"  ⚠ Skipping zoomed plot (no data after filtering)")
        else:
            print(f"  ⚠ Skipping zoomed plot (no valid data after quality filters)")

    print(f"All plots saved to: {plots_dir}/")


def calculate_elbow_quality_metrics(csv_path, n_clusters):
    """
    Calculate elbow quality metrics for each cluster and add them to the CSV.

    Metrics calculated:
    - recon_elbow_k, recon_elbow_strength: Elbow location and strength for reconstruction loss
    - recon_pct_decrease: Percent decrease from max to min reconstruction loss
    - recon_is_monotonic: Whether reconstruction loss is monotonic up to the elbow
    - arch_elbow_k, arch_elbow_strength: Elbow location and strength for archetypal loss
    - arch_pct_decrease: Percent decrease from max to min archetypal loss
    - arch_is_monotonic: Whether archetypal loss is monotonic up to the elbow
    - k_differential: Absolute difference between recon and arch elbow k values
    """
    import pandas as pd
    import numpy as np

    def calculate_elbow_score(x, y):
        """Calculate elbow using perpendicular distance method."""
        if len(x) < 3:
            return None, 0.0
        x_norm = (np.array(x) - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
        y_norm = (np.array(y) - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else np.zeros_like(y)
        distances = np.abs(x_norm + y_norm - 1) / np.sqrt(2)
        elbow_idx = np.argmax(distances)
        elbow_k = x[elbow_idx]
        elbow_strength = distances[elbow_idx]
        return elbow_k, elbow_strength

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Calculate elbow metrics for each cluster
    results = []
    for cluster_id in df['cluster_id'].unique():
        cluster_data = df[df['cluster_id'] == cluster_id].sort_values('aanet_k')
        if len(cluster_data) < 3:
            continue

        k_vals = cluster_data['aanet_k'].values

        # Reconstruction loss metrics
        recon_losses = cluster_data['aanet_recon_loss'].values
        recon_elbow_k, recon_elbow_strength = calculate_elbow_score(k_vals, recon_losses)
        recon_max = np.max(recon_losses)
        recon_min = np.min(recon_losses)
        recon_pct_decrease = (recon_max - recon_min) / recon_max * 100 if recon_max > 0 else 0

        # Check monotonicity up to elbow for recon
        recon_is_monotonic = False
        if recon_elbow_k is not None:
            elbow_idx = np.where(k_vals == recon_elbow_k)[0][0]
            recon_is_monotonic = all(recon_losses[i] >= recon_losses[i+1] for i in range(elbow_idx))

        # Archetypal loss metrics
        arch_losses = cluster_data['aanet_archetypal_loss'].values
        arch_elbow_k, arch_elbow_strength = calculate_elbow_score(k_vals, arch_losses)
        arch_max = np.max(arch_losses)
        arch_min = np.min(arch_losses)
        arch_pct_decrease = (arch_max - arch_min) / arch_max * 100 if arch_max > 0 else 0

        # Check monotonicity up to elbow for arch
        arch_is_monotonic = False
        if arch_elbow_k is not None:
            elbow_idx = np.where(k_vals == arch_elbow_k)[0][0]
            arch_is_monotonic = all(arch_losses[i] >= arch_losses[i+1] for i in range(elbow_idx))

        # K differential
        k_differential = abs(recon_elbow_k - arch_elbow_k) if (recon_elbow_k and arch_elbow_k) else np.nan

        results.append({
            'cluster_id': cluster_id,
            'aanet_recon_loss_elbow_k': recon_elbow_k,
            'aanet_recon_loss_elbow_strength': recon_elbow_strength,
            'recon_pct_decrease': recon_pct_decrease,
            'recon_is_monotonic': recon_is_monotonic,
            'aanet_archetypal_loss_elbow_k': arch_elbow_k,
            'aanet_archetypal_loss_elbow_strength': arch_elbow_strength,
            'arch_pct_decrease': arch_pct_decrease,
            'arch_is_monotonic': arch_is_monotonic,
            'k_differential': k_differential
        })

    # Convert to DataFrame
    elbow_df = pd.DataFrame(results)

    # Merge with original data (keeping all rows from original, but each cluster gets same elbow metrics)
    df_merged = df.merge(elbow_df, on='cluster_id', how='left')

    # Save back to CSV
    df_merged.to_csv(csv_path, index=False)

    # Print summary statistics
    print(f"\nElbow Quality Metrics Summary for n_clusters={n_clusters}:")
    print(f"  Total clusters analyzed: {len(elbow_df)}")
    print(f"\n  Reconstruction Loss:")
    print(f"    Monotonic to elbow: {elbow_df['recon_is_monotonic'].sum()} / {len(elbow_df)} ({100*elbow_df['recon_is_monotonic'].sum()/len(elbow_df):.1f}%)")
    print(f"    >=20% decrease: {(elbow_df['recon_pct_decrease'] >= 20).sum()} / {len(elbow_df)} ({100*(elbow_df['recon_pct_decrease'] >= 20).sum()/len(elbow_df):.1f}%)")
    print(f"    Monotonic AND >=20% decrease: {(elbow_df['recon_is_monotonic'] & (elbow_df['recon_pct_decrease'] >= 20)).sum()} / {len(elbow_df)} ({100*(elbow_df['recon_is_monotonic'] & (elbow_df['recon_pct_decrease'] >= 20)).sum()/len(elbow_df):.1f}%)")
    print(f"\n  Archetypal Loss:")
    print(f"    Monotonic to elbow: {elbow_df['arch_is_monotonic'].sum()} / {len(elbow_df)} ({100*elbow_df['arch_is_monotonic'].sum()/len(elbow_df):.1f}%)")
    print(f"    >=20% decrease: {(elbow_df['arch_pct_decrease'] >= 20).sum()} / {len(elbow_df)} ({100*(elbow_df['arch_pct_decrease'] >= 20).sum()/len(elbow_df):.1f}%)")
    print(f"    Monotonic AND >=20% decrease: {(elbow_df['arch_is_monotonic'] & (elbow_df['arch_pct_decrease'] >= 20)).sum()} / {len(elbow_df)} ({100*(elbow_df['arch_is_monotonic'] & (elbow_df['arch_pct_decrease'] >= 20)).sum()/len(elbow_df):.1f}%)")
    print(f"\n  Both recon AND arch monotonic: {(elbow_df['recon_is_monotonic'] & elbow_df['arch_is_monotonic']).sum()} / {len(elbow_df)} ({100*(elbow_df['recon_is_monotonic'] & elbow_df['arch_is_monotonic']).sum()/len(elbow_df):.1f}%)")
    print(f"  K differential <= 1: {(elbow_df['k_differential'] <= 1).sum()} / {len(elbow_df)} ({100*(elbow_df['k_differential'] <= 1).sum()/len(elbow_df):.1f}%)")


if __name__ == "__main__":
    main()
