
import os
from tqdm import tqdm
import argparse
import json
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from clustering.config import ClusteringConfig
from clustering.config import GeometryFittingConfig, SubspaceParams, SamplingConfig
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


def compute_cluster_activation_pca_ranks(
    model,
    sae,
    sampler,
    clustering_result,
    n_samples=5,
    batch_size=1024,
    seq_len=128,
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
        n_samples: Number of diverse batches to sample per cluster (default 5)
        batch_size: Batch size for forward passes (default 1024 for low variance PCA)
        seq_len: Sequence length (default 128)
        variance_threshold: Variance explained threshold for practical rank (default 0.95)
        device: Device to run on

    Returns:
        dict mapping cluster_id -> {"avg_rank": float, "avg_variance": float}
    """
    from sklearn.decomposition import PCA

    cluster_labels = clustering_result.cluster_labels
    n_clusters = clustering_result.n_clusters
    hook_name = sae.cfg.hook_name

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

        # Collect activations from n_samples diverse batches
        all_activations = []

        for sample_idx in range(n_samples):
            # Sample a batch of tokens
            tokens = sampler.sample_tokens_batch(batch_size, seq_len, device)

            with torch.no_grad():
                # Forward pass through model
                _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                acts = cache[hook_name]  # (batch, seq, d_model)
                acts_flat = acts.reshape(-1, acts.shape[-1])  # (batch*seq, d_model)

                # Encode with SAE
                feature_acts = sae_encode_features(sae, acts_flat)  # (batch*seq, d_sae)

                # Filter to only this cluster's latents
                cluster_acts = feature_acts[:, latent_indices]  # (batch*seq, n_cluster_latents)

                # Only keep rows where at least one latent is active
                active_mask = (cluster_acts.abs().sum(dim=1) > 1e-5)
                if active_mask.any():
                    cluster_acts_active = cluster_acts[active_mask].cpu().numpy()
                    all_activations.append(cluster_acts_active)

        if not all_activations:
            # No active samples found
            pca_results[cluster_id] = {"avg_rank": 0, "avg_variance": 0.0}
            continue

        # Concatenate all samples
        X = np.vstack(all_activations)  # (n_total_samples, n_cluster_latents)

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
            "avg_variance": float(variance_explained)
        }

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
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name for text sampling (e.g., 'wikitext', 'monology/pile-uncopyrighted')")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1", help="Dataset config (use 'default' for The Pile, None will be converted to default)")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--activity_batch_size", type=int, default=16, help="Batch size for activity stats")
    parser.add_argument("--activity_batches", type=int, default=1024, help="Number of batches for activity stats")
    parser.add_argument("--activity_seq_len", type=int, default=128, help="Sequence length for activity stats")
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
    parser.add_argument("--aanet_early_stop_patience", type=int, default=10, help="Early stopping patience based on validation loss.")
    parser.add_argument("--aanet_early_stop_delta", type=float, default=1e-4, help="Minimum improvement in validation loss to reset patience.")
    parser.add_argument("--aanet_lr_patience", type=int, default=5, help="ReduceLROnPlateau patience in epochs.")
    parser.add_argument("--aanet_lr_factor", type=float, default=0.5, help="Factor to reduce learning rate when plateau is detected.")
    parser.add_argument("--aanet_grad_clip", type=float, default=1.0, help="Gradient clipping norm (set <=0 to disable).")
    parser.add_argument("--aanet_restarts_no_extrema", type=int, default=3, help="Number of random restarts when no warm-start extrema are available.")
    parser.add_argument("--aanet_streaming_steps", type=int, default=1000, help="Number of streaming steps (batches) for AAnet training.")
    parser.add_argument("--aanet_warmup_steps", type=int, default=50, help="Number of batches to collect for extrema initialization.")
    parser.add_argument("--aanet_prefetch_size", type=int, default=1024, help="Prefetch buffer size for RealDataSampler.")
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
    # Handle dataset config (The Pile doesn't use a config, just pass None or "default")
    dataset_config = None if args.dataset_config.lower() in ["none", "default"] else args.dataset_config
    print(f"Using dataset: {args.dataset_name}, config: {dataset_config}, split: {args.dataset_split}")
    sampler = RealDataSampler(
        model,
        dataset_name=args.dataset_name,
        dataset_config=dataset_config,
        split=args.dataset_split,
        seed=args.seed,
        prefetch_size=args.aanet_prefetch_size
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
        
        clustering_config = ClusteringConfig(
            site=site,
            method="k_subspaces",
            selected_k=0,
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
                max_activations=args.total_samples, # Retained as it was not explicitly removed by the instruction
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
        
        # Try to load existing clustering result if resuming
        if args.resume_from and os.path.exists(clustering_pkl_path):
            print(f"Loading existing clustering result from {clustering_pkl_path}")
            try:
                import pickle
                with open(clustering_pkl_path, "rb") as f:
                    result = pickle.load(f)
                print(f"Loaded clustering result with {result.n_clusters} clusters")
            except Exception as e:
                print(f"Failed to load clustering result: {e}. Re-running clustering.")
                result = None

        if result is None:
            result = pipeline.run(
                model=model,
                cache={},
                data_source=sampler,
                site_dir=output_dir,
                component_beliefs_flat=None,
            )
            # Save clustering result
            import pickle
            with open(clustering_pkl_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Saved clustering result to {clustering_pkl_path}")

        # Compute activation PCA ranks if not already present
        if result.cluster_stats and not any("activation_pca_rank" in stats for stats in result.cluster_stats.values()):
            print(f"\nComputing activation PCA ranks for {result.n_clusters} clusters...")
            pca_ranks = compute_cluster_activation_pca_ranks(
                model=model,
                sae=sae,
                sampler=sampler,
                clustering_result=result,
                n_samples=5,
                batch_size=1024,
                seq_len=args.activity_seq_len,
                variance_threshold=0.95,
                device=args.device
            )

            # Add to cluster_stats
            for cid, rank_info in pca_ranks.items():
                if cid in result.cluster_stats:
                    result.cluster_stats[cid]["activation_pca_rank"] = rank_info["avg_rank"]
                    result.cluster_stats[cid]["activation_pca_variance_explained"] = rank_info["avg_variance"]

            # Re-save clustering result with PCA ranks
            with open(clustering_pkl_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Added PCA ranks to clustering result and saved to {clustering_pkl_path}")

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
                import gc; gc.collect(); torch.cuda.empty_cache()

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
                
                print(f"Starting Streaming Training for k={k} ({args.aanet_streaming_steps} steps)...")
                for step in tqdm(range(args.aanet_streaming_steps), desc=f"Streaming k={k}"):
                    # 1. Stream ONE batch
                    tokens = sampler.sample_tokens_batch(args.aanet_batch_size, args.activity_seq_len, args.device)
                    
                    with torch.no_grad():
                        _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                        acts = cache[hook_name]
                        acts_flat = acts.reshape(-1, acts.shape[-1])
                        feature_acts, _, _ = sae_encode_features(sae, acts_flat)
                    
                    # 2. Train (Distribute to all clusters in this trainer)
                    step_losses = trainer.train_step(feature_acts)
                    for cid, metrics in step_losses.items():
                        for key, value in metrics.items():
                            metrics_history[cid][key].append(value)
                    
                    # 3. Discard batch
                    del tokens, cache, acts, acts_flat, feature_acts
                
                # Save results for this k
                print(f"Saving results for k={k}...")
                for cid, model_inst in trainer.models.items():
                    history = metrics_history[cid]
                    final_metrics = {}
                    for key, values in history.items():
                        final_metrics[key] = np.mean(values[-10:]) if values else float("nan")
                    
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
                        if "explained_variance_ratio" in stats:
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
                
                # Cleanup trainer
                del trainer
                gc.collect()
                torch.cuda.empty_cache()

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
            
            for step in tqdm(range(args.aanet_streaming_steps), desc="Streaming"):
                # 1. Stream ONE batch
                tokens = sampler.sample_tokens_batch(args.aanet_batch_size, args.activity_seq_len, args.device)
                
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
                    acts = cache[hook_name]
                    acts_flat = acts.reshape(-1, acts.shape[-1])
                    feature_acts, _, _ = sae_encode_features(sae, acts_flat)
                
                # 2. Train (Distribute to ALL trainers)
                for k, trainer in trainers.items():
                    step_losses = trainer.train_step(feature_acts)
                    for cid, metrics in step_losses.items():
                        for key, value in metrics.items():
                            training_history[k][cid][key].append(value)
                
                # 3. Discard batch
                del tokens, cache, acts, acts_flat, feature_acts
            
            # Save Results (Concurrent)
            print("Saving results...")
            for k, trainer in trainers.items():
                for cid, model_inst in trainer.models.items():
                    history = training_history[k][cid]
                    final_metrics = {}
                    for key, values in history.items():
                        final_metrics[key] = np.mean(values[-10:]) if values else float("nan")
                    
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
                        if "explained_variance_ratio" in stats:
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

if __name__ == "__main__":
    main()
