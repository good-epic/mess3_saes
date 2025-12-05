
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

    # Extrema Arguments
    parser.add_argument("--extrema_enabled", dest="extrema_enabled", action="store_true", default=True, help="Enable Laplacian extrema warm start.")
    parser.add_argument("--no_extrema", dest="extrema_enabled", action="store_false", help="Disable Laplacian extrema warm start.")
    parser.add_argument("--extrema_knn", type=int, default=10, help="kNN value for Laplacian extrema.")
    parser.add_argument("--extrema_disable_subsample", action="store_true", help="Disable internal subsampling.")
    parser.add_argument("--extrema_max_points", type=int, default=10000, help="Maximum samples used for extrema computation.")
    parser.add_argument("--extrema_pca", type=float, default=None, help="PCA components (int > 1) or variance (float < 1) for extrema graph.")
    parser.add_argument("--extrema_seed", type=int, default=0, help="Seed for extrema subsampling.")

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
    sampler = RealDataSampler(model, seed=args.seed)
    

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
        
        result = pipeline.run(
            model=model,
            cache={}, 
            data_source=sampler,
            site_dir=os.path.join(args.output_dir, f"clusters_{n_clusters}"),
            component_beliefs_flat=None,
        )
        
        print(f"Running AAnet fitting for n_clusters={n_clusters}")
        
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
            
        datasets = build_real_aanet_datasets(
            model=model,
            sampler=sampler,
            layer_hook=hook_name,
            sae=sae,
            aanet_descriptors=descriptors,
            batch_size=32,
            seq_len=128,
            num_batches=10,
            activation_threshold=0.0,
            device=args.device
        )
        
        aanet_config = TrainingConfig(
            epochs=args.aanet_epochs,
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
            num_workers=args.aanet_num_workers,
            shuffle=True,
            val_fraction=args.aanet_val_fraction,
            min_val_size=args.aanet_val_min_size,
            early_stop_patience=args.aanet_early_stop_patience,
            early_stop_delta=args.aanet_early_stop_delta,
            lr_patience=args.aanet_lr_patience,
            lr_factor=args.aanet_lr_factor,
            grad_clip=args.aanet_grad_clip,
            restarts_no_extrema=args.aanet_restarts_no_extrema,
        )
        print_interval = max(1, len(descriptors) // 30)
        for i, desc in enumerate(descriptors):
            print(f"Fitting AAnet (n_clusters={n_clusters}): {i}/{len(descriptors)}")
            dataset = datasets[desc.cluster_id]
            if dataset.data.shape[0] < 32:
                print(f"Skipping cluster {desc.cluster_id} (too few samples)")
                continue
                
            data_tensor = dataset.data.to(args.device)
            
            extrema_tensor = None
            try:
                extrema_tensor = compute_diffusion_extrema(
                    dataset.data.cpu().numpy(),
                    max_k=8,
                    config=ExtremaConfig(
                        enabled=args.extrema_enabled,
                        knn=args.extrema_knn,
                        subsample=not args.extrema_disable_subsample,
                        max_points=args.extrema_max_points,
                        pca_components=args.extrema_pca,
                        random_seed=args.extrema_seed
                    )
                )
                if extrema_tensor is not None:
                    extrema_tensor = extrema_tensor.to(args.device)
            except Exception as e:
                print(f"Extrema computation failed for cluster {desc.cluster_id}: {e}")
            
            for k in range(2, 9):
                print(f"  Fitting AAnet k={k} for cluster {desc.cluster_id}")
                aanet_result = train_aanet_model(
                    data_tensor,
                    k=k,
                    config=aanet_config,
                    device=args.device,
                    diffusion_extrema=extrema_tensor
                )
                
                # Collect metrics for this run
                row = {
                    "n_clusters_total": n_clusters,
                    "cluster_id": desc.cluster_id,
                    "n_latents": len(desc.latent_indices),
                    "aanet_k": k,
                    "aanet_status": aanet_result.status,
                    "aanet_loss": aanet_result.metrics.get("loss_final"),
                    "aanet_recon_loss": aanet_result.metrics.get("reconstruction_loss_final"),
                    "aanet_arche_loss": aanet_result.metrics.get("archetypal_loss_final"),
                    "aanet_extrema_loss": aanet_result.metrics.get("extrema_loss", 0.0),
                    "aanet_val_loss": aanet_result.metrics.get("val_loss_final"),
                    "aanet_in_simplex": aanet_result.metrics.get("in_simplex_fraction"),
                }
                
                # Add clustering metrics if available
                if result.subspace_diagnostics:
                    # subspace_diagnostics might be flat or nested, let's check structure
                    # It likely contains 'reconstruction_errors' dict
                    recon_errors = result.subspace_diagnostics.get("reconstruction_errors", {})
                    row["cluster_recon_error"] = recon_errors.get(desc.cluster_id, float("nan"))
                    
                    cluster_ranks = result.subspace_diagnostics.get("cluster_ranks", {})
                    row["cluster_rank"] = cluster_ranks.get(desc.cluster_id, float("nan"))

                consolidated_results.append(row)

        # Save consolidated results to CSV
        if consolidated_results:
            import pandas as pd
            df = pd.DataFrame(consolidated_results)
            csv_path = os.path.join(args.output_dir, f"consolidated_metrics_n{n_clusters}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved consolidated metrics to {csv_path}")
            
            # Print top clusters by AAnet loss (for k=best?) or just dump
            print("\nTop clusters by AAnet reconstruction loss (k=best):")
            # Group by cluster_id and take min loss?
            # Or just print a few good ones
            df_ok = df[df["aanet_status"] == "ok"]
            if not df_ok.empty:
                df_sorted = df_ok.sort_values("aanet_recon_loss")
                print(df_sorted[["cluster_id", "aanet_k", "aanet_recon_loss", "cluster_recon_error"]].head(10))

if __name__ == "__main__":
    main()
