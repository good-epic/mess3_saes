
import os
import argparse
import json
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from clustering import ClusteringConfig
from clustering.config import GeometryFittingConfig
from aanet_pipeline import build_aanet_datasets, compute_diffusion_extrema, train_aanet_model, TrainingConfig, ExtremaConfig
from aanet_pipeline.cluster_summary import AAnetDescriptor
import jax
from huggingface_hub import login

from real_data_utils import RealDataSampler
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


    args = parser.parse_args()

    if args.hf_token:
        print("Logging in to Hugging Face with provided token...")
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        print("Logging in to Hugging Face with HF_TOKEN environment variable...")
        login(token=os.environ["HF_TOKEN"])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading Model: {args.model_name}")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)
    
    print(f"Loading SAE: {args.sae_release} - {args.sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(args.sae_release, args.sae_id, device=args.device)
    print(f"SAE Config: {cfg_dict}")
    print(f"SAE Sparsity: {sparsity}")
    print(f"SAE Hook Name: {sae.cfg.hook_name}")

    print("Initializing RealDataSampler...")
    sampler = RealDataSampler(model, seed=args.seed)
    

    # Determine site and hook_name from SAE config
    if hasattr(sae.cfg, "hook_name"):
        hook_name = sae.cfg.hook_name
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
        print("Warning: sae.cfg.hook_name not found. Attempting to parse from sae_id.")
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
        sae.cfg.hook_name = hook_name

    for n_clusters in args.n_clusters_list:
        print(f"\n--- Running Analysis for n_clusters={n_clusters} ---")
        
        clustering_config = ClusteringConfig(
            site=site,
            method="k_subspaces",
            subspace_params={
                "n_clusters": n_clusters,
            },
            sampling_config={
                "sample_sequences": 1024,
                "activation_batches": 1024,
                "max_activations": args.total_samples,
                "sample_seq_len": 128,
                "latent_activity_threshold": args.latent_activity_threshold,
            },
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
            epochs=100,
            batch_size=256,
            learning_rate=1e-3,
        )
        
        for desc in descriptors:
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
                    config=ExtremaConfig(enabled=True)
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
                
if __name__ == "__main__":
    main()
