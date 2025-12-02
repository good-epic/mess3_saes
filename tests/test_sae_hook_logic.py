
import torch
from sae_lens import SAE
from clustering import ClusteringConfig
from real_data_tests.real_pipeline import RealDataClusteringPipeline
from clustering.config import SubspaceParams

def test_sae_hook_logic():
    print("Testing SAE hook logic...")
    
    # Use a small SAE
    release = "gemma-scope-2b-pt-res"
    sae_id = "layer_0/width_16k/average_l0_105" # A small one
    device = "cpu" # Keep it simple/fast if possible, or cuda if available
    if torch.cuda.is_available():
        device = "cuda"
        
    print(f"Loading SAE: {release} - {sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(release, sae_id, device=device)
    
    print(f"SAE Config Metadata keys: {sae.cfg.metadata.keys()}")
    if "hook_name" in sae.cfg.metadata:
        print(f"Metadata hook_name: {sae.cfg.metadata['hook_name']}")
    
    # Create dummy config
    config = ClusteringConfig(
        method="k_subspaces",
        site="layer_0",
        selected_k=0,
        subspace_params=SubspaceParams(n_clusters=10)
    )
    
    # Instantiate pipeline
    print("Instantiating RealDataClusteringPipeline...")
    try:
        pipeline = RealDataClusteringPipeline(sae=sae, config=config)
        print(f"Success! Pipeline hook_name: {pipeline.hook_name}")
        
        expected_hook = "blocks.0.hook_resid_post"
        if pipeline.hook_name == expected_hook:
            print("Verification PASSED: hook_name matches expected.")
        else:
            print(f"Verification FAILED: Expected {expected_hook}, got {pipeline.hook_name}")
            
    except Exception as e:
        print(f"Verification FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sae_hook_logic()
