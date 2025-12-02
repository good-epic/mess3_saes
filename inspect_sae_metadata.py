
from sae_lens import SAE
import torch

def inspect_sae():
    print("Loading SAE...")
    # Use the one the user is interested in
    release = "gemma-scope-9b-pt-res"
    sae_id = "layer_20/width_32k/average_l0_57"
    
    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(release, sae_id, device="cpu")
        
        print("\n--- SAE Config ---")
        print(sae.cfg)
        
        print("\n--- Sparsity Object ---")
        print(sparsity)
        
        print("\n--- Feature Frequencies ---")
        # Check for common attributes for feature frequencies
        if hasattr(sae, "feature_frequencies"):
            print(f"Found feature_frequencies with shape: {sae.feature_frequencies.shape}")
            print(f"Sample: {sae.feature_frequencies[:10]}")
        elif sparsity is not None:
             print("Checking sparsity tensor...")
             # sparsity might be the feature frequencies?
             if isinstance(sparsity, torch.Tensor):
                 print(f"Sparsity tensor shape: {sparsity.shape}")
                 print(f"Sample: {sparsity[:10]}")
             else:
                 print(f"Sparsity object type: {type(sparsity)}")
        else:
            print("No obvious feature frequencies found.")
            
    except Exception as e:
        print(f"Error loading SAE: {e}")

if __name__ == "__main__":
    inspect_sae()
