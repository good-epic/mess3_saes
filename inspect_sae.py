
import torch
from sae_lens import SAE

def inspect_sae():
    release = "gemma-scope-9b-pt-res"
    sae_id = "layer_20/width_32k/average_l0_57"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SAE: {release} - {sae_id}")
    try:
        sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(release, sae_id, device=device)
        print(f"SAE Type: {type(sae)}")
        print("Attributes/Methods:")
        for attr in dir(sae):
            if not attr.startswith("__"):
                print(f"  {attr}")
                
        # Test encode/decode if they exist
        if hasattr(sae, "encode"):
            print("\nTesting encode...")
            dummy_input = torch.randn(1, sae.cfg.d_in, device=device)
            out = sae.encode(dummy_input)
            print(f"Encode output shape: {out.shape}")
            
        if hasattr(sae, "decode"):
            print("\nTesting decode...")
            dummy_features = torch.randn(1, sae.cfg.d_sae, device=device)
            recon = sae.decode(dummy_features)
            print(f"Decode output shape: {recon.shape}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_sae()
