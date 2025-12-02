
import torch
from sae_lens import SAE
from mess3_gmg_analysis_utils import sae_encode_features, sae_decode_features

def test_sae_encoding():
    release = "gemma-scope-9b-pt-res"
    sae_id = "layer_20/width_32k/average_l0_57"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SAE: {release} - {sae_id}")
    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(release, sae_id, device=device)
    
    d_in = sae.cfg.d_in
    d_sae = sae.cfg.d_sae
    batch_size = 10
    
    print("Creating dummy activations...")
    acts = torch.randn(batch_size, d_in, device=device)
    
    print("Testing sae_encode_features...")
    feature_acts, x_mean, x_std = sae_encode_features(sae, acts)
    
    print(f"Feature acts shape: {feature_acts.shape}")
    assert feature_acts.shape == (batch_size, d_sae)
    assert x_mean is None
    assert x_std is None
    
    print("Testing sae_decode_features...")
    recon = sae_decode_features(sae, feature_acts, x_mean, x_std)
    
    print(f"Reconstruction shape: {recon.shape}")
    assert recon.shape == (batch_size, d_in)
    
    print("VERIFICATION PASSED")

if __name__ == "__main__":
    test_sae_encoding()
