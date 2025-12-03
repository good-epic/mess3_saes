
import torch
from transformer_lens import HookedTransformer
import sys

def debug_hooks():
    model_name = "gemma-2-2b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name} on {device}")
    print(f"Creating dummy model for: {model_name}")
    try:
        from transformer_lens import HookedTransformerConfig
        
        # Tiny Gemma 2 config
        cfg = HookedTransformerConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=8,
            n_heads=8,
            d_vocab=1000,
            act_fn="gelu", 
            normalization_type="RMS",
            model_name=model_name,
        )
        
        model = HookedTransformer(cfg)
        
        print(f"Dummy model created. n_layers: {model.cfg.n_layers}")
        print("Sample hook names (first 20):")
        all_hooks = list(model.hook_dict.keys())
        for hook in all_hooks[:20]:
            print(f"  {hook}")
            
        print("\nSearching for 'resid_post' hooks:")
        resid_hooks = [h for h in all_hooks if "resid_post" in h]
        for h in resid_hooks[:10]:
            print(f"  {h}")
            
        target_hook = "blocks.20.hook_resid_post"
        if target_hook in all_hooks:
            print(f"\nSUCCESS: '{target_hook}' found in model hooks.")
        else:
            print(f"\nFAILURE: '{target_hook}' NOT found in model hooks.")
            # Suggest alternatives
            print("Did you mean one of these?")
            for h in resid_hooks:
                if "20" in h:
                    print(f"  {h}")
                    
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    debug_hooks()
