
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

def test_sampler():
    print("Loading model...")
    # Use a tiny model for testing sampler logic if possible, but we need tokenizer.
    # gpt2 is small and fast.
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    iterator = iter(dataset)
    
    print("Sampling...")
    tokens_list = []
    batch_size = 2
    sample_len = 10
    
    while len(tokens_list) < batch_size:
        try:
            text = next(iterator)["text"]
            tokens = model.to_tokens(text, prepend_bos=True).squeeze(0)
            if len(tokens) >= sample_len:
                tokens_list.append(tokens[:sample_len])
        except StopIteration:
            break
            
    tokens_tensor = torch.stack(tokens_list)
    print(f"Sampled shape: {tokens_tensor.shape}")
    assert tokens_tensor.shape == (batch_size, sample_len)
    print("Sampler test passed!")

def test_sae_loading():
    print("Testing SAE loading...")
    # This might fail if not downloaded, but let's see if it finds the release
    try:
        sae, _, _ = SAE.from_pretrained("gemma-scope-2b-pt-res", "layer_12/width_16k/canonical", device="cpu")
        print("SAE loaded successfully!")
        print(f"SAE config: {sae.cfg.d_sae}")
    except Exception as e:
        print(f"SAE loading failed: {e}")

if __name__ == "__main__":
    test_sampler()
    # test_sae_loading() # Skip to avoid long download if not needed immediately
