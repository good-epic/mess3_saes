
import torch
import numpy as np
from datasets import load_dataset
from transformer_lens import HookedTransformer
from mess3_gmg_analysis_utils import sae_encode_features, sae_decode_features
from typing import Dict, Sequence, Tuple
from aanet_pipeline.cluster_summary import AAnetDescriptor
from aanet_pipeline.cluster_summary import AAnetDescriptor, AAnetDatasetResult
from BatchTopK.sae import TopKSAE

from tqdm import tqdm
import gc

from tqdm import tqdm
import gc

def _prepare_cluster_indices(
    descriptors: Sequence[AAnetDescriptor],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    index_map: Dict[int, torch.Tensor] = {}
    for desc in descriptors:
        if desc.latent_indices:
            index_map[desc.cluster_id] = torch.tensor(desc.latent_indices, device=device, dtype=torch.long)
        else:
            index_map[desc.cluster_id] = torch.empty((0,), device=device, dtype=torch.long)
    return index_map


class RealDataSampler:
    """
    Sampler for real text data (e.g. OpenWebText, The Pile, Wikitext).
    Supports streaming and flexible batch sampling.
    """
    def __init__(self, model: HookedTransformer, dataset_name: str = "wikitext", dataset_config: str = "wikitext-103-v1", split: str = "train", streaming: bool = True, seed: int = 42):
        self.model = model
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.streaming = streaming
        self.seed = seed
        
        print(f"Loading dataset: {dataset_name} ({dataset_config}) split={split} streaming={streaming}")
        self.dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
        if streaming:
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)
        else:
            self.dataset = self.dataset.shuffle(seed=seed)
            
        self.iterator = iter(self.dataset)
        
    def sample_tokens_batch(self, batch_size: int, sample_len: int, device: str) -> torch.Tensor:
        """
        Sample a batch of tokens from the dataset.
        """
        tokens_list = []
        while len(tokens_list) < batch_size:
            try:
                item = next(self.iterator)
                text = item.get("text", "")
                if not text:
                    continue
                    
                # Tokenize
                # prepend_bos=True is standard for many models, check if needed for Gemma 2
                # HookedTransformer usually handles this if configured, but let's be explicit if needed.
                # Gemma 2 usually expects BOS.
                tokens = self.model.to_tokens(text, prepend_bos=True).squeeze(0)
                
                # If sequence is long enough, take a chunk
                if len(tokens) >= sample_len:
                    # We could take multiple chunks from one text if it's very long, 
                    # but for simplicity and i.i.d assumptions, let's take one per doc for now 
                    # or maybe a random chunk?
                    # Taking the beginning is biased towards beginnings of docs.
                    # Let's take a random chunk if it's much longer.
                    if len(tokens) > sample_len + 10:
                        start_idx = np.random.randint(0, len(tokens) - sample_len)
                        tokens_list.append(tokens[start_idx : start_idx + sample_len])
                    else:
                        tokens_list.append(tokens[:sample_len])
            except StopIteration:
                # Reset iterator
                self.iterator = iter(self.dataset)
                
        return torch.stack(tokens_list).to(device)

def collect_real_activity_stats(
    model,
    sae,
    sampler: RealDataSampler,
    hook_name: str,
    batch_size: int,
    sample_len: int,
    n_batches: int,
    activation_eps: float,
    device: str,
    collect_matrix: bool = False,
):
    """
    Collect latent activity statistics for real data.
    Adapted from multipartite_utils.collect_latent_activity_data but for RealDataSampler.
    """
    # Determine dict_size
    dict_size = None
    if hasattr(sae, "cfg"):
        if hasattr(sae.cfg, "dict_size"):
            dict_size = sae.cfg.dict_size
        elif hasattr(sae.cfg, "d_sae"):
            dict_size = sae.cfg.d_sae
            
    if dict_size is None:
        if hasattr(sae, "dict_size"):
            dict_size = sae.dict_size
        elif hasattr(sae, "d_sae"):
            dict_size = sae.d_sae
        elif hasattr(sae, "W_dec"):
             dict_size = sae.W_dec.shape[0]
        else:
            raise ValueError("Could not determine dict_size from SAE object")

    active_counts = torch.zeros(dict_size, dtype=torch.float64)
    mean_abs_sum = torch.zeros(dict_size, dtype=torch.float64)
    total_samples = 0
    binary_batches = [] if collect_matrix else []

    print(f"Collecting activity stats: {n_batches} batches of size {batch_size}")
    
    print_interval = max(1, n_batches // 30)
    for i in range(n_batches):
        if i % print_interval == 0 or i == n_batches - 1:
            print(f"Collecting Activity Stats: {i}/{n_batches} ({(i/n_batches)*100:.1f}%)")
        tokens = sampler.sample_tokens_batch(batch_size, sample_len, device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[hook_name])
            acts = cache[hook_name]
            # Flatten: (batch * seq, d_model)
            acts_flat = acts.reshape(-1, acts.shape[-1])
            
            # Encode
            # sae_encode_features expects (sae, acts)
            # It returns (feature_acts, x_mean, x_std) usually?
            # Let's check mess3_gmg_analysis_utils.sae_encode_features signature if possible, 
            # or just use sae(acts) if it's standard sae_lens.
            # sae_lens SAE forward returns a dict usually.
            # But sae_encode_features is a utility in this repo.
            feature_acts, _, _ = sae_encode_features(sae, acts_flat)
            
        mask = (feature_acts.abs() > activation_eps)
        active_counts += mask.sum(dim=0).to(torch.float64).cpu()
        mean_abs_sum += feature_acts.abs().sum(dim=0).to(torch.float64).cpu()
        total_samples += mask.shape[0]

        if collect_matrix:
            binary_batches.append(mask.cpu())
            
        del tokens, cache, acts, acts_flat, feature_acts, mask
        gc.collect()
        torch.cuda.empty_cache()
            
    activity_rates = (active_counts / max(total_samples, 1)).numpy()
    mean_abs_activation = (mean_abs_sum / max(total_samples, 1)).numpy()
    latent_matrix = None
    if collect_matrix:
        latent_matrix = torch.cat(binary_batches, dim=0).numpy()
        
    return {
        "activity_rates": activity_rates,
        "mean_abs_activation": mean_abs_activation,
        "latent_matrix": latent_matrix,
        "total_samples": total_samples,
    }

def build_real_aanet_datasets(
    *,
    model,
    sampler: RealDataSampler,
    layer_hook: str,
    sae: TopKSAE,
    aanet_descriptors: Sequence[AAnetDescriptor],
    batch_size: int,
    seq_len: int,
    num_batches: int,
    activation_threshold: float,
    device: torch.device,
    max_samples_per_cluster: int | None = None,
    min_cluster_samples: int = 0,
    seed: int = 0,
    token_positions: Sequence[int] | None = None,
) -> Dict[int, AAnetDatasetResult]:
    """
    Build AAnet datasets using RealDataSampler, avoiding JAX dependencies.
    """
    sae.eval()
    model.eval()

    # Note: seed is unused here as RealDataSampler handles its own seeding/shuffling
    
    cluster_indices = _prepare_cluster_indices(aanet_descriptors, device)
    storage: Dict[int, list[torch.Tensor]] = {desc.cluster_id: [] for desc in aanet_descriptors}
    total_counts: Dict[int, int] = {desc.cluster_id: 0 for desc in aanet_descriptors}
    kept_counts: Dict[int, int] = {desc.cluster_id: 0 for desc in aanet_descriptors}

    for _ in tqdm(range(num_batches), desc="Building AAnet Datasets"):
        tokens = sampler.sample_tokens_batch(batch_size, seq_len, str(device))

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None, names_filter=[layer_hook])
            acts = cache[layer_hook]
        if token_positions:
            seq_len_current = acts.shape[1]
            for pos in token_positions:
                if pos < 0 or pos >= seq_len_current:
                    raise ValueError(f"Token index {pos} is out of bounds for sequence length {seq_len_current}")
            acts = acts[:, token_positions, :]
        acts_flat = acts.reshape(-1, acts.shape[-1]).to(device)
        feature_acts, x_mean, x_std = sae_encode_features(sae, acts_flat)
        feature_acts = feature_acts.detach()
        zero_template = torch.zeros_like(feature_acts, device=device)

        for desc in aanet_descriptors:
            indices = cluster_indices[desc.cluster_id]
            if indices.numel() == 0:
                continue
            subset = feature_acts[:, indices]
            if activation_threshold > 0.0:
                active_mask = (subset > activation_threshold).any(dim=1)
            else:
                active_mask = (subset > 0).any(dim=1)
            total_counts[desc.cluster_id] += int(feature_acts.shape[0])
            kept = int(active_mask.sum().item())
            if kept == 0:
                continue
            kept_counts[desc.cluster_id] += kept
            cluster_latents = zero_template.clone()
            cluster_latents[:, indices] = subset
            recon = sae_decode_features(sae, cluster_latents, x_mean, x_std)
            selected = recon[active_mask].detach().cpu()
            storage[desc.cluster_id].append(selected)

        del tokens, cache, acts, acts_flat, feature_acts, x_mean, x_std, zero_template, subset, active_mask, cluster_latents, recon, selected
        gc.collect()
        torch.cuda.empty_cache()

    results: Dict[int, AAnetDatasetResult] = {}
    for desc in aanet_descriptors:
        tensors = storage[desc.cluster_id]
        if tensors:
            data = torch.cat(tensors, dim=0)
        else:
            data = torch.empty((0, sae.cfg["act_size"]), dtype=torch.float32)
        if max_samples_per_cluster is not None and data.shape[0] > max_samples_per_cluster:
            perm = torch.randperm(data.shape[0])[:max_samples_per_cluster]
            data = data[perm]
        kept = kept_counts[desc.cluster_id]
        total = total_counts[desc.cluster_id]
        ignored_fraction = 1.0 - (kept / total) if total > 0 else 1.0
        results[desc.cluster_id] = AAnetDatasetResult(
            descriptor=desc,
            data=data,
            kept_samples=kept,
            total_samples=total,
            ignored_fraction=ignored_fraction,
        )

    for desc in aanet_descriptors:
        if results[desc.cluster_id].data.shape[0] < min_cluster_samples and min_cluster_samples > 0:
            # record class but keep data as-is; caller can decide how to handle
            pass

    return results
