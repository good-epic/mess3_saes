import os
# Force JAX to use CPU to avoid eigendecomposition issues on GPU
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import jax

from sae_variants.banded_cov_sae import BandedCovarianceSAE
from multipartite_utils import MultipartiteSampler, _resolve_device
from training_and_analysis_utils import _generate_sequences

# Force JAX to use CPU to avoid eigendecomposition issues on GPU
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax


def load_banded_sae(sae_path: str, device: str = "cpu") -> BandedCovarianceSAE:
    """Load a BandedCovarianceSAE from a checkpoint."""
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE checkpoint not found at {sae_path}")
    
    ckpt = torch.load(sae_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["cfg"])
    cfg["device"] = "cuda" if device.startswith("cuda") else "cpu"
    
    sae = BandedCovarianceSAE(cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae

def extract_parameters(sae: BandedCovarianceSAE) -> Dict[str, np.ndarray]:
    """Extract alpha, beta, and compute coupling coefficients."""
    if sae.use_alpha:
        alpha = sae.alpha.detach().cpu().numpy()
    else:
        alpha = sae.alpha.cpu().numpy()
        
    if sae.use_beta:
        beta = sae.beta.detach().cpu().numpy()
    else:
        beta = sae.beta.cpu().numpy()
        
    beta_slope = sae.beta_slope
    p = sae.p
    
    slope_denom = beta_slope if beta_slope != 0 else 1.0
    alpha_term = np.tanh(alpha / slope_denom)
    
    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_term": alpha_term,
        "p": p,
        "beta_slope": beta_slope
    }

def compute_coupling_matrix(params: Dict[str, np.ndarray], dict_size: int) -> np.ndarray:
    """Compute the effective coupling matrix C where x_i approx sum C_{i,k} x_{i-k}."""
    alpha_term = params["alpha_term"]
    beta = params["beta"]
    p = params["p"]
    
    coupling_coeffs = np.zeros((dict_size, p))
    
    for k in range(p):
        lag = k + 1
        term = np.power(alpha_term, lag) * beta[k]
        coupling_coeffs[:, k] = term
        
    return coupling_coeffs

def compute_implied_correlation_matrix(coupling_coeffs: np.ndarray, dict_size: int, p: int) -> np.ndarray:
    """
    Compute theoretical correlation matrix, masked to be 0 if |i-j| > p.
    """
    # We only need to compute correlations for |i-j| <= p.
    # However, the user asked to "look up the pairwise correlation... if abs(i-j) > p, estimated correlation is 0".
    # This implies we should compute the full implied correlation (or at least the band) and then mask it.
    # Since we need to do this for *all pairs* within a cluster (which might be far apart), 
    # we actually need the *full* implied correlation matrix, but then we *artificially* zero out entries outside the band.
    # Wait, the user said: "our estimated correlations only go back p steps... So it should be easy to look up... if abs > p ... is 0".
    # This suggests the user *believes* the model only models correlations up to lag p.
    # In reality, an AR(p) process has infinite impulse response, so correlations decay but don't become zero.
    # BUT, the user explicitly asked to treat them as 0 outside lag p. I will follow this instruction.
    
    # So we compute the full correlation matrix (or a large block) and then apply the mask.
    # Since computing full 4096 x 4096 inverse is expensive-ish but doable (done in previous step), let's do it.
    
    # Construct (I - A)
    I_minus_A = np.eye(dict_size)
    for i in range(dict_size):
        for k_idx in range(coupling_coeffs.shape[1]):
            lag = k_idx + 1
            j = i - lag
            if j >= 0:
                I_minus_A[i, j] = -coupling_coeffs[i, k_idx]
                
    try:
        inv_matrix = np.linalg.inv(I_minus_A)
        sigma = inv_matrix @ inv_matrix.T
        diag = np.sqrt(np.diag(sigma))
        outer_diag = np.outer(diag, diag)
        full_corr = sigma / (outer_diag + 1e-8)
        
        # Apply mask: set to 0 if |i-j| > p
        # We can do this efficiently
        mask = np.abs(np.arange(dict_size)[:, None] - np.arange(dict_size)[None, :]) <= p
        masked_corr = full_corr * mask
        return masked_corr
        
    except np.linalg.LinAlgError:
        print("Warning: Matrix inversion failed. Returning Identity.")
        return np.eye(dict_size)

def get_ground_truth_clusters(
    sae: BandedCovarianceSAE,
    sampler: MultipartiteSampler,
    batch_size: int = 2048,
    total_samples: int = 25000,
    site: str = "layer_1"
) -> Tuple[Dict[int, str], List[str]]:
    """
    Regress latents against ground truth components to assign clusters.
    Returns:
        latent_to_cluster: Dict[latent_idx, cluster_name]
        component_names: List[str]
        activation_rates: np.ndarray
    """
    # 1. Collect data
    # We need activations and belief states.
    # Similar logic to latent_geometry_analyze_metrics.py
    
    # Generate sequences
    # We need to know the model to run the sampler? 
    # Actually MultipartiteSampler generates *observations* and *states*.
    # We need the transformer to get the *activations* at the site.
    # The user didn't explicitly say to load the transformer, but we need it to get SAE inputs.
    # Wait, we can just use the sampler to get the *inputs* to the transformer, run the transformer, get activations.
    # Let's assume we have access to the transformer or can load it.
    # The previous script `latent_geometry_analyze_metrics.py` loads it.
    
    # For simplicity, I'll assume we can load the transformer using `_load_transformer`.
    from multipartite_utils import _load_transformer
    
    # Construct dummy args for _load_transformer
    class Args:
        model_ckpt = "outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt"
        d_model = 128
        n_heads = 4
        n_layers = 3
        n_ctx = 16
        d_head = 32
        d_vocab = None
        act_fn = "relu"
        
    print(f"Sampler vocab size: {sampler.vocab_size}")
    print(f"Component vocab sizes: {sampler.component_vocab_sizes}")
    
    vocab_size_hack = 432
    
    device = next(sae.parameters()).device
    model, _ = _load_transformer(Args(), device, vocab_size_hack)
    
    # Helper to sample batch
    rng_key = jax.random.PRNGKey(42)
    
    def sample_batch_helper(batch_size: int):
        nonlocal rng_key
        rng_key, beliefs, tokens, inputs_list = sampler.sample_python(rng_key, batch_size, Args.n_ctx)
        
        # Convert to numpy/torch
        # beliefs: (batch, seq, dim)
        # tokens: (batch, seq)
        
        # We need to structure beliefs by component name for the analysis
        beliefs_dict = {}
        offset = 0
        for i, comp in enumerate(sampler.components):
            dim = comp.state_dim
            # beliefs is concatenated.
            # Wait, sample_python returns concatenated beliefs.
            # We need to slice it back.
            # Or use inputs_list? No, inputs are observations.
            # We need states.
            
            # Slice beliefs
            comp_belief = np.array(beliefs[..., offset:offset+dim])
            beliefs_dict[comp.name] = comp_belief
            offset += dim
            
        return {
            "tokens": np.array(tokens, dtype=np.int64),
            "beliefs": beliefs_dict
        }
    
    # Collect activations and beliefs
    all_acts = []
    all_beliefs = []
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"Collecting {total_samples} samples...")
    
    with torch.no_grad():
        for _ in range(num_batches):
            batch = sample_batch_helper(batch_size)
            tokens = torch.tensor(batch["tokens"], device=device)
            
            # Get model activations
            _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.1.hook_resid_post"]) # Hardcoded for layer_1 for now
            site_act = cache[f"blocks.1.hook_resid_post"] # (batch, seq, d_model)
            
            # Get SAE feature activations
            # sae(input) -> feature_acts
            # We need to flatten batch and seq
            flat_site_act = site_act.reshape(-1, site_act.shape[-1])
            
            # Manual encode: acts = relu((x - b_dec) @ W_enc)
            # Note: BandedCovarianceSAE preprocesses input (centering), but here we just use raw weights?
            # forward() does: x_cent = x - self.b_dec; acts = F.relu(x_cent @ self.W_enc)
            # So we should do the same.
            x_cent = flat_site_act - sae.b_dec
            feature_acts = torch.relu(x_cent @ sae.W_enc) # (batch*seq, dict_size)
            
            all_acts.append(feature_acts.cpu().numpy())
            
            # Get beliefs
            # batch["beliefs"] is a dict of arrays (batch, seq, dim)
            # We need to flatten and concatenate them
            # Order matters! We need a consistent component order.
            # Sampler doesn't give order easily, but `batch["beliefs"]` keys are component names.
            # Let's sort them to be deterministic.
            comp_names = sorted(batch["beliefs"].keys())
            
            batch_beliefs = []
            for name in comp_names:
                b = batch["beliefs"][name] # (batch, seq, dim)
                b_flat = b.reshape(-1, b.shape[-1])
                batch_beliefs.append(b_flat)
            
            # Concatenate components: (batch*seq, total_belief_dim)
            batch_beliefs_concat = np.concatenate(batch_beliefs, axis=1)
            all_beliefs.append(batch_beliefs_concat)
            
    all_acts = np.concatenate(all_acts, axis=0)[:total_samples]
    all_beliefs = np.concatenate(all_beliefs, axis=0)[:total_samples]
    
    # 2. Regress
    print("Running regressions...")
    latent_to_cluster = {}
    
    # We need to know which columns of all_beliefs correspond to which component
    # Re-construct slices
    slices = {}
    offset = 0
    # We need a sample batch to know dimensions
    sample_batch = sample_batch_helper(1)
    comp_names = sorted(sample_batch["beliefs"].keys())
    for name in comp_names:
        dim = sample_batch["beliefs"][name].shape[-1]
        slices[name] = (offset, offset + dim)
        offset += dim
        
    # For each latent, find best component R2
    # We can do this efficiently?
    # LinearRegression for each latent against each component is expensive if done naively.
    # But we can do it per component.
    
    num_latents = sae.cfg["dict_size"]
    best_r2 = np.zeros(num_latents) - 1.0
    best_comp = np.array(["unassigned"] * num_latents, dtype=object)
    
    for name in comp_names:
        start, end = slices[name]
        targets = all_beliefs[:, start:end]
        
        # Fit all latents against this component?
        # Or fit this component against all latents?
        # "max single variable belief state prediction is component X"
        # Means: for latent i, predict y_c. R2(i, c). Maximize over c.
        
        # We can use correlation if targets are 1D, but they are multi-dimensional.
        # So we need a regression.
        # y = Xw + b. y is latent (N,), X is component belief (N, dim).
        # Wait, usually we predict belief FROM latent. "predicting the true belief state from the generating process"
        # "max single variable belief state prediction" -> Latent predicts Belief.
        # y = Belief, x = Latent.
        # R2 of predicting Belief using Latent.
        
        # Since Belief is multi-dimensional, we likely mean the average R2 across dimensions, or R2 of the vector?
        # Usually "variance explained".
        # Let's use the standard R2 score for multioutput regression.
        
        # To do this efficiently for 4096 latents:
        # For each latent x_i:
        #   Fit y_c ~ x_i.
        #   Calculate R2.
        
        # This is still a lot of loops.
        # Optimization:
        # y_c (N, d), x_i (N, 1).
        # w = (x^T x)^-1 x^T y.
        # Since x is 1D (plus bias), this is fast.
        
        # Precompute centered y and x to avoid bias term in loop?
        # Or just use sklearn for simplicity if it's fast enough. 4096 is small.
        
        print(f"  Fitting {name}...")
        reg = LinearRegression()
        # We can fit all latents at once? No, X is different for each latent.
        # But we can invert the problem? No.
        
        # Let's just loop. It's 4096 latents * 10 components = 40k regressions. 
        # With N=25000, it might take a minute.
        
        # Vectorized approach:
        # Centered X (latents): (N, L)
        # Centered Y (belief): (N, D)
        # Slope w = (x . y) / (x . x)  (for 1D x)
        # w shape (L, D)
        # Pred Y_hat = x * w
        # R2 = 1 - sum(y - y_hat)^2 / sum(y - mean)^2
        
        X = all_acts - all_acts.mean(axis=0) # (N, L)
        Y = targets - targets.mean(axis=0)   # (N, D)
        
        # Variance of X
        var_x = np.sum(X**2, axis=0) # (L,)
        
        # Covariance X, Y
        # (L, N) @ (N, D) -> (L, D)
        cov_xy = X.T @ Y
        
        # Slopes (L, D)
        # w = cov_xy / var_x[:, None]
        # Handle div by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            w = cov_xy / var_x[:, None]
            w[~np.isfinite(w)] = 0.0
            
        # Predictions: Y_hat = X[:, :, None] * w[None, :, :] -> (N, L, D)
        # This is too big (25000 * 4096 * D).
        # Compute RSS directly?
        # RSS = sum((y - xw)^2) = sum(y^2) - 2w sum(xy) + w^2 sum(x^2)
        # sum(y^2) is (D,) -> sum over D -> scalar per latent?
        # We want R2 per latent.
        # R2 = 1 - RSS / TSS.
        # TSS = sum(y^2) (since centered).
        
        # We want R2 for the *vector* y.
        # R2_multivar = 1 - sum_d RSS_d / sum_d TSS_d
        
        TSS_d = np.sum(Y**2, axis=0) # (D,)
        total_TSS = np.sum(TSS_d)
        
        # RSS_d = sum_n (y_nd - x_n w_d)^2
        #       = sum_n y_nd^2 - 2 w_d sum_n x_n y_nd + w_d^2 sum_n x_n^2
        #       = TSS_d - 2 w_d cov_xy_d + w_d^2 var_x
        
        # We need RSS per latent.
        # RSS (L,) = sum_d (TSS_d - 2 w_ld cov_ld + w_ld^2 var_x_l)
        
        term2 = 2 * w * cov_xy # (L, D)
        term3 = (w**2) * var_x[:, None] # (L, D)
        
        RSS_d = TSS_d[None, :] - term2 + term3 # (L, D)
        total_RSS = np.sum(RSS_d, axis=1) # (L,)
        
        r2 = 1 - total_RSS / total_TSS
        
        # Update best
        better_mask = r2 > best_r2
        best_r2[better_mask] = r2[better_mask]
        best_comp[better_mask] = name
        
    # Create dict
    for i in range(num_latents):
        if best_r2[i] > 0.01: # Threshold to avoid noise assignment
            latent_to_cluster[i] = best_comp[i]
        else:
            latent_to_cluster[i] = "unassigned"
            
    # Calculate activation rates
    # all_acts: (total_samples, num_latents)
    activation_rates = (all_acts > 0).mean(axis=0)
    
    print(f"Component names found: {comp_names}")
    unique, counts = np.unique(best_comp, return_counts=True)
    print(f"Cluster assignment distribution: {dict(zip(unique, counts))}")
    
    return latent_to_cluster, comp_names, activation_rates

def analyze_correlations(
    sae: BandedCovarianceSAE,
    latent_to_cluster: Dict[int, str],
    comp_names: List[str],
    output_dir: str,
    suffix: str = ""
):
    params = extract_parameters(sae)
    coupling = compute_coupling_matrix(params, sae.cfg["dict_size"])
    corr_matrix = compute_implied_correlation_matrix(coupling, sae.cfg["dict_size"], sae.p)
    
    results = []
    
    for comp in comp_names:
        cluster_indices = [i for i, c in latent_to_cluster.items() if c == comp]
        if not cluster_indices:
            continue
            
        # Within-cluster correlation
        # Average of corr[i, j] for i, j in cluster, i != j
        # Since we masked |i-j| > p, many will be 0.
        # We include zeros in the average? "average of the pariwise estimated correlation between all pairs"
        # Yes.
        
        # Extract submatrix
        sub_corr = corr_matrix[np.ix_(cluster_indices, cluster_indices)]
        # Remove diagonal
        np.fill_diagonal(sub_corr, np.nan)
        within_mean = np.nanmean(sub_corr)
        
        # Between-cluster correlation
        # Average of corr[i, j] for i in cluster, j NOT in cluster
        non_cluster_indices = [i for i in range(sae.cfg["dict_size"]) if i not in cluster_indices]
        if not non_cluster_indices:
            between_mean = 0.0
        else:
            sub_corr_between = corr_matrix[np.ix_(cluster_indices, non_cluster_indices)]
            between_mean = np.mean(sub_corr_between)
            
        results.append({
            "component": comp,
            "within_corr": within_mean,
            "between_corr": between_mean,
            "size": len(cluster_indices)
        })
        
    df = pd.DataFrame(results)
    print(df)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="within_corr", y="between_corr", hue="component", size="size", sizes=(20, 200))
    plt.title("Banded Correlation Analysis by Ground Truth Cluster")
    plt.xlabel("Average Within-Cluster Correlation")
    plt.ylabel("Average Between-Cluster Correlation")
    plt.axline((0, 0), slope=1, color="gray", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"cluster_correlation_scatter{suffix}.png"))
    plt.close()
    
    # Save CSV
    df.to_csv(os.path.join(output_dir, f"cluster_correlation_stats{suffix}.csv"), index=False)

def plot_activation_rates(activation_rates: np.ndarray, output_dir: str):
    """Plot histogram of activation rates."""
    # Linear scale
    plt.figure(figsize=(10, 6))
    plt.hist(activation_rates, bins=50, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Latent Activation Rates (Linear Scale)")
    plt.xlabel("Activation Rate")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "activation_rate_histogram_linear.png"))
    plt.close()
    
    # Log scale (x-axis) for non-zero rates
    nonzero_rates = activation_rates[activation_rates > 0]
    if len(nonzero_rates) > 0:
        plt.figure(figsize=(10, 6))
        # Use log-spaced bins for log-scale histogram
        min_rate = nonzero_rates.min()
        max_rate = nonzero_rates.max()
        bins = np.logspace(np.log10(min_rate), np.log10(max_rate), 50)
        
        plt.hist(nonzero_rates, bins=bins, edgecolor='black', alpha=0.7)
        plt.xscale('log')
        plt.title("Distribution of Latent Activation Rates (Log Scale)")
        plt.xlabel("Activation Rate (Log Scale)")
        plt.ylabel("Count")
        plt.grid(axis='y', alpha=0.5)
        plt.savefig(os.path.join(output_dir, "activation_rate_histogram_log.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_path", type=str, default="outputs/saes/multipartite_003_bsae/layer_1_banded_ls_0p005__la_0p005.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/reports/multipartite_003e/ground_truth_diagnostics")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading SAE from {args.sae_path}...")
    sae = load_banded_sae(args.sae_path, args.device)
    
    print("Initializing sampler...")
    # We need to construct the sampler.
    # Assuming we can use the config from the checkpoint or default?
    # The user's environment has `process_configs.json`.
    # We'll use the default config name "3xmess3_2xtquant_003" as seen in other scripts.
    
    from multipartite_utils import MultipartiteSampler, _load_process_stack
    import json
    
    with open("process_configs.json", "r") as f:
        all_configs = json.load(f)
        proc_cfg = all_configs["3xmess3_2xtquant_003"]
        
    print(f"Loaded proc_cfg: {proc_cfg}")
    
    from multipartite_utils import build_components_from_config
    components = build_components_from_config(proc_cfg)
    print(f"Directly built components: {[c.name for c in components]}")
    
    sampler = MultipartiteSampler(components)
    print(f"Sampler components: {[c.name for c in sampler.components]}")
    
    # Calculate activation rates
    # all_acts: (total_samples, dict_size)
    # We need to re-calculate all_acts or return it from get_ground_truth_clusters
    # Currently get_ground_truth_clusters does NOT return all_acts.
    # I should modify get_ground_truth_clusters to return activation rates or all_acts.
    
    # Let's modify get_ground_truth_clusters to return activation rates.
    
    print("Assigning clusters...")
    latent_to_cluster, comp_names, activation_rates = get_ground_truth_clusters(sae, sampler, site="layer_1")
    
    print("Plotting activation rate histogram...")
    plot_activation_rates(activation_rates, args.output_dir)
    
    cutoffs = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    
    for cutoff in cutoffs:
        print(f"\nAnalyzing with activation rate cutoff: {cutoff}")
        
        # Filter latents
        filtered_latent_to_cluster = {
            i: c for i, c in latent_to_cluster.items() 
            if activation_rates[i] > cutoff
        }
        
        if not filtered_latent_to_cluster:
            print(f"No latents meet cutoff {cutoff}. Skipping.")
            continue
            
        print(f"Retained {len(filtered_latent_to_cluster)} / {len(latent_to_cluster)} latents")
        
        analyze_correlations(sae, filtered_latent_to_cluster, comp_names, args.output_dir, suffix=f"_cutoff_{cutoff}")
    
    print("Done!")

if __name__ == "__main__":
    main()
