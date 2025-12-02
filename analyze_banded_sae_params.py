import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import json
import glob
import re

from sae_variants.banded_cov_sae import BandedCovarianceSAE

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

def load_metrics_summary(metrics_path: str) -> Dict:
    """Load the metrics summary JSON."""
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics summary not found at {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)

def get_active_rates(metrics_summary: Dict, site: str, sae_name: str, total_samples: int = 7680) -> np.ndarray:
    """Extract activation rates for a specific SAE."""
    try:
        # Navigate the nested structure: site -> sequence -> banded -> sae_name -> active_latents_last_quarter
        active_dict = metrics_summary[site]["sequence"]["banded"][sae_name]["active_latents_last_quarter"]
        
        # We don't know the exact dict size from metrics alone, but we can infer or resize later.
        # For now, return a map.
        rates = {}
        for idx_str, (count, _) in active_dict.items():
            rates[int(idx_str)] = count / total_samples
            
        return rates
    except KeyError:
        print(f"Warning: Could not find active rates for {site} {sae_name}")
        return {}

def extract_parameters(sae: BandedCovarianceSAE) -> Dict[str, np.ndarray]:
    """Extract alpha, beta, and compute coupling coefficients."""
    # Extract raw parameters
    if sae.use_alpha:
        alpha = sae.alpha.detach().cpu().numpy()
    else:
        alpha = sae.alpha.cpu().numpy() # Buffer
        
    if sae.use_beta:
        beta = sae.beta.detach().cpu().numpy()
    else:
        beta = sae.beta.cpu().numpy() # Buffer
        
    beta_slope = sae.beta_slope
    p = sae.p
    
    # Compute alpha term: tanh(alpha / beta_slope)
    # Handle beta_slope = 0 case
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
    
    # C is a (dict_size, p) matrix where C[i, k] is the coefficient for lag k+1
    # C[i, k] = (alpha_term[i])^(k+1) * beta[k]
    
    coupling_coeffs = np.zeros((dict_size, p))
    
    for k in range(p):
        # k is 0-indexed, corresponds to lag k+1
        lag = k + 1
        term = np.power(alpha_term, lag) * beta[k]
        coupling_coeffs[:, k] = term
        
    return coupling_coeffs

def compute_implied_correlation_matrix(coupling_coeffs: np.ndarray, subset_indices: List[int]) -> np.ndarray:
    """
    Compute theoretical correlation matrix for a subset of latents.
    Sigma = (I - A)^-1 * (I - A)^-T
    Corr_ij = Sigma_ij / sqrt(Sigma_ii * Sigma_jj)
    """
    start_idx = min(subset_indices)
    end_idx = max(subset_indices)
    size = end_idx - start_idx + 1
    
    # Construct (I - A) for this block
    I_minus_A = np.eye(size)
    
    for i in range(size):
        global_i = start_idx + i
        # For each lag k (1 to p)
        for k_idx in range(coupling_coeffs.shape[1]):
            lag = k_idx + 1
            global_j = global_i - lag
            
            # Check if neighbor is within our block
            if global_j >= start_idx:
                local_j = global_j - start_idx
                # A[i, j] = coupling_coeffs[global_i, k_idx]
                # (I - A)[i, j] = -A[i, j]
                I_minus_A[i, local_j] = -coupling_coeffs[global_i, k_idx]
                
    # Invert to get (I - A)^-1
    try:
        inv_matrix = np.linalg.inv(I_minus_A)
        # Sigma = inv * inv.T (assuming epsilon has identity covariance)
        sigma = inv_matrix @ inv_matrix.T
        
        # Normalize to correlation
        diag = np.sqrt(np.diag(sigma))
        outer_diag = np.outer(diag, diag)
        correlation = sigma / (outer_diag + 1e-8) # Avoid div by zero
        
        return correlation
    except np.linalg.LinAlgError:
        print("Warning: Matrix inversion failed. Returning Identity.")
        return np.eye(size)

def plot_analysis(params: Dict[str, np.ndarray], coupling_coeffs: np.ndarray, 
                 active_rates: Dict[int, float], output_dir: str, site: str, sae_name: str,
                 activation_threshold: float = 0.0):
    """Generate and save plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter latents based on activation rate
    active_indices = [i for i, rate in active_rates.items() if rate > activation_threshold]
    active_indices = sorted(active_indices)
    
    if not active_indices:
        print(f"No latents found with activation rate > {activation_threshold} for {site} {sae_name}")
        return

    print(f"Plotting for {len(active_indices)} active latents (threshold {activation_threshold})")

    # 1. Alpha/Correlation Histogram (Filtered)
    # Plot tanh(alpha/beta_slope) for active latents
    alpha_terms = params["alpha_term"][active_indices]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(alpha_terms, bins=50, kde=True)
    plt.title(f"Distribution of Alpha Terms (tanh(alpha/beta_slope)) - {site}\n{sae_name} (Active > {activation_threshold})")
    plt.xlabel("Alpha Term (Correlation Decay)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, f"{site}_{sae_name}_alpha_hist.png"))
    plt.close()
    
    # 2. Coupling Heatmap (Direct Coefficients)
    # Plot C[i, k] for active latents
    # X-axis: Latent Index (subset), Y-axis: Lag
    
    active_coupling = coupling_coeffs[active_indices, :].T # (p, num_active)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(active_coupling, cmap="coolwarm", center=0, cbar_kws={'label': 'Coupling Coefficient'})
    plt.title(f"Direct Coupling Coefficients (Lag vs Latent) - {site}\n{sae_name} (Active > {activation_threshold})")
    plt.xlabel("Latent Index (Filtered)")
    plt.ylabel("Lag (k)")
    plt.yticks(np.arange(active_coupling.shape[0]) + 0.5, np.arange(1, active_coupling.shape[0] + 1))
    plt.savefig(os.path.join(output_dir, f"{site}_{sae_name}_coupling_heatmap.png"))
    plt.close()
    
    # 3. Implied Correlation Heatmap
    # Compute full correlation matrix for first 4096 latents (or full dict if smaller)
    dict_size = coupling_coeffs.shape[0]
    if dict_size > 5000:
        print("Dictionary too large for full inversion, computing for first 4096...")
        calc_indices = list(range(4096))
    else:
        calc_indices = list(range(dict_size))
        
    full_corr = compute_implied_correlation_matrix(coupling_coeffs, calc_indices)
    
    # Slice to active indices (intersection with calc_indices)
    valid_active = [i for i in active_indices if i in calc_indices]
    
    if len(valid_active) > 1000:
        print(f"Too many active latents ({len(valid_active)}) for heatmap, subsampling to first 1000 active.")
        valid_active = valid_active[:1000]
    
    if not valid_active:
        print("No valid active latents for correlation heatmap.")
        return

    # Map global indices to local indices in full_corr
    local_indices = [i - calc_indices[0] for i in valid_active]
    active_corr = full_corr[np.ix_(local_indices, local_indices)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(active_corr, cmap="viridis", center=0, vmin=-1, vmax=1)
    plt.title(f"Implied Correlation Matrix - {site}\n{sae_name} (Active > {activation_threshold})")
    plt.xlabel("Latent Index (Filtered)")
    plt.ylabel("Latent Index (Filtered)")
    plt.savefig(os.path.join(output_dir, f"{site}_{sae_name}_implied_correlation.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze Banded SAE Parameters")
    parser.add_argument("--sae_dir", type=str, default="outputs/saes/multipartite_003_bsae", help="Directory containing SAE checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis/banded_params_corr", help="Output directory for plots")
    parser.add_argument("--layer", type=str, default="layer_1", help="Layer to analyze (e.g., layer_1)")
    parser.add_argument("--activation_threshold", type=float, default=0.0, help="Activation rate threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    args = parser.parse_args()
    
    metrics_path = os.path.join(args.sae_dir, "metrics_summary.json")
    print(f"Loading metrics from {metrics_path}...")
    metrics_summary = load_metrics_summary(metrics_path)
    
    # Find all files for the specified layer
    pattern = os.path.join(args.sae_dir, f"{args.layer}_banded_*.pt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching {pattern}")
        return
        
    print(f"Found {len(files)} SAEs for {args.layer}")
    
    for sae_path in files:
        filename = os.path.basename(sae_path)
        # Parse sae_name from filename: layer_1_banded_{sae_name}.pt
        # format: layer_1_banded_ls_0p005__la_0p005.pt
        # prefix: layer_1_banded_
        prefix = f"{args.layer}_banded_"
        if not filename.startswith(prefix):
            continue
            
        sae_name = filename[len(prefix):-3] # remove prefix and .pt
        
        print(f"Analyzing {sae_name}...")
        
        sae = load_banded_sae(sae_path, args.device)
        params = extract_parameters(sae)
        coupling_coeffs = compute_coupling_matrix(params, sae.cfg["dict_size"])
        
        # Get active rates
        # The metrics_summary keys might match sae_name
        active_rates = get_active_rates(metrics_summary, args.layer, sae_name)
        
        plot_analysis(params, coupling_coeffs, active_rates, args.output_dir, args.layer, sae_name, args.activation_threshold)
        
    print("Done!")

if __name__ == "__main__":
    main()
