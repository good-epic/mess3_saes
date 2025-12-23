# %% Imports
import pandas as pd
import numpy as np
import requests
import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import time

# %% Configuration
# Neuronpedia API configuration
API_BASE = "https://www.neuronpedia.org/api"

# Model configuration
MODEL_ID = "gemma-2-9b"  # Base model name for Neuronpedia
LAYER_NUM = 20

# SAE configuration - we need to determine the correct Neuronpedia SAE ID
# Options to try (Neuronpedia naming convention):
# - "20-gemmascope-res-32k" (32k width)
# - "20-gemmascope-res-131k" (131k width - confirmed available)
# - "20-gemmascope-res-65k" (65k width)
# - "20-gemmascope-res-16k" (16k width)

# Start with what we know exists
NEURONPEDIA_SAE_ID = "20-gemmascope-res-131k"  # Confirmed available on Neuronpedia

# Try to find 32k version
ALTERNATIVE_SAE_IDS = [
    "20-gemmascope-res-32k",
    "20-gemmascope-res-65k",
    "20-gemmascope-res-16k"
]

# Data paths
DATA_DIR = Path("/home/mattylev/projects/simplex/SAEs/mess3_sae/outputs/real_data_analysis")
OUTPUT_DIR = Path("/home/mattylev/projects/simplex/SAEs/mess3_sae/scratch/neuronpedia_summaries")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% Helper Functions

def test_neuronpedia_connection(model_id: str, sae_id: str, test_index: int = 0) -> bool:
    """
    Test if we can connect to Neuronpedia API for a specific model/SAE combination.
    """
    url = f"{API_BASE}/feature/{model_id}/{sae_id}/{test_index}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✓ Successfully connected to {model_id}/{sae_id}")
            return True
        else:
            print(f"✗ Failed to connect to {model_id}/{sae_id}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error connecting to {model_id}/{sae_id}: {e}")
        return False

def get_feature_info(model_id: str, sae_id: str, feature_index: int, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch feature information from Neuronpedia API.

    Returns dict with:
    - explanations: List of explanations for this feature
    - activations: Activation examples
    - feature_url: URL to view on Neuronpedia
    """
    url = f"{API_BASE}/feature/{model_id}/{sae_id}/{feature_index}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return {
                    "feature_index": feature_index,
                    "model_id": model_id,
                    "sae_id": sae_id,
                    "feature_url": f"https://www.neuronpedia.org/{model_id}/{sae_id}/{feature_index}",
                    "data": data
                }
            elif response.status_code == 404:
                print(f"  Feature {feature_index} not found (404)")
                return None
            else:
                print(f"  Attempt {attempt+1}/{max_retries}: Status {response.status_code}")
                time.sleep(1)
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries}: Error - {e}")
            time.sleep(1)

    print(f"  Failed to fetch feature {feature_index} after {max_retries} attempts")
    return None

def extract_summary_from_feature(feature_info: Dict) -> Dict:
    """
    Extract key summary information from Neuronpedia feature response.
    """
    if not feature_info or "data" not in feature_info:
        return {}

    data = feature_info["data"]

    summary = {
        "feature_index": feature_info["feature_index"],
        "feature_url": feature_info["feature_url"],
        "explanations": [],
        "num_activations": 0,
        "top_activating_text": []
    }

    # Extract explanations
    if "explanations" in data and data["explanations"]:
        for exp in data["explanations"]:
            summary["explanations"].append({
                "text": exp.get("description", ""),
                "score": exp.get("score", None)
            })

    # Extract activation examples
    if "activations" in data and data["activations"]:
        summary["num_activations"] = len(data["activations"])
        # Get top few activation examples
        for act in data["activations"][:5]:  # Top 5 examples
            summary["top_activating_text"].append({
                "text": act.get("tokens", ""),
                "activation_value": act.get("values", [])
            })

    return summary

# %% Load Elbow Data

print("Loading elbow data from all n_clusters...")
all_elbow_df = pd.DataFrame()

for clusters_dir in sorted(DATA_DIR.glob("clusters_*")):
    n_str = clusters_dir.name.split("_")[1]
    n_clusters = int(n_str)

    csv_path = clusters_dir / f"consolidated_metrics_n{n_str}.csv"
    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)
    df['n_clusters_total'] = n_clusters
    all_elbow_df = pd.concat([all_elbow_df, df], ignore_index=True)

print(f"Loaded {len(all_elbow_df)} rows from {len(all_elbow_df['n_clusters_total'].unique())} different n_clusters values")

# %% Calculate Elbow Metrics

def calculate_elbow_score(x, y):
    """Calculate elbow using perpendicular distance from line."""
    if len(x) < 3:
        return None, 0.0

    x_norm = (np.array(x) - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
    y_norm = (np.array(y) - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else np.zeros_like(y)

    distances = np.abs(x_norm + y_norm - 1) / np.sqrt(2)

    elbow_idx = np.argmax(distances)
    elbow_k = x[elbow_idx]
    elbow_strength = distances[elbow_idx]

    return elbow_k, elbow_strength

# Group by cluster and n_clusters to calculate elbows
print("\nCalculating elbow metrics...")
elbow_results = []

for (n_clust, cluster_id), group in all_elbow_df.groupby(['n_clusters_total', 'cluster_id']):
    group = group.sort_values('aanet_k')

    if len(group) < 3:
        continue

    k_values = group['aanet_k'].tolist()

    # Calculate elbows for each loss type
    recon_elbow_k, recon_strength = calculate_elbow_score(k_values, group['aanet_recon_loss'].tolist())
    arch_elbow_k, arch_strength = calculate_elbow_score(k_values, group['aanet_archetypal_loss'].tolist())
    extrema_elbow_k, extrema_strength = calculate_elbow_score(k_values, group['aanet_extrema_loss'].tolist())

    elbow_results.append({
        'n_clusters_total': n_clust,
        'cluster_id': cluster_id,
        'n_latents': group['n_latents'].iloc[0],
        'latent_indices': group['latent_indices'].iloc[0],
        'aanet_recon_loss_elbow_k': recon_elbow_k,
        'aanet_recon_loss_elbow_strength': recon_strength,
        'aanet_archetypal_loss_elbow_k': arch_elbow_k,
        'aanet_archetypal_loss_elbow_strength': arch_strength,
        'aanet_extrema_loss_elbow_k': extrema_elbow_k,
        'aanet_extrema_loss_elbow_strength': extrema_strength,
    })

elbow_df = pd.DataFrame(elbow_results)
print(f"Calculated elbows for {len(elbow_df)} clusters")

# %% Test Neuronpedia Connection

print("\n" + "="*80)
print("Testing Neuronpedia API connection...")
print("="*80)

# Test with the known-good SAE
if test_neuronpedia_connection(MODEL_ID, NEURONPEDIA_SAE_ID, test_index=100):
    SELECTED_SAE_ID = NEURONPEDIA_SAE_ID
    print(f"\nUsing SAE: {SELECTED_SAE_ID}")
else:
    # Try alternatives
    print("\nTrying alternative SAE IDs...")
    SELECTED_SAE_ID = None
    for alt_sae_id in ALTERNATIVE_SAE_IDS:
        if test_neuronpedia_connection(MODEL_ID, alt_sae_id, test_index=100):
            SELECTED_SAE_ID = alt_sae_id
            print(f"\nUsing alternative SAE: {SELECTED_SAE_ID}")
            break

    if SELECTED_SAE_ID is None:
        print("\n⚠ WARNING: Could not connect to Neuronpedia API with any known SAE ID!")
        print("You may need to:")
        print("1. Check if your SAE is available on Neuronpedia")
        print("2. Manually specify the correct SAE ID in NEURONPEDIA_SAE_ID")
        print("3. Upload your SAE to Neuronpedia if it's not available yet")

# %% Select Top Clusters

print("\n" + "="*80)
print("Selecting most interesting clusters...")
print("="*80)

# Selection criteria: High elbow strength for reconstruction or archetypal loss
# Focus on reconstruction loss as discussed
top_n_per_group = 5  # Top N clusters per n_clusters value

selected_clusters = []

for n_clust in sorted(elbow_df['n_clusters_total'].unique()):
    group = elbow_df[elbow_df['n_clusters_total'] == n_clust].copy()

    # Sort by reconstruction loss elbow strength (primary criterion)
    group = group.sort_values('aanet_recon_loss_elbow_strength', ascending=False)

    # Take top N
    top_clusters = group.head(top_n_per_group)

    print(f"\nn_clusters={n_clust}: Selected {len(top_clusters)} clusters")
    for idx, row in top_clusters.iterrows():
        print(f"  Cluster {row['cluster_id']:3d}: "
              f"n_latents={row['n_latents']:4d}, "
              f"recon_elbow_strength={row['aanet_recon_loss_elbow_strength']:.4f}, "
              f"arch_elbow_strength={row['aanet_archetypal_loss_elbow_strength']:.4f}")
        selected_clusters.append(row)

selected_df = pd.DataFrame(selected_clusters)
print(f"\nTotal selected clusters: {len(selected_df)}")

# %% Fetch Neuronpedia Summaries

if SELECTED_SAE_ID is None:
    print("\n⚠ Skipping Neuronpedia fetch - no valid SAE connection")
else:
    print("\n" + "="*80)
    print("Fetching Neuronpedia summaries for selected clusters...")
    print("="*80)

    # Note: User's SAE has 32k features (indices 0-32767)
    # Neuronpedia might have 131k features
    # We'll try to fetch, but indices might not match if widths differ

    all_summaries = []

    for idx, cluster_row in selected_df.iterrows():
        n_clust = cluster_row['n_clusters_total']
        cluster_id = cluster_row['cluster_id']

        print(f"\nFetching cluster {cluster_id} from n_clusters={n_clust}...")

        # Parse latent indices
        latent_indices_str = cluster_row['latent_indices']
        try:
            latent_indices = eval(latent_indices_str)  # Convert string repr of list to list
        except:
            print(f"  Error parsing latent indices: {latent_indices_str[:100]}")
            continue

        print(f"  {len(latent_indices)} latents in cluster")

        cluster_summaries = {
            "n_clusters_total": n_clust,
            "cluster_id": cluster_id,
            "n_latents": len(latent_indices),
            "elbow_metrics": {
                "recon_strength": cluster_row['aanet_recon_loss_elbow_strength'],
                "arch_strength": cluster_row['aanet_archetypal_loss_elbow_strength'],
                "recon_k": cluster_row['aanet_recon_loss_elbow_k'],
                "arch_k": cluster_row['aanet_archetypal_loss_elbow_k'],
            },
            "features": []
        }

        # Fetch first 10 features from this cluster (to avoid overwhelming API)
        for feature_idx in latent_indices[:10]:
            print(f"  Fetching feature {feature_idx}...", end="")
            feature_info = get_feature_info(MODEL_ID, SELECTED_SAE_ID, feature_idx)

            if feature_info:
                summary = extract_summary_from_feature(feature_info)
                cluster_summaries["features"].append(summary)
                print(" ✓")
            else:
                print(" ✗")

            # Rate limiting
            time.sleep(0.5)

        all_summaries.append(cluster_summaries)

        # Save incrementally
        output_file = OUTPUT_DIR / f"cluster_summaries_n{n_clust}_c{cluster_id}.json"
        with open(output_file, 'w') as f:
            json.dump(cluster_summaries, f, indent=2)
        print(f"  Saved to {output_file}")

    # Save combined results
    combined_file = OUTPUT_DIR / "all_cluster_summaries.json"
    with open(combined_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✓ Saved all summaries to {combined_file}")

    # Create human-readable report
    report_file = OUTPUT_DIR / "summary_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NEURONPEDIA CLUSTER SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        for cluster_sum in all_summaries:
            n_clust = cluster_sum['n_clusters_total']
            cluster_id = cluster_sum['cluster_id']

            f.write(f"\n{'='*80}\n")
            f.write(f"Cluster {cluster_id} (n_clusters={n_clust})\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total latents: {cluster_sum['n_latents']}\n")
            f.write(f"Reconstruction elbow strength: {cluster_sum['elbow_metrics']['recon_strength']:.4f}\n")
            f.write(f"Archetypal elbow strength: {cluster_sum['elbow_metrics']['arch_strength']:.4f}\n")
            f.write(f"Reconstruction elbow k: {cluster_sum['elbow_metrics']['recon_k']}\n")
            f.write(f"Archetypal elbow k: {cluster_sum['elbow_metrics']['arch_k']}\n")
            f.write(f"\n")

            f.write(f"Features (showing first {len(cluster_sum['features'])}):\n")
            f.write(f"{'-'*80}\n")

            for feat in cluster_sum['features']:
                f.write(f"\nFeature {feat['feature_index']}:\n")
                f.write(f"  URL: {feat['feature_url']}\n")

                if feat.get('explanations'):
                    f.write(f"  Explanations:\n")
                    for exp in feat['explanations']:
                        score = f" (score: {exp['score']:.3f})" if exp['score'] is not None else ""
                        f.write(f"    - {exp['text']}{score}\n")
                else:
                    f.write(f"  No explanations available\n")

                if feat.get('top_activating_text'):
                    f.write(f"  Top activation examples:\n")
                    for i, act in enumerate(feat['top_activating_text'][:3], 1):
                        text_preview = str(act['text'])[:100]
                        f.write(f"    {i}. {text_preview}...\n")

                f.write(f"\n")

    print(f"✓ Saved human-readable report to {report_file}")

print("\n" + "="*80)
print("Done!")
print("="*80)
