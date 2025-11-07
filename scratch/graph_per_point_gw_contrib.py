#%%

import json
import pandas as pd
from pathlib import Path

# Base directory for the run
run_dir = "outputs/reports/multipartite_003e/kvar_20251010-155918"
sites = ["embeddings", "layer_0", "layer_1", "layer_2"]

# Collect all data
rows = []

for site in sites:
    cluster_summary_path = Path(run_dir) / site / "cluster_summary.json"

    if not cluster_summary_path.exists():
        print(f"Skipping {site} - file not found")
        continue

    data = json.load(open(cluster_summary_path))

    # Check if geometry refinement exists
    if 'geometry_refinement' not in data or 'clusters' not in data['geometry_refinement']:
        print(f"Skipping {site} - no geometry refinement")
        continue

    clustering_method = data.get('clustering_method', 'unknown')

    # Loop through clusters
    for cluster_id, cluster_data in data['geometry_refinement']['clusters'].items():
        n_points = cluster_data.get('n_initial_members', 0)

        # Loop through geometries
        for geometry_name, geometry_fit in cluster_data['all_geometry_fits'].items():
            optimal_distance = geometry_fit['optimal_distance']

            # Get per-point contributions
            gw_contributions = geometry_fit.get('per_point_raw_values', {}).get('gw_distortion_contributions', [])

            rows.append({
                'site': site,
                'clustering_method': clustering_method,
                'cluster_id': int(cluster_id),
                'geometry_name': geometry_name,
                'n_points': n_points,
                'optimal_distance': optimal_distance,
                'gw_distortion_contributions': gw_contributions
            })

# Create dataframe
df = pd.DataFrame(rows)

print(f"Created dataframe with {len(df)} rows")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nSites: {df['site'].unique()}")
print(f"Geometries: {df['geometry_name'].unique()}")
print(f"Clusters per site: {df.groupby('site')['cluster_id'].nunique()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataframe shape: {df.shape}")

#%%
import matplotlib.pyplot as plt
import numpy as np

# Get unique geometries and sort them
geometries = sorted(df['geometry_name'].unique())
print(f"\nGeometries: {geometries}")

# Create 3x3 grid for all sites combined (histograms of optimal distance)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for idx, geom in enumerate(geometries):
    ax = axes[idx]
    data = df[df['geometry_name'] == geom]['optimal_distance']

    ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
    ax.set_title(f"{geom} (n={len(data)})")
    ax.set_xlabel("Optimal Distance")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

# Hide unused subplots if any
for idx in range(len(geometries), 9):
    axes[idx].set_visible(False)

plt.suptitle("Optimal Distance by Geometry (All Sites)", fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(f"{run_dir}/optimal_distance_by_geometry_all_sites.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Scatter plot: Number of latents (cluster size) vs. optimal distance, combined over all sites ---

plt.figure(figsize=(9, 6))
for geom in geometries:
    mask = df['geometry_name'] == geom
    x = df[mask]['n_points']
    y = df[mask]['optimal_distance']
    plt.scatter(x, y, label=geom, alpha=0.7)
plt.xlabel("Number of Latents in Cluster")
plt.ylabel("Optimal Distance")
plt.title("Cluster Size vs. Optimal Distance (Combined Over Sites)")
plt.legend(fontsize=9, loc="best", frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{run_dir}/n_points_vs_optimal_distance_scatter_all_sites.png", dpi=150, bbox_inches='tight')
plt.show()


#%%
# Create 3x3 grid for each site separately
for site in sites:
    site_df = df[df['site'] == site]

    if len(site_df) == 0:
        print(f"No data for {site}")
        continue

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, geom in enumerate(geometries):
        ax = axes[idx]
        data = site_df[site_df['geometry_name'] == geom]['optimal_distance']

        ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
        ax.set_title(f"{geom} (n={len(data)})")
        ax.set_xlabel("Optimal Distance")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if any
    for idx in range(len(geometries), 9):
        axes[idx].set_visible(False)

    plt.suptitle(f"Optimal Distance by Geometry ({site})", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/optimal_distance_by_geometry_{site}.png", dpi=150, bbox_inches='tight')
    plt.show()

print(f"\nSaved plots to {run_dir}/")

#%%
# Create 3x3 grid for per-point contributions (all sites combined)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for idx, geom in enumerate(geometries):
    ax = axes[idx]
    geom_data = df[df['geometry_name'] == geom]

    # Flatten all per-point contributions for this geometry
    all_contributions = []
    for contribs in geom_data['gw_distortion_contributions']:
        all_contributions.extend(contribs)

    if len(all_contributions) > 0:
        ax.hist(all_contributions, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f"{geom} (n={len(all_contributions)} points)")
        ax.set_xlabel("Per-Point GW Contribution")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

# Hide unused subplots if any
for idx in range(len(geometries), 9):
    axes[idx].set_visible(False)

plt.suptitle("Per-Point GW Contributions by Geometry (All Sites)", fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(f"{run_dir}/per_point_gw_contrib_by_geometry_all_sites.png", dpi=150, bbox_inches='tight')
plt.show()

#%%
# Create 3x3 grid for per-point contributions for each site separately
for site in sites:
    site_df = df[df['site'] == site]

    if len(site_df) == 0:
        print(f"No data for {site}")
        continue

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, geom in enumerate(geometries):
        ax = axes[idx]
        geom_data = site_df[site_df['geometry_name'] == geom]

        # Flatten all per-point contributions for this geometry
        all_contributions = []
        for contribs in geom_data['gw_distortion_contributions']:
            all_contributions.extend(contribs)

        if len(all_contributions) > 0:
            ax.hist(all_contributions, bins=50, edgecolor='black', alpha=0.7)
            ax.set_title(f"{geom} (n={len(all_contributions)} points)")
            ax.set_xlabel("Per-Point GW Contribution")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

    # Hide unused subplots if any
    for idx in range(len(geometries), 9):
        axes[idx].set_visible(False)

    plt.suptitle(f"Per-Point GW Contributions by Geometry ({site})", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/per_point_gw_contrib_by_geometry_{site}.png", dpi=150, bbox_inches='tight')
    plt.show()

print(f"\nSaved per-point contribution plots to {run_dir}/")

#%%
# Create a single histogram for all optimal distances (all sites, all geometries)
fig, ax = plt.subplots(figsize=(10, 6))

all_optimal_distances = df['optimal_distance'].values
ax.hist(all_optimal_distances, bins=50, edgecolor='black', alpha=0.7)
ax.set_title(f"Optimal Distance Distribution (All Sites, All Geometries)\nn={len(all_optimal_distances)}", fontsize=14)
ax.set_xlabel("Optimal Distance", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{run_dir}/optimal_distance_all_combined.png", dpi=150, bbox_inches='tight')
plt.show()

#%%
# Create a single histogram for all per-point contributions (all sites, all geometries)
fig, ax = plt.subplots(figsize=(10, 6))

all_contributions = []
for contribs in df['gw_distortion_contributions']:
    all_contributions.extend(contribs)

ax.hist(all_contributions, bins=50, edgecolor='black', alpha=0.7)
ax.set_title(f"Per-Point GW Contribution Distribution (All Sites, All Geometries)\nn={len(all_contributions)} points", fontsize=14)
ax.set_xlabel("Per-Point GW Contribution", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{run_dir}/per_point_gw_contrib_all_combined.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved combined histograms to {run_dir}/")

#%%
# Create percentile curves (CDF) for per-point contributions (all sites combined)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for idx, geom in enumerate(geometries):
    ax = axes[idx]
    geom_data = df[df['geometry_name'] == geom]

    # Flatten all per-point contributions for this geometry
    all_contributions = []
    for contribs in geom_data['gw_distortion_contributions']:
        all_contributions.extend(contribs)

    if len(all_contributions) > 0:
        all_contributions = np.array(all_contributions)
        sorted_vals = np.sort(all_contributions)
        percentiles = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100

        ax.plot(sorted_vals, percentiles, linewidth=2)
        ax.axvline(x=0.001, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(x=0.0005, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(x=0.0001, color='red', linestyle='--', linewidth=1.5)
        ax.set_title(f"{geom} (n={len(all_contributions)} points)")
        ax.set_xlabel("Per-Point GW Contribution")
        ax.set_ylabel("Percentile")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 0.001)

# Hide unused subplots if any
for idx in range(len(geometries), 9):
    axes[idx].set_visible(False)

plt.suptitle("Per-Point GW Contribution Percentiles by Geometry (All Sites)", fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(f"{run_dir}/per_point_gw_contrib_percentiles_all_sites.png", dpi=150, bbox_inches='tight')
plt.show()

#%%
# Create percentile curves for each site separately
for site in sites:
    site_df = df[df['site'] == site]

    if len(site_df) == 0:
        print(f"No data for {site}")
        continue

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, geom in enumerate(geometries):
        ax = axes[idx]
        geom_data = site_df[site_df['geometry_name'] == geom]

        # Flatten all per-point contributions for this geometry
        all_contributions = []
        for contribs in geom_data['gw_distortion_contributions']:
            all_contributions.extend(contribs)

        if len(all_contributions) > 0:
            all_contributions = np.array(all_contributions)
            sorted_vals = np.sort(all_contributions)
            percentiles = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100

            ax.plot(sorted_vals, percentiles, linewidth=2)
            ax.axvline(x=0.001, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(x=0.0005, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(x=0.0001, color='red', linestyle='--', linewidth=1.5)
            ax.set_title(f"{geom} (n={len(all_contributions)} points)")
            ax.set_xlabel("Per-Point GW Contribution")
            ax.set_ylabel("Percentile")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 0.001)

    # Hide unused subplots if any
    for idx in range(len(geometries), 9):
        axes[idx].set_visible(False)

    plt.suptitle(f"Per-Point GW Contribution Percentiles by Geometry ({site})", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/per_point_gw_contrib_percentiles_{site}.png", dpi=150, bbox_inches='tight')
    plt.show()

print(f"\nSaved percentile curve plots to {run_dir}/")