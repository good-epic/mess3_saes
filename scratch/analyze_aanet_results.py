# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cluster_selection import calculate_elbow_score, select_promising_clusters

# %% Selection parameters
# These control the SD thresholds for cluster selection
SD_STRONG = 0.5      # For categories A (strong both) and D (perfect agreement): mean + SD_STRONG * std
SD_OUTLIER = 2     # For categories B (recon outliers) and C (arch outliers): mean + SD_OUTLIER * std
DELTA_K_THRESHOLD = 2  # Maximum allowed |k_differential| for quality filter

# %% Setup - Load data
# Configure paths
base_dir = Path(__file__).parent.parent / "outputs" / "real_data_analysis_canonical"
n_clusters_list = [512]

# NOTE: As of the latest updates, analyze_real_saes.py now outputs the full CSV
# with all quality metrics (elbow metrics, monotonicity, pct_decrease, etc.)
# directly in consolidated_metrics_n{n}.csv - no separate "corrected" version needed.
# This script is kept for ad-hoc exploration and manual cluster selection.

# Load all consolidated metrics
all_metrics = {}
for n in n_clusters_list:
    # Try standard CSV first (new behavior), fall back to _corrected (old behavior)
    csv_path = base_dir / f"clusters_{n}" / f"consolidated_metrics_n{n}.csv"
    if not csv_path.exists():
        csv_path = base_dir / f"clusters_{n}" / f"consolidated_metrics_n{n}_corrected.csv"

    if csv_path.exists():
        all_metrics[n] = pd.read_csv(csv_path)
        print(f"Loaded n={n}: {len(all_metrics[n])} rows from {csv_path.name}")
    else:
        print(f"Missing: {csv_path}")

# %% Setup - Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

# Choose which n_clusters to analyze (can change this)
primary_n_clusters = 768
df = all_metrics[n]

print(f"\nAnalyzing n_clusters={primary_n_clusters}")
print(f"Shape: {df.shape}")
print(f"Unique clusters: {df['cluster_id'].nunique()}")
print(f"Unique k values: {sorted(df['aanet_k'].unique())}")

# %% Analysis 1: Decoder direction rank distribution
# Since clustering doesn't change with k, decoder_dir_rank is the same for all k
# Only plot one point per cluster (take k=2 arbitrarily)
df_k2 = df[df['aanet_k'] == 2].copy()

# Check what data is available
print("Data availability check:")
print(f"  decoder_dir_rank non-null: {df_k2['decoder_dir_rank'].notna().sum()} / {len(df_k2)}")
print(f"  activation_pca_rank non-null: {df_k2['activation_pca_rank'].notna().sum()} / {len(df_k2)}")

# Plot decoder rank distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(df_k2['decoder_dir_rank'], bins=30, alpha=0.7, edgecolor='black')
ax.set_xlabel('Decoder Direction Rank')
ax.set_ylabel('Number of Clusters')
ax.set_title(f'Decoder Direction Rank Distribution (n_clusters={n})')
ax.grid(True, alpha=0.3)

# Scatter: decoder rank vs number of latents in cluster
ax = axes[1]
scatter = ax.scatter(
    df_k2['n_latents'],
    df_k2['decoder_dir_rank'],
    alpha=0.6,
    s=30,
    c=df_k2['n_latents'],
    cmap='viridis'
)
ax.set_xlabel('Number of Latents in Cluster')
ax.set_ylabel('Decoder Direction Rank')
ax.set_title('Decoder Rank vs Cluster Size')
plt.colorbar(scatter, ax=ax, label='Cluster Size')

plt.tight_layout()
plt.show()

print(f"\nDecoder rank stats:")
print(df_k2['decoder_dir_rank'].describe())

# %% Analysis 1b: Geometric Rank vs PCA Rank
# Compare the decoder direction rank (geometric) with the activation PCA rank
valid_rank_data = df_k2.dropna(subset=['decoder_dir_rank', 'activation_pca_rank'])

if len(valid_rank_data) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter = ax.scatter(
        valid_rank_data['decoder_dir_rank'],
        valid_rank_data['activation_pca_rank'],
        c=valid_rank_data['n_latents'],
        cmap='viridis',
        alpha=0.6,
        s=40
    )

    # Add diagonal line (y=x)
    max_rank = max(valid_rank_data['decoder_dir_rank'].max(), valid_rank_data['activation_pca_rank'].max())
    ax.plot([0, max_rank], [0, max_rank], 'r--', alpha=0.5, label='y=x')

    ax.set_xlabel('Geometric Rank (Decoder Directions)', fontsize=12)
    ax.set_ylabel('PCA Rank (Activations)', fontsize=12)
    ax.set_title(f'Geometric Rank vs PCA Rank (n_clusters={primary_n_clusters})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Cluster Size (n_latents)')

    plt.tight_layout()
    plt.show()

    # Print correlation
    corr = valid_rank_data['decoder_dir_rank'].corr(valid_rank_data['activation_pca_rank'])
    print(f"\nCorrelation between geometric rank and PCA rank: {corr:.3f}")
    print(f"Clusters plotted: {len(valid_rank_data)}")
    print(f"\nPCA rank stats:")
    print(valid_rank_data['activation_pca_rank'].describe())
else:
    print("No valid data for geometric rank vs PCA rank plot (both columns have NaN)")

# %% Analysis 2: AAnet losses vs n_latents (colored by k)
loss_types = ['aanet_loss', 'aanet_recon_loss', 'aanet_archetypal_loss', 'aanet_extrema_loss']
k_values = sorted(df['aanet_k'].unique())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, loss_type in enumerate(loss_types):
    ax = axes[idx]

    for k in k_values:
        df_k = df[df['aanet_k'] == k]
        ax.scatter(
            df_k['n_latents'],
            df_k[loss_type],
            alpha=0.5,
            s=20,
            label=f'k={k}'
        )

    ax.set_xlabel('Number of Latents in Cluster')
    ax.set_ylabel(loss_type.replace('aanet_', '').replace('_', ' ').title())
    ax.set_title(f'{loss_type.replace("aanet_", "").replace("_", " ").title()} vs Cluster Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_yscale('log')

plt.tight_layout()
plt.show()

# %% Analysis 3: Boxplots of losses vs k
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, loss_type in enumerate(loss_types):
    ax = axes[idx]

    # Prepare data for boxplot
    data_by_k = [df[df['aanet_k'] == k][loss_type].values for k in k_values]

    bp = ax.boxplot(
        data_by_k,
        labels=[str(k) for k in k_values],
        patch_artist=True,
        showfliers=False  # Hide outliers for cleaner view
    )

    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('k (Number of Archetypes)')
    ax.set_ylabel(loss_type.replace('aanet_', '').replace('_', ' ').title())
    ax.set_title(f'{loss_type.replace("aanet_", "").replace("_", " ").title()} Distribution by k')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print(f"Summary Statistics by k (n_clusters={n})")
print("="*60)
for k in k_values:
    df_k = df[df['aanet_k'] == k]
    print(f"\nk={k}:")
    for loss_type in loss_types:
        mean_val = df_k[loss_type].mean()
        std_val = df_k[loss_type].std()
        print(f"  {loss_type:25s}: {mean_val:.6f} ± {std_val:.6f}")

# %% Analysis 4: K-elbow detection for each cluster
# (calculate_elbow_score imported from cluster_selection module)

# Calculate elbow for each cluster and each loss type
elbow_results = []

for cluster_id in df['cluster_id'].unique():
    df_cluster = df[df['cluster_id'] == cluster_id].sort_values('aanet_k')

    row = {'cluster_id': cluster_id, 'n_latents': df_cluster.iloc[0]['n_latents']}

    for loss_type in loss_types:
        k_vals = df_cluster['aanet_k'].values
        loss_vals = df_cluster[loss_type].values

        elbow_k, elbow_strength = calculate_elbow_score(k_vals, loss_vals)

        row[f'{loss_type}_elbow_k'] = elbow_k
        row[f'{loss_type}_elbow_strength'] = elbow_strength

    elbow_results.append(row)

elbow_df = pd.DataFrame(elbow_results)

print("\n" + "="*60)
print(f"Elbow Detection Results (n_clusters={n})")
print("="*60)
print(f"\nTop 10 clusters by total loss elbow strength:")
top_elbows = elbow_df.nlargest(10, 'aanet_loss_elbow_strength')
print(top_elbows[['cluster_id', 'n_latents', 'aanet_loss_elbow_k', 'aanet_loss_elbow_strength']].to_string(index=False))

# Plot elbow strength distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, loss_type in enumerate(loss_types):
    ax = axes[idx]

    # Histogram of elbow strengths
    ax.hist(elbow_df[f'{loss_type}_elbow_strength'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Elbow Strength')
    ax.set_ylabel('Number of Clusters')
    ax.set_title(f'{loss_type.replace("aanet_", "").replace("_", " ").title()} Elbow Strength Distribution')
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_strength = elbow_df[f'{loss_type}_elbow_strength'].mean()
    ax.axvline(mean_strength, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_strength:.3f}')
    ax.legend()

plt.tight_layout()
plt.show()

# %% Analysis 5: Elbow k value distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, loss_type in enumerate(loss_types):
    ax = axes[idx]

    # Count how many clusters have elbow at each k
    elbow_k_col = f'{loss_type}_elbow_k'
    k_counts = elbow_df[elbow_k_col].value_counts().sort_index()

    ax.bar(k_counts.index, k_counts.values, alpha=0.7, edgecolor='black')
    ax.set_xlabel('k value at elbow')
    ax.set_ylabel('Number of Clusters')
    ax.set_title(f'{loss_type.replace("aanet_", "").replace("_", " ").title()} - Distribution of Elbow K')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print summary
print("\n" + "="*60)
print("Elbow K Distribution Summary")
print("="*60)
for loss_type in loss_types:
    elbow_k_col = f'{loss_type}_elbow_k'
    print(f"\n{loss_type}:")
    k_counts = elbow_df[elbow_k_col].value_counts().sort_index()
    for k, count in k_counts.items():
        pct = 100 * count / len(elbow_df)
        print(f"  k={int(k)}: {count} clusters ({pct:.1f}%)")

# %% Analysis 6: Elbow statistics across all n_clusters
# Calculate elbows for all n_clusters values
all_elbow_results = []

for n_clust in all_metrics.keys():
    df_n = all_metrics[n_clust]

    for cluster_id in df_n['cluster_id'].unique():
        df_cluster = df_n[df_n['cluster_id'] == cluster_id].sort_values('aanet_k')

        row = {
            'n_clusters_total': n_clust,
            'cluster_id': cluster_id,
            'n_latents': df_cluster.iloc[0]['n_latents']
        }

        # Add latent_indices if available in the data
        if 'latent_indices' in df_cluster.columns:
            row['latent_indices'] = df_cluster.iloc[0]['latent_indices']

        for loss_type in loss_types:
            k_vals = df_cluster['aanet_k'].values
            loss_vals = df_cluster[loss_type].values

            elbow_k, elbow_strength = calculate_elbow_score(k_vals, loss_vals)

            row[f'{loss_type}_elbow_k'] = elbow_k
            row[f'{loss_type}_elbow_strength'] = elbow_strength

        # Add quality metrics from corrected CSV (same for all rows of this cluster)
        if 'recon_is_monotonic' in df_cluster.columns:
            row['recon_is_monotonic'] = df_cluster.iloc[0]['recon_is_monotonic']
        if 'arch_is_monotonic' in df_cluster.columns:
            row['arch_is_monotonic'] = df_cluster.iloc[0]['arch_is_monotonic']
        if 'recon_pct_decrease' in df_cluster.columns:
            row['recon_pct_decrease'] = df_cluster.iloc[0]['recon_pct_decrease']
        if 'arch_pct_decrease' in df_cluster.columns:
            row['arch_pct_decrease'] = df_cluster.iloc[0]['arch_pct_decrease']
        if 'k_differential' in df_cluster.columns:
            row['k_differential'] = df_cluster.iloc[0]['k_differential']

        all_elbow_results.append(row)

all_elbow_df = pd.DataFrame(all_elbow_results)

# Plot elbow strength distributions by n_clusters
for loss_type in loss_types:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    strength_col = f'{loss_type}_elbow_strength'

    for idx, n_clust in enumerate(sorted(all_metrics.keys())):
        if idx >= 4:
            break
        ax = axes[idx]

        data = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust][strength_col]

        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', color=f'C{idx}')
        ax.set_xlabel('Elbow Strength')
        ax.set_ylabel('Number of Clusters')
        ax.set_title(f'n_clusters={n_clust}')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_val:.3f}')
        ax.legend()

    fig.suptitle(f'{loss_type.replace("aanet_", "").replace("_", " ").title()} Elbow Strength by Total Clusters',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()

# %% Analysis 7: Elbow strength by k value (where the elbow occurs)
for loss_type in loss_types:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    strength_col = f'{loss_type}_elbow_strength'
    elbow_k_col = f'{loss_type}_elbow_k'

    for idx, n_clust in enumerate(sorted(all_metrics.keys())):
        if idx >= 4:
            break
        ax = axes[idx]

        data = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust]

        # Create box plot for elbow strength at each k value
        k_values_present = sorted(data[elbow_k_col].dropna().unique())
        data_by_k = [data[data[elbow_k_col] == k][strength_col].values for k in k_values_present]

        bp = ax.boxplot(data_by_k, tick_labels=[str(int(k)) for k in k_values_present],
                        patch_artist=True, showfliers=True)

        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values_present)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('k value at elbow')
        ax.set_ylabel('Elbow Strength')
        ax.set_title(f'n_clusters={n_clust}')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{loss_type.replace("aanet_", "").replace("_", " ").title()} Elbow Strength vs K Value',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()

# Print summary across all n_clusters
print("\n" + "="*80)
print("Elbow Statistics Across All n_clusters")
print("="*80)
for loss_type in loss_types:
    print(f"\n{loss_type}:")
    print(f"  {'n_clusters':>12} {'Mean Strength':>15} {'Median Strength':>17} {'Std Strength':>15}")
    print("  " + "-"*60)
    for n_clust in sorted(all_metrics.keys()):
        data = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust][f'{loss_type}_elbow_strength']
        print(f"  {n_clust:>12} {data.mean():>15.4f} {data.median():>17.4f} {data.std():>15.4f}")

# %% Analysis 8: Reconstruction vs Archetypal loss, colored by elbow k differential
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, n_clust in enumerate(sorted(all_metrics.keys())):
    if idx >= 4:
        break
    ax = axes[idx]

    # Get elbow data for this n_clusters
    plot_data = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust].copy()

    # Calculate k differential (recon elbow k - archetypal elbow k)
    plot_data['k_differential'] = (plot_data['aanet_recon_loss_elbow_k'] -
                                    plot_data['aanet_archetypal_loss_elbow_k'])

    # Sort so differential == 0 is plotted last (on top)
    plot_data['plot_order'] = (plot_data['k_differential'] == 0).astype(int)
    plot_data = plot_data.sort_values('plot_order')

    # Define distinct colors for each k differential value
    # Get unique differentials and assign colors
    unique_diffs = sorted(plot_data['k_differential'].unique())

    # Create distinct color palette
    # Use tab20 for more colors, or combine multiple palettes if needed
    if len(unique_diffs) <= 20:
        color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        # Combine multiple palettes for more colors
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        set3 = plt.cm.Set3(np.linspace(0, 1, 12))
        color_palette = np.vstack([tab20, set3])

    diff_to_color = {}
    color_idx = 0
    for diff in unique_diffs:
        if diff == 0:
            diff_to_color[diff] = 'black'  # Highlight same-k clusters
        else:
            diff_to_color[diff] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

    colors = [diff_to_color[diff] for diff in plot_data['k_differential']]

    # Create scatter plot
    ax.scatter(
        plot_data['aanet_recon_loss_elbow_strength'],
        plot_data['aanet_archetypal_loss_elbow_strength'],
        c=colors,
        s=30,
        alpha=0.5
    )

    ax.set_xlabel('Reconstruction Loss Elbow Strength')
    ax.set_ylabel('Archetypal Loss Elbow Strength')
    ax.set_title(f'n_clusters={n_clust}')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    for diff in sorted(unique_diffs):
        label = f'Δk={int(diff)}' if diff != 0 else 'Δk=0 (same)'
        legend_elements.append(Patch(facecolor=diff_to_color[diff], label=label, alpha=0.5))
    ax.legend(handles=legend_elements, loc='best', fontsize=8, framealpha=0.9)

    # Print statistics
    same_k_pct = 100 * (plot_data['k_differential'] == 0).sum() / len(plot_data)
    print(f"\nn_clusters={n_clust}:")
    print(f"  Clusters with same elbow k: {same_k_pct:.1f}%")
    print(f"  Archetypal elbow strength range: [{plot_data['aanet_archetypal_loss_elbow_strength'].min():.4f}, {plot_data['aanet_archetypal_loss_elbow_strength'].max():.4f}]")
    print(f"  Archetypal elbow strength mean: {plot_data['aanet_archetypal_loss_elbow_strength'].mean():.4f}")

plt.suptitle('Reconstruction vs Archetypal Elbow Strength (colored by elbow k differential)',
             fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

# %% Analysis 9: Same plot but excluding top 2 values for either elbow strength
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, n_clust in enumerate(sorted(all_metrics.keys())):
    if idx >= 4:
        break
    ax = axes[idx]

    # Get elbow data for this n_clusters
    plot_data = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust].copy()

    # Calculate k differential (recon elbow k - archetypal elbow k)
    plot_data['k_differential'] = (plot_data['aanet_recon_loss_elbow_k'] -
                                    plot_data['aanet_archetypal_loss_elbow_k'])

    # Filter out top 2 values for either metric
    recon_threshold = plot_data['aanet_recon_loss_elbow_strength'].nlargest(2).min()
    arch_threshold = plot_data['aanet_archetypal_loss_elbow_strength'].nlargest(2).min()

    plot_data_filtered = plot_data[
        (plot_data['aanet_recon_loss_elbow_strength'] < recon_threshold) &
        (plot_data['aanet_archetypal_loss_elbow_strength'] < arch_threshold)
    ].copy()

    # Sort so differential == 0 is plotted last (on top)
    plot_data_filtered['plot_order'] = (plot_data_filtered['k_differential'] == 0).astype(int)
    plot_data_filtered = plot_data_filtered.sort_values('plot_order')

    # Define distinct colors for each k differential value
    unique_diffs = sorted(plot_data_filtered['k_differential'].unique())

    # Create distinct color palette
    if len(unique_diffs) <= 20:
        color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        set3 = plt.cm.Set3(np.linspace(0, 1, 12))
        color_palette = np.vstack([tab20, set3])

    diff_to_color = {}
    color_idx = 0
    for diff in unique_diffs:
        if diff == 0:
            diff_to_color[diff] = 'black'
        else:
            diff_to_color[diff] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

    colors = [diff_to_color[diff] for diff in plot_data_filtered['k_differential']]

    # Create scatter plot
    ax.scatter(
        plot_data_filtered['aanet_recon_loss_elbow_strength'],
        plot_data_filtered['aanet_archetypal_loss_elbow_strength'],
        c=colors,
        s=30,
        alpha=0.5
    )

    ax.set_xlabel('Reconstruction Loss Elbow Strength')
    ax.set_ylabel('Archetypal Loss Elbow Strength')
    ax.set_title(f'n_clusters={n_clust} (excl. top 2 outliers)')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    for diff in sorted(unique_diffs):
        label = f'Δk={int(diff)}' if diff != 0 else 'Δk=0 (same)'
        legend_elements.append(Patch(facecolor=diff_to_color[diff], label=label, alpha=0.5))
    ax.legend(handles=legend_elements, loc='best', fontsize=8, framealpha=0.9)

    # Print statistics
    n_excluded = len(plot_data) - len(plot_data_filtered)
    same_k_pct = 100 * (plot_data_filtered['k_differential'] == 0).sum() / len(plot_data_filtered)
    print(f"\nn_clusters={n_clust} (filtered):")
    print(f"  Excluded {n_excluded} clusters (top 2 in either metric)")
    print(f"  Remaining clusters: {len(plot_data_filtered)}")
    print(f"  Clusters with same elbow k: {same_k_pct:.1f}%")
    print(f"  Recon elbow strength range: [{plot_data_filtered['aanet_recon_loss_elbow_strength'].min():.4f}, {plot_data_filtered['aanet_recon_loss_elbow_strength'].max():.4f}]")
    print(f"  Archetypal elbow strength range: [{plot_data_filtered['aanet_archetypal_loss_elbow_strength'].min():.4f}, {plot_data_filtered['aanet_archetypal_loss_elbow_strength'].max():.4f}]")

plt.suptitle('Reconstruction vs Archetypal Elbow Strength (excl. top 2 outliers, colored by k differential)',
             fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

# %% Analysis 10: Cluster Selection Visualization
# (select_promising_clusters imported from cluster_selection module)

# Select promising clusters for each n_clusters value
all_selected = {}
all_category_stats = {}

print("\n" + "="*80)
print("CLUSTER SELECTION SUMMARY")
print("="*80)

for n_clust in sorted(all_elbow_df['n_clusters_total'].unique()):
    selected_ids, cat_stats = select_promising_clusters(
        all_elbow_df, n_clust,
        delta_k_threshold=DELTA_K_THRESHOLD,
        sd_outlier=SD_OUTLIER,
        sd_strong=SD_STRONG,
        verbose=True
    )
    all_selected[n_clust] = selected_ids
    all_category_stats[n_clust] = cat_stats

    print(f"\nn_clusters={n_clust}:")
    print(f"  Total clusters (after Δk filter): {len(all_elbow_df[(all_elbow_df['n_clusters_total'] == n_clust) & (abs(all_elbow_df['aanet_recon_loss_elbow_k'] - all_elbow_df['aanet_archetypal_loss_elbow_k']) <= 1)])}")
    print(f"  Selected: {len(selected_ids)}")
    print(f"    Category A (Strong Both): {len(cat_stats['A_strong_both'])}")
    print(f"    Category B (Recon Outliers): {len(cat_stats['B_recon_outliers'])}")
    print(f"    Category C (Arch Outliers): {len(cat_stats['C_arch_outliers'])}")
    print(f"    Category D (Perfect Agreement): {len(cat_stats['D_agreement'])}")

total_selected = sum(len(s) for s in all_selected.values())
print(f"\n{'='*80}")
print(f"TOTAL SELECTED CLUSTERS: {total_selected}")
print(f"{'='*80}")

# Estimate latents to interpret
selected_rows = all_elbow_df[
    all_elbow_df.apply(lambda row: row['cluster_id'] in all_selected.get(row['n_clusters_total'], set()), axis=1)
]
total_latents = selected_rows['n_latents'].sum()
print(f"\nEstimated latents to interpret: {total_latents:,}")
print(f"Estimated cost at $0.003/latent: ${total_latents * 0.003:.2f}")

# Print the actual selected cluster IDs
print("\n" + "="*80)
print("SELECTED CLUSTER IDs (for --manual_cluster_ids)")
print("="*80)
for n_clust in sorted(all_selected.keys()):
    selected_ids = sorted(all_selected[n_clust])
    cat_stats = all_category_stats[n_clust]

    print(f"\nn_clusters={n_clust}: {len(selected_ids)} clusters")
    print(f"  All IDs: {','.join(map(str, selected_ids))}")
    print(f"  --manual_cluster_ids format: \"{n_clust}:{','.join(map(str, selected_ids))}\"")

    # Also print by category
    if cat_stats['A_strong_both']:
        print(f"  Category A (Strong Both): {sorted(cat_stats['A_strong_both'])}")
    if cat_stats['B_recon_outliers']:
        print(f"  Category B (Recon Outliers): {sorted(cat_stats['B_recon_outliers'])}")
    if cat_stats['C_arch_outliers']:
        print(f"  Category C (Arch Outliers): {sorted(cat_stats['C_arch_outliers'])}")
    if cat_stats['D_agreement']:
        print(f"  Category D (Perfect Agreement): {sorted(cat_stats['D_agreement'])}")

# %% Analysis 10b: Side-by-side visualization

n_cluster_values_10b = sorted(all_elbow_df['n_clusters_total'].unique())
n_rows_10b = len(n_cluster_values_10b)
fig, axes = plt.subplots(n_rows_10b, 2, figsize=(20, 6 * n_rows_10b))

# Handle case where there's only 1 row (axes won't be 2D)
if n_rows_10b == 1:
    axes = axes.reshape(1, -1)

for idx, n_clust in enumerate(n_cluster_values_10b):

    # Get data for this n_clusters
    plot_data = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust].copy()
    plot_data['k_differential'] = (plot_data['aanet_recon_loss_elbow_k'] -
                                    plot_data['aanet_archetypal_loss_elbow_k'])

    # Sort so differential == 0 is plotted last (on top) for left plot
    plot_data['plot_order'] = (plot_data['k_differential'] == 0).astype(int)
    plot_data = plot_data.sort_values('plot_order')

    # LEFT COLUMN: Original plot with Δk coloring
    ax_left = axes[idx, 0]

    # Define distinct colors for each k differential value
    unique_diffs = sorted(plot_data['k_differential'].unique())

    if len(unique_diffs) <= 20:
        color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        set3 = plt.cm.Set3(np.linspace(0, 1, 12))
        color_palette = np.vstack([tab20, set3])

    diff_to_color = {}
    color_idx = 0
    for diff in unique_diffs:
        if diff == 0:
            diff_to_color[diff] = 'black'
        else:
            diff_to_color[diff] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

    colors = [diff_to_color[diff] for diff in plot_data['k_differential']]

    ax_left.scatter(
        plot_data['aanet_recon_loss_elbow_strength'],
        plot_data['aanet_archetypal_loss_elbow_strength'],
        c=colors,
        s=30,
        alpha=0.5
    )

    ax_left.set_xlabel('Reconstruction Loss Elbow Strength', fontsize=10)
    ax_left.set_ylabel('Archetypal Loss Elbow Strength', fontsize=10)
    ax_left.set_title(f'n_clusters={n_clust} (Original, colored by Δk)', fontsize=11)
    ax_left.grid(True, alpha=0.3)

    # Add legend for left plot
    from matplotlib.patches import Patch
    legend_elements = []
    for diff in sorted(unique_diffs)[:10]:  # Limit legend entries
        label = f'Δk={int(diff)}' if diff != 0 else 'Δk=0'
        legend_elements.append(Patch(facecolor=diff_to_color[diff], label=label, alpha=0.5))
    ax_left.legend(handles=legend_elements, loc='best', fontsize=8, framealpha=0.9)

    # RIGHT COLUMN: Selected by Category
    ax_right = axes[idx, 1]

    # Calculate quality-filtered data to get thresholds
    quality_filtered = plot_data.copy()
    quality_filtered = quality_filtered[quality_filtered['n_latents'] >= 2].copy()
    quality_filtered = quality_filtered[quality_filtered['recon_is_monotonic'] == True].copy()
    quality_filtered = quality_filtered[quality_filtered['arch_is_monotonic'] == True].copy()
    quality_filtered = quality_filtered[quality_filtered['recon_pct_decrease'] >= 20].copy()
    quality_filtered = quality_filtered[quality_filtered['arch_pct_decrease'] >= 20].copy()
    quality_filtered = quality_filtered[quality_filtered['k_differential'].abs() <= DELTA_K_THRESHOLD].copy()

    # Calculate thresholds from quality-filtered data
    recon_mean = quality_filtered['aanet_recon_loss_elbow_strength'].mean()
    recon_std = quality_filtered['aanet_recon_loss_elbow_strength'].std()
    arch_mean = quality_filtered['aanet_archetypal_loss_elbow_strength'].mean()
    arch_std = quality_filtered['aanet_archetypal_loss_elbow_strength'].std()
    recon_strong = recon_mean + SD_STRONG * recon_std
    arch_strong = arch_mean + SD_STRONG * arch_std

    # Mark selected vs rejected
    selected_set = all_selected[n_clust]
    plot_data['is_selected'] = plot_data['cluster_id'].isin(selected_set)

    # Identify clusters that meet strength threshold but failed quality filters
    plot_data['meets_strength'] = (
        (plot_data['aanet_recon_loss_elbow_strength'] > recon_strong) &
        (plot_data['aanet_archetypal_loss_elbow_strength'] > arch_strong)
    )
    plot_data['passed_quality'] = plot_data['cluster_id'].isin(quality_filtered['cluster_id'])
    plot_data['failed_quality_but_strong'] = plot_data['meets_strength'] & ~plot_data['passed_quality']

    # Plot rejected first (gray) - exclude the "failed quality but strong" ones
    rejected = plot_data[~plot_data['is_selected'] & ~plot_data['failed_quality_but_strong']]
    ax_right.scatter(
        rejected['aanet_recon_loss_elbow_strength'],
        rejected['aanet_archetypal_loss_elbow_strength'],
        c='lightgray',
        s=30,
        alpha=0.3,
        label=f'Rejected ({len(rejected)})'
    )

    # Plot "failed quality but strong" as new category E (purple)
    failed_quality_strong = plot_data[plot_data['failed_quality_but_strong']]
    if len(failed_quality_strong) > 0:
        ax_right.scatter(
            failed_quality_strong['aanet_recon_loss_elbow_strength'],
            failed_quality_strong['aanet_archetypal_loss_elbow_strength'],
            c='#9467bd',  # purple
            s=50,
            alpha=0.7,
            marker='s',  # square marker to distinguish
            label=f'E: Strong but Failed Quality ({len(failed_quality_strong)})'
        )

    # Assign category to each selected cluster (priority: A, D, B, C)
    cat_stats = all_category_stats[n_clust]
    category_colors = {
        'A': '#1f77b4',  # blue
        'D': '#ff7f0e',  # orange
        'B': '#2ca02c',  # green
        'C': '#d62728',  # red
    }
    category_names = {
        'A': 'Strong Both',
        'D': 'Perfect Agreement',
        'B': 'Recon Outliers',
        'C': 'Arch Outliers',
    }

    def get_cluster_category(cluster_id):
        # Priority order: A, D, B, C
        if cluster_id in cat_stats['A_strong_both']:
            return 'A'
        elif cluster_id in cat_stats['D_agreement']:
            return 'D'
        elif cluster_id in cat_stats['B_recon_outliers']:
            return 'B'
        elif cluster_id in cat_stats['C_arch_outliers']:
            return 'C'
        return None

    # Group selected points by category
    selected = plot_data[plot_data['is_selected']].copy()
    selected['category'] = selected['cluster_id'].apply(get_cluster_category)

    # Plot each category
    for cat in ['A', 'D', 'B', 'C']:
        cat_data = selected[selected['category'] == cat]
        if len(cat_data) > 0:
            ax_right.scatter(
                cat_data['aanet_recon_loss_elbow_strength'],
                cat_data['aanet_archetypal_loss_elbow_strength'],
                c=category_colors[cat],
                s=50,
                alpha=0.7,
                label=f'{cat}: {category_names[cat]} ({len(cat_data)})'
            )

    ax_right.set_xlabel('Reconstruction Loss Elbow Strength', fontsize=10)
    ax_right.set_ylabel('Archetypal Loss Elbow Strength', fontsize=10)
    ax_right.set_title(f'n_clusters={n_clust} (Selected by Category)', fontsize=11)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(loc='best', fontsize=9, framealpha=0.9)

plt.suptitle('Cluster Selection Visualization: Original (left) vs Selected by Category (right)',
             fontsize=14, y=0.995)
plt.tight_layout()
plt.show()

# %% Analysis 10c: Zoomed version (excluding top 2 outliers)

n_cluster_values = sorted(all_elbow_df['n_clusters_total'].unique())
n_rows = len(n_cluster_values)
fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows))

# Handle case where there's only 1 row (axes won't be 2D)
if n_rows == 1:
    axes = axes.reshape(1, -1)

for idx, n_clust in enumerate(n_cluster_values):

    # Get data for this n_clusters
    plot_data = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust].copy()

    # Apply quality filters to get the filtered dataset
    quality_filtered = plot_data.copy()
    quality_filtered = quality_filtered[quality_filtered['n_latents'] >= 2].copy()
    quality_filtered = quality_filtered[quality_filtered['recon_is_monotonic'] == True].copy()
    quality_filtered = quality_filtered[quality_filtered['arch_is_monotonic'] == True].copy()
    quality_filtered = quality_filtered[quality_filtered['recon_pct_decrease'] >= 20].copy()
    quality_filtered = quality_filtered[quality_filtered['arch_pct_decrease'] >= 20].copy()
    quality_filtered = quality_filtered[quality_filtered['k_differential'].abs() <= DELTA_K_THRESHOLD].copy()

    # Calculate mean and SD from quality-filtered data for axis limits
    recon_mean = quality_filtered['aanet_recon_loss_elbow_strength'].mean()
    recon_std = quality_filtered['aanet_recon_loss_elbow_strength'].std()
    arch_mean = quality_filtered['aanet_archetypal_loss_elbow_strength'].mean()
    arch_std = quality_filtered['aanet_archetypal_loss_elbow_strength'].std()

    # Set axis limits to mean + 10*SD
    xlim_max = recon_mean + 50 * recon_std
    ylim_max = arch_mean + 10 * arch_std
    print(f"xlim_max: {xlim_max}, ylim_max: {ylim_max}")

    # For display, filter out top 2 values for either metric
    plot_data['k_differential'] = (plot_data['aanet_recon_loss_elbow_k'] -
                                    plot_data['aanet_archetypal_loss_elbow_k'])
    recon_threshold = plot_data['aanet_recon_loss_elbow_strength'].nlargest(2).min()
    arch_threshold = plot_data['aanet_archetypal_loss_elbow_strength'].nlargest(2).min()

    plot_data_filtered = plot_data[
        (plot_data['aanet_recon_loss_elbow_strength'] < recon_threshold) &
        (plot_data['aanet_archetypal_loss_elbow_strength'] < arch_threshold)
    ].copy()

    # Sort so differential == 0 is plotted last (on top) for left plot
    plot_data_filtered['plot_order'] = (plot_data_filtered['k_differential'] == 0).astype(int)
    plot_data_filtered = plot_data_filtered.sort_values('plot_order')

    # LEFT COLUMN: Original plot with Δk coloring
    ax_left = axes[idx, 0]

    # Define distinct colors for each k differential value
    unique_diffs = sorted(plot_data_filtered['k_differential'].unique())

    if len(unique_diffs) <= 20:
        color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        set3 = plt.cm.Set3(np.linspace(0, 1, 12))
        color_palette = np.vstack([tab20, set3])

    diff_to_color = {}
    color_idx = 0
    for diff in unique_diffs:
        if diff == 0:
            diff_to_color[diff] = 'black'
        else:
            diff_to_color[diff] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

    colors = [diff_to_color[diff] for diff in plot_data_filtered['k_differential']]

    ax_left.scatter(
        plot_data_filtered['aanet_recon_loss_elbow_strength'],
        plot_data_filtered['aanet_archetypal_loss_elbow_strength'],
        c=colors,
        s=30,
        alpha=0.5
    )

    ax_left.set_xlabel('Reconstruction Loss Elbow Strength', fontsize=10)
    ax_left.set_ylabel('Archetypal Loss Elbow Strength', fontsize=10)
    ax_left.set_title(f'n_clusters={n_clust} (Zoomed, colored by Δk)', fontsize=11)
    ax_left.grid(True, alpha=0.3)

    # Set axis limits to mean + 10*SD (from quality-filtered data)
    ax_left.set_xlim(0, xlim_max)
    ax_left.set_ylim(0, ylim_max)

    # Add legend for left plot
    from matplotlib.patches import Patch
    legend_elements = []
    for diff in sorted(unique_diffs)[:10]:  # Limit legend entries
        label = f'Δk={int(diff)}' if diff != 0 else 'Δk=0'
        legend_elements.append(Patch(facecolor=diff_to_color[diff], label=label, alpha=0.5))
    ax_left.legend(handles=legend_elements, loc='best', fontsize=8, framealpha=0.9)

    # RIGHT COLUMN: Selected by Category
    ax_right = axes[idx, 1]

    # Calculate strength thresholds from quality-filtered data
    recon_strong = recon_mean + SD_STRONG * recon_std
    arch_strong = arch_mean + SD_STRONG * arch_std

    # Mark selected vs rejected
    selected_set = all_selected[n_clust]
    plot_data_filtered['is_selected'] = plot_data_filtered['cluster_id'].isin(selected_set)

    # Identify clusters that meet strength threshold but failed quality filters
    plot_data_filtered['meets_strength'] = (
        (plot_data_filtered['aanet_recon_loss_elbow_strength'] > recon_strong) &
        (plot_data_filtered['aanet_archetypal_loss_elbow_strength'] > arch_strong)
    )
    plot_data_filtered['passed_quality'] = plot_data_filtered['cluster_id'].isin(quality_filtered['cluster_id'])
    plot_data_filtered['failed_quality_but_strong'] = plot_data_filtered['meets_strength'] & ~plot_data_filtered['passed_quality']

    # Plot rejected first (gray) - exclude the "failed quality but strong" ones
    rejected = plot_data_filtered[~plot_data_filtered['is_selected'] & ~plot_data_filtered['failed_quality_but_strong']]
    ax_right.scatter(
        rejected['aanet_recon_loss_elbow_strength'],
        rejected['aanet_archetypal_loss_elbow_strength'],
        c='lightgray',
        s=30,
        alpha=0.3,
        label=f'Rejected ({len(rejected)})'
    )

    # Plot "failed quality but strong" as new category E (purple)
    failed_quality_strong = plot_data_filtered[plot_data_filtered['failed_quality_but_strong']]
    if len(failed_quality_strong) > 0:
        ax_right.scatter(
            failed_quality_strong['aanet_recon_loss_elbow_strength'],
            failed_quality_strong['aanet_archetypal_loss_elbow_strength'],
            c='#9467bd',  # purple
            s=50,
            alpha=0.7,
            marker='s',  # square marker to distinguish
            label=f'E: Strong but Failed Quality ({len(failed_quality_strong)})'
        )

    # Assign category to each selected cluster (priority: A, D, B, C)
    cat_stats = all_category_stats[n_clust]
    category_colors = {
        'A': '#1f77b4',  # blue
        'D': '#ff7f0e',  # orange
        'B': '#2ca02c',  # green
        'C': '#d62728',  # red
    }
    category_names = {
        'A': 'Strong Both',
        'D': 'Perfect Agreement',
        'B': 'Recon Outliers',
        'C': 'Arch Outliers',
    }

    def get_cluster_category(cluster_id):
        # Priority order: A, D, B, C
        if cluster_id in cat_stats['A_strong_both']:
            return 'A'
        elif cluster_id in cat_stats['D_agreement']:
            return 'D'
        elif cluster_id in cat_stats['B_recon_outliers']:
            return 'B'
        elif cluster_id in cat_stats['C_arch_outliers']:
            return 'C'
        return None

    # Group selected points by category
    selected = plot_data_filtered[plot_data_filtered['is_selected']].copy()
    selected['category'] = selected['cluster_id'].apply(get_cluster_category)

    # Plot each category
    for cat in ['A', 'D', 'B', 'C']:
        cat_data = selected[selected['category'] == cat]
        if len(cat_data) > 0:
            ax_right.scatter(
                cat_data['aanet_recon_loss_elbow_strength'],
                cat_data['aanet_archetypal_loss_elbow_strength'],
                c=category_colors[cat],
                s=50,
                alpha=0.7,
                label=f'{cat}: {category_names[cat]} ({len(cat_data)})'
            )

    ax_right.set_xlabel('Reconstruction Loss Elbow Strength', fontsize=10)
    ax_right.set_ylabel('Archetypal Loss Elbow Strength', fontsize=10)
    ax_right.set_title(f'n_clusters={n_clust} (Zoomed, Selected by Category)', fontsize=11)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(loc='best', fontsize=9, framealpha=0.9)

    # Set axis limits to mean + 10*SD (from quality-filtered data)
    ax_right.set_xlim(0, xlim_max)
    ax_right.set_ylim(0, ylim_max)

plt.suptitle('Cluster Selection Visualization - ZOOMED (excl. top 2 outliers): Original (left) vs Selected by Category (right)',
             fontsize=14, y=0.995)
plt.tight_layout()
plt.show()

# %% Analysis 10d: Rejection Diagnostics - Why are clusters being rejected?
# This helps understand why clusters in the "balanced" region (near y=x) aren't being selected

print("\n" + "="*100)
print("REJECTION DIAGNOSTICS: Why are clusters being rejected?")
print("="*100)

for n_clust in sorted(all_elbow_df['n_clusters_total'].unique()):
    print(f"\n{'='*80}")
    print(f"n_clusters = {n_clust}")
    print(f"{'='*80}")

    group = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust].copy()
    total = len(group)
    print(f"Total clusters: {total}")

    # Track rejection reasons
    rejection_counts = {}

    # Quality filter 1: n_latents >= 2
    failed_n_latents = group[group['n_latents'] < 2]
    rejection_counts['n_latents < 2'] = len(failed_n_latents)
    group = group[group['n_latents'] >= 2].copy()

    # Quality filter 2: recon_is_monotonic
    failed_recon_mono = group[group['recon_is_monotonic'] != True]
    rejection_counts['recon_is_monotonic=False'] = len(failed_recon_mono)
    group = group[group['recon_is_monotonic'] == True].copy()

    # Quality filter 3: arch_is_monotonic
    failed_arch_mono = group[group['arch_is_monotonic'] != True]
    rejection_counts['arch_is_monotonic=False'] = len(failed_arch_mono)
    group = group[group['arch_is_monotonic'] == True].copy()

    # Quality filter 4: recon_pct_decrease >= 20
    failed_recon_pct = group[group['recon_pct_decrease'] < 20]
    rejection_counts['recon_pct_decrease < 20'] = len(failed_recon_pct)
    group = group[group['recon_pct_decrease'] >= 20].copy()

    # Quality filter 5: arch_pct_decrease >= 20
    failed_arch_pct = group[group['arch_pct_decrease'] < 20]
    rejection_counts['arch_pct_decrease < 20'] = len(failed_arch_pct)
    group = group[group['arch_pct_decrease'] >= 20].copy()

    # Quality filter 6: k_differential <= 1
    failed_k_diff = group[group['k_differential'].abs() > 1]
    rejection_counts['|k_differential| > 1'] = len(failed_k_diff)
    group = group[group['k_differential'].abs() <= DELTA_K_THRESHOLD].copy()

    passed_quality = len(group)
    print(f"\nQuality filter breakdown (cumulative):")
    for reason, count in rejection_counts.items():
        print(f"  {reason}: {count} rejected")
    print(f"  → Passed all quality filters: {passed_quality}/{total} ({100*passed_quality/total:.1f}%)")

    if passed_quality == 0:
        continue

    # Now check selection thresholds
    recon_mean = group['aanet_recon_loss_elbow_strength'].mean()
    recon_std = group['aanet_recon_loss_elbow_strength'].std()
    arch_mean = group['aanet_archetypal_loss_elbow_strength'].mean()
    arch_std = group['aanet_archetypal_loss_elbow_strength'].std()

    # Selection thresholds
    recon_strong = recon_mean + SD_STRONG * recon_std  # For category A/D
    arch_strong = arch_mean + SD_STRONG * arch_std
    recon_outlier = recon_mean + SD_OUTLIER * recon_std  # For category B/C
    arch_outlier = arch_mean + SD_OUTLIER * arch_std

    print(f"\nSelection thresholds (on quality-filtered data):")
    print(f"  Recon: mean={recon_mean:.4f}, SD={recon_std:.4f}")
    print(f"    Strong (A/D): > {recon_strong:.4f} (mean + {SD_STRONG}*SD)")
    print(f"    Outlier (B): > {recon_outlier:.4f} (mean + {SD_OUTLIER}*SD)")
    print(f"  Arch: mean={arch_mean:.4f}, SD={arch_std:.4f}")
    print(f"    Strong (A/D): > {arch_strong:.4f} (mean + {SD_STRONG}*SD)")
    print(f"    Outlier (C): > {arch_outlier:.4f} (mean + {SD_OUTLIER}*SD)")

    # Count how many meet each criterion
    strong_recon = group['aanet_recon_loss_elbow_strength'] > recon_strong
    strong_arch = group['aanet_archetypal_loss_elbow_strength'] > arch_strong
    outlier_recon = group['aanet_recon_loss_elbow_strength'] > recon_outlier
    outlier_arch = group['aanet_archetypal_loss_elbow_strength'] > arch_outlier
    k_diff_zero = group['k_differential'] == 0

    print(f"\nSelection criterion counts (quality-filtered clusters):")
    print(f"  Recon > mean+{SD_STRONG}*SD: {strong_recon.sum()}/{passed_quality}")
    print(f"  Arch > mean+{SD_STRONG}*SD: {strong_arch.sum()}/{passed_quality}")
    print(f"  Recon > mean+{SD_OUTLIER}*SD (outlier): {outlier_recon.sum()}/{passed_quality}")
    print(f"  Arch > mean+{SD_OUTLIER}*SD (outlier): {outlier_arch.sum()}/{passed_quality}")
    print(f"  k_differential == 0: {k_diff_zero.sum()}/{passed_quality}")

    # Category breakdown
    cat_a = (strong_recon & strong_arch).sum()
    cat_b = outlier_recon.sum()
    cat_c = outlier_arch.sum()
    cat_d = (k_diff_zero & ((group['aanet_recon_loss_elbow_strength'] > recon_mean) |
                            (group['aanet_archetypal_loss_elbow_strength'] > arch_mean))).sum()

    selected = len(all_selected.get(n_clust, set()))
    not_selected = passed_quality - selected

    print(f"\nCategory counts:")
    print(f"  A (strong both): {cat_a}")
    print(f"  B (recon outlier): {cat_b}")
    print(f"  C (arch outlier): {cat_c}")
    print(f"  D (perfect agreement): {cat_d}")
    print(f"  Total selected: {selected}")
    print(f"  NOT selected (passed quality but below thresholds): {not_selected}")

# %% Analysis 10e: Details on "Strong but Failed Quality" clusters (purple squares)
# These are clusters that have strong elbow strengths but failed quality filters

print("\n" + "="*100)
print("DETAILS ON 'STRONG BUT FAILED QUALITY' CLUSTERS (Purple Squares)")
print("="*100)
print("These clusters have both recon AND arch elbow strength > mean + SD_STRONG*std")
print("but failed at least one quality filter.")

for n_clust in sorted(all_elbow_df['n_clusters_total'].unique()):
    print(f"\n{'='*80}")
    print(f"n_clusters = {n_clust}")
    print(f"{'='*80}")

    # Get all data for this n_clusters
    group = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust].copy()

    # Calculate quality-filtered data to get thresholds
    quality_filtered = group.copy()
    quality_filtered = quality_filtered[quality_filtered['n_latents'] >= 2].copy()
    quality_filtered = quality_filtered[quality_filtered['recon_is_monotonic'] == True].copy()
    quality_filtered = quality_filtered[quality_filtered['arch_is_monotonic'] == True].copy()
    quality_filtered = quality_filtered[quality_filtered['recon_pct_decrease'] >= 20].copy()
    quality_filtered = quality_filtered[quality_filtered['arch_pct_decrease'] >= 20].copy()
    quality_filtered = quality_filtered[quality_filtered['k_differential'].abs() <= DELTA_K_THRESHOLD].copy()

    if len(quality_filtered) == 0:
        print("No clusters passed quality filters - cannot calculate thresholds")
        continue

    # Calculate thresholds from quality-filtered data
    recon_mean = quality_filtered['aanet_recon_loss_elbow_strength'].mean()
    recon_std = quality_filtered['aanet_recon_loss_elbow_strength'].std()
    arch_mean = quality_filtered['aanet_archetypal_loss_elbow_strength'].mean()
    arch_std = quality_filtered['aanet_archetypal_loss_elbow_strength'].std()
    recon_strong = recon_mean + SD_STRONG * recon_std
    arch_strong = arch_mean + SD_STRONG * arch_std

    # Find clusters that meet strength threshold
    group['meets_strength'] = (
        (group['aanet_recon_loss_elbow_strength'] > recon_strong) &
        (group['aanet_archetypal_loss_elbow_strength'] > arch_strong)
    )
    group['passed_quality'] = group['cluster_id'].isin(quality_filtered['cluster_id'])

    # The purple squares: strong but failed quality
    purple_squares = group[group['meets_strength'] & ~group['passed_quality']].copy()

    if len(purple_squares) == 0:
        print("No 'strong but failed quality' clusters found")
        continue

    print(f"Found {len(purple_squares)} clusters with strong elbow strengths that failed quality filters:\n")

    # For each purple square, identify which filter(s) it failed
    for _, row in purple_squares.iterrows():
        cid = row['cluster_id']
        print(f"  Cluster {cid}:")
        print(f"    n_latents: {row['n_latents']}")
        print(f"    Recon elbow strength: {row['aanet_recon_loss_elbow_strength']:.4f} (threshold: {recon_strong:.4f})")
        print(f"    Arch elbow strength: {row['aanet_archetypal_loss_elbow_strength']:.4f} (threshold: {arch_strong:.4f})")
        print(f"    k_differential: {row['k_differential']}")

        # Check which filters failed
        failed_filters = []
        if row['n_latents'] < 2:
            failed_filters.append(f"n_latents < 2 (has {row['n_latents']})")
        if row['recon_is_monotonic'] != True:
            failed_filters.append(f"recon_is_monotonic=False")
        if row['arch_is_monotonic'] != True:
            failed_filters.append(f"arch_is_monotonic=False")
        if row['recon_pct_decrease'] < 20:
            failed_filters.append(f"recon_pct_decrease={row['recon_pct_decrease']:.1f}% < 20%")
        if row['arch_pct_decrease'] < 20:
            failed_filters.append(f"arch_pct_decrease={row['arch_pct_decrease']:.1f}% < 20%")
        if abs(row['k_differential']) > DELTA_K_THRESHOLD:
            failed_filters.append(f"|k_differential|={abs(row['k_differential'])} > {DELTA_K_THRESHOLD}")

        print(f"    FAILED FILTERS: {', '.join(failed_filters)}")
        print()

    # Plot elbow curves for each purple square cluster
    print(f"\nPlotting elbow curves for {len(purple_squares)} purple square clusters...")

    for _, row in purple_squares.iterrows():
        cid = row['cluster_id']

        # Get all k values for this cluster from the full metrics
        if n_clust in all_metrics:
            cluster_data = all_metrics[n_clust][all_metrics[n_clust]['cluster_id'] == cid].sort_values('aanet_k')

            if len(cluster_data) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                k_vals = cluster_data['aanet_k'].values
                recon_losses = cluster_data['aanet_recon_loss'].values
                arch_losses = cluster_data['aanet_archetypal_loss'].values

                # Get elbow k values
                recon_elbow_k = row['aanet_recon_loss_elbow_k']
                arch_elbow_k = row['aanet_archetypal_loss_elbow_k']

                # Left plot: Reconstruction loss
                ax = axes[0]
                ax.plot(k_vals, recon_losses, 'b-o', markersize=8, linewidth=2)
                if recon_elbow_k in k_vals:
                    elbow_idx = list(k_vals).index(recon_elbow_k)
                    ax.axvline(x=recon_elbow_k, color='red', linestyle='--', alpha=0.7, label=f'Elbow k={int(recon_elbow_k)}')
                    ax.plot(recon_elbow_k, recon_losses[elbow_idx], 'ro', markersize=12, zorder=5)
                ax.set_xlabel('k (number of vertices)', fontsize=11)
                ax.set_ylabel('Reconstruction Loss', fontsize=11)
                ax.set_title(f'Reconstruction Loss\nStrength={row["aanet_recon_loss_elbow_strength"]:.4f}, Monotonic={row["recon_is_monotonic"]}, Decrease={row["recon_pct_decrease"]:.1f}%', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Right plot: Archetypal loss
                ax = axes[1]
                ax.plot(k_vals, arch_losses, 'g-o', markersize=8, linewidth=2)
                if arch_elbow_k in k_vals:
                    elbow_idx = list(k_vals).index(arch_elbow_k)
                    ax.axvline(x=arch_elbow_k, color='red', linestyle='--', alpha=0.7, label=f'Elbow k={int(arch_elbow_k)}')
                    ax.plot(arch_elbow_k, arch_losses[elbow_idx], 'ro', markersize=12, zorder=5)
                ax.set_xlabel('k (number of vertices)', fontsize=11)
                ax.set_ylabel('Archetypal Loss', fontsize=11)
                ax.set_title(f'Archetypal Loss\nStrength={row["aanet_archetypal_loss_elbow_strength"]:.4f}, Monotonic={row["arch_is_monotonic"]}, Decrease={row["arch_pct_decrease"]:.1f}%', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Build failed filters string for suptitle
                failed_filters = []
                if row['n_latents'] < 2:
                    failed_filters.append(f"n_latents<2")
                if row['recon_is_monotonic'] != True:
                    failed_filters.append(f"recon_monotonic=F")
                if row['arch_is_monotonic'] != True:
                    failed_filters.append(f"arch_monotonic=F")
                if row['recon_pct_decrease'] < 20:
                    failed_filters.append(f"recon_dec<20%")
                if row['arch_pct_decrease'] < 20:
                    failed_filters.append(f"arch_dec<20%")
                if abs(row['k_differential']) > DELTA_K_THRESHOLD:
                    failed_filters.append(f"|Δk|>{DELTA_K_THRESHOLD}")

                plt.suptitle(f"Cluster {cid} (n={n_clust}, {row['n_latents']} latents) - FAILED: {', '.join(failed_filters)}",
                            fontsize=12, y=1.02)
                plt.tight_layout()
                plt.show()

# %% Print detailed information about selected latents
print("\n" + "="*100)
print("DETAILED SELECTED LATENT INFORMATION")
print("="*100)

for n_clust in sorted(all_elbow_df['n_clusters_total'].unique()):
    print(f"\n{'='*100}")
    print(f"n_clusters = {n_clust}")
    print(f"{'='*100}")

    # Get data for this n_clusters
    group = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust].copy()

    # Apply quality filters (same as in select_promising_clusters)
    group = group[group['n_latents'] >= 2].copy()
    group = group[group['recon_is_monotonic'] == True].copy()
    group = group[group['arch_is_monotonic'] == True].copy()
    group = group[group['recon_pct_decrease'] >= 20].copy()
    group = group[group['arch_pct_decrease'] >= 20].copy()
    group = group[group['k_differential'].abs() <= DELTA_K_THRESHOLD].copy()

    if len(group) == 0:
        print("No clusters after quality filters")
        continue

    # Calculate mean and std for this FILTERED group
    recon_mean = group['aanet_recon_loss_elbow_strength'].mean()
    recon_std = group['aanet_recon_loss_elbow_strength'].std()
    arch_mean = group['aanet_archetypal_loss_elbow_strength'].mean()
    arch_std = group['aanet_archetypal_loss_elbow_strength'].std()

    # Get selected clusters
    selected_set = all_selected[n_clust]
    cat_stats = all_category_stats[n_clust]

    if len(selected_set) == 0:
        print("No clusters selected")
        continue

    # Filter to selected clusters and add category
    selected_df = group[group['cluster_id'].isin(selected_set)].copy()

    def get_cluster_category(cluster_id):
        if cluster_id in cat_stats['A_strong_both']:
            return 'A'
        elif cluster_id in cat_stats['D_agreement']:
            return 'D'
        elif cluster_id in cat_stats['B_recon_outliers']:
            return 'B'
        elif cluster_id in cat_stats['C_arch_outliers']:
            return 'C'
        return None

    selected_df['category'] = selected_df['cluster_id'].apply(get_cluster_category)

    # Calculate SDs above mean
    selected_df['recon_sds'] = (selected_df['aanet_recon_loss_elbow_strength'] - recon_mean) / recon_std
    selected_df['arch_sds'] = (selected_df['aanet_archetypal_loss_elbow_strength'] - arch_mean) / arch_std

    # Sort by category priority (A, D, B, C), then by distance from origin
    category_order = {'A': 0, 'D': 1, 'B': 2, 'C': 3}
    selected_df['cat_order'] = selected_df['category'].map(category_order)
    selected_df['distance'] = np.sqrt(selected_df['aanet_recon_loss_elbow_strength']**2 +
                                      selected_df['aanet_archetypal_loss_elbow_strength']**2)
    selected_df = selected_df.sort_values(['cat_order', 'distance'], ascending=[True, False])

    # Print header
    print(f"\n{'Category':<10} {'Cluster ID':<12} {'N_Latents':<12} {'Recon k':<10} {'Recon Strength (SDs)':<25} {'Arch k':<10} {'Arch Strength (SDs)':<25}")
    print("-" * 104)

    # Print each selected cluster
    for _, row in selected_df.iterrows():
        cat = row['category']
        cluster_id = row['cluster_id']
        n_latents = row['n_latents']
        recon_k = int(row['aanet_recon_loss_elbow_k'])
        recon_strength = row['aanet_recon_loss_elbow_strength']
        recon_sds = row['recon_sds']
        arch_k = int(row['aanet_archetypal_loss_elbow_k'])
        arch_strength = row['aanet_archetypal_loss_elbow_strength']
        arch_sds = row['arch_sds']

        # Print main row
        print(f"{cat:<10} {cluster_id:<12} {n_latents:<12} {recon_k:<10} {recon_strength:.4f} ({recon_sds:+.2f} SD){'':<8} {arch_k:<10} {arch_strength:.4f} ({arch_sds:+.2f} SD)")

        # Print latent indices on next line if available
        if 'latent_indices' in row.index:
            latent_indices = row['latent_indices']
            print(f"{'':>10} Latent indices: {latent_indices}")
        print()  # Empty line between clusters

    # Plot elbow curves for each selected cluster
    print(f"\nPlotting elbow curves for {len(selected_df)} selected clusters...")

    for _, row in selected_df.iterrows():
        cid = row['cluster_id']
        cat = row['category']

        # Get all k values for this cluster from the full metrics
        if n_clust in all_metrics:
            cluster_data = all_metrics[n_clust][all_metrics[n_clust]['cluster_id'] == cid].sort_values('aanet_k')

            if len(cluster_data) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                k_vals = cluster_data['aanet_k'].values
                recon_losses = cluster_data['aanet_recon_loss'].values
                arch_losses = cluster_data['aanet_archetypal_loss'].values

                # Get elbow k values
                recon_elbow_k = row['aanet_recon_loss_elbow_k']
                arch_elbow_k = row['aanet_archetypal_loss_elbow_k']

                # Left plot: Reconstruction loss
                ax = axes[0]
                ax.plot(k_vals, recon_losses, 'b-o', markersize=8, linewidth=2)
                if recon_elbow_k in k_vals:
                    elbow_idx = list(k_vals).index(recon_elbow_k)
                    ax.axvline(x=recon_elbow_k, color='red', linestyle='--', alpha=0.7, label=f'Elbow k={int(recon_elbow_k)}')
                    ax.plot(recon_elbow_k, recon_losses[elbow_idx], 'ro', markersize=12, zorder=5)
                ax.set_xlabel('k (number of vertices)', fontsize=11)
                ax.set_ylabel('Reconstruction Loss', fontsize=11)
                ax.set_title(f'Reconstruction Loss\nStrength={row["aanet_recon_loss_elbow_strength"]:.4f} ({row["recon_sds"]:+.2f} SD), Decrease={row["recon_pct_decrease"]:.1f}%', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Right plot: Archetypal loss
                ax = axes[1]
                ax.plot(k_vals, arch_losses, 'g-o', markersize=8, linewidth=2)
                if arch_elbow_k in k_vals:
                    elbow_idx = list(k_vals).index(arch_elbow_k)
                    ax.axvline(x=arch_elbow_k, color='red', linestyle='--', alpha=0.7, label=f'Elbow k={int(arch_elbow_k)}')
                    ax.plot(arch_elbow_k, arch_losses[elbow_idx], 'ro', markersize=12, zorder=5)
                ax.set_xlabel('k (number of vertices)', fontsize=11)
                ax.set_ylabel('Archetypal Loss', fontsize=11)
                ax.set_title(f'Archetypal Loss\nStrength={row["aanet_archetypal_loss_elbow_strength"]:.4f} ({row["arch_sds"]:+.2f} SD), Decrease={row["arch_pct_decrease"]:.1f}%', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Category colors for title
                cat_colors = {'A': 'blue', 'B': 'green', 'C': 'red', 'D': 'orange'}
                cat_names = {'A': 'Strong Both', 'B': 'Recon Outlier', 'C': 'Arch Outlier', 'D': 'Perfect Agreement'}

                plt.suptitle(f"Cluster {cid} (n={n_clust}, {row['n_latents']} latents) - Category {cat}: {cat_names.get(cat, 'Unknown')}",
                            fontsize=12, y=1.02, color=cat_colors.get(cat, 'black'))
                plt.tight_layout()
                plt.show()

print("\n" + "="*100)

# %% Analysis: Training stability by cluster size
# Calculate variance of last 20 values for each loss type, grouped by n_latents

import json

print("\nAnalyzing training stability from full training curves...")

# Define latent bins
def get_latent_bin(n_latents):
    if n_latents == 1:
        return '1'
    elif 2 <= n_latents <= 3:
        return '2-3'
    elif 4 <= n_latents <= 6:
        return '4-6'
    elif 7 <= n_latents <= 10:
        return '7-10'
    elif 11 <= n_latents <= 15:
        return '11-15'
    elif 16 <= n_latents <= 30:
        return '16-30'
    elif 31 <= n_latents <= 50:
        return '31-50'
    else:
        return '50+'

# Load training curves for n=768 (or whichever n_clusters you're analyzing)
training_curves_path = base_dir / f"clusters_{primary_n_clusters}" / f"training_curves_n{primary_n_clusters}.jsonl"

stability_data = {
    'loss': [],
    'reconstruction_loss': [],
    'archetypal_loss': [],
    'extrema_loss': []
}

if training_curves_path.exists():
    print(f"Loading training curves from {training_curves_path}")

    with open(training_curves_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            cluster_id = data['cluster_id']
            aanet_k = data['aanet_k']

            # Get n_latents for this cluster
            cluster_data = all_metrics[primary_n_clusters]
            cluster_info = cluster_data[cluster_data['cluster_id'] == cluster_id]
            if len(cluster_info) == 0:
                continue
            n_latents = cluster_info.iloc[0]['n_latents']
            latent_bin = get_latent_bin(n_latents)

            # Calculate variance of last 20 values for each loss type
            hist = data['metrics_history']
            for loss_type in stability_data.keys():
                if loss_type in hist and len(hist[loss_type]) >= 20:
                    last_20 = hist[loss_type][-20:]
                    variance = np.var(last_20)
                    stability_data[loss_type].append({
                        'cluster_id': cluster_id,
                        'aanet_k': aanet_k,
                        'n_latents': n_latents,
                        'latent_bin': latent_bin,
                        'variance': variance,
                        'mean_last_20': np.mean(last_20),
                        'min_last_20': np.min(last_20),
                        'max_last_20': np.max(last_20)
                    })

    # Create boxplots for each k value separately
    loss_types_full = ['loss', 'reconstruction_loss', 'archetypal_loss', 'extrema_loss']
    loss_labels = ['Total Loss', 'Reconstruction Loss', 'Archetypal Loss', 'Extrema Loss']

    # Define bin order
    bin_order = ['1', '2-3', '4-6', '7-10', '11-15', '16-30', '31-50', '50+']

    # Get unique k values
    k_values = sorted(set([entry['aanet_k'] for entry in stability_data['loss']]))

    # Create one figure for each k value
    for k in k_values:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (loss_type, label) in enumerate(zip(loss_types_full, loss_labels)):
            ax = axes[idx]

            if not stability_data[loss_type]:
                ax.text(0.5, 0.5, f'No data for {label}', ha='center', va='center')
                continue

            # Filter for this k value
            df_stability = pd.DataFrame(stability_data[loss_type])
            df_k = df_stability[df_stability['aanet_k'] == k]

            # Prepare data for boxplot
            data_by_bin = []
            labels_used = []
            for bin_label in bin_order:
                bin_data = df_k[df_k['latent_bin'] == bin_label]['variance']
                if len(bin_data) > 0:
                    data_by_bin.append(bin_data.values)
                    labels_used.append(f'{bin_label}\n(n={len(bin_data)})')

            if data_by_bin:
                bp = ax.boxplot(data_by_bin, tick_labels=labels_used, patch_artist=True, showfliers=True)

                # Color the boxes
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data_by_bin)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xlabel('Number of Latents in Cluster')
                ax.set_ylabel('Variance of Last 20 Training Values')
                ax.set_title(f'{label} - Training Stability by Cluster Size')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=0)
            else:
                ax.text(0.5, 0.5, f'No data for {label}', ha='center', va='center')

        plt.suptitle(f'Training Stability Analysis for k={k}: Variance of Last 20 Training Steps by Cluster Size',
                     fontsize=14, y=0.995)
        plt.tight_layout()
        plt.show()

    # Print summary statistics by k value
    print("\n" + "="*80)
    print("Training Stability Summary Statistics by k value")
    print("="*80)

    for k in k_values:
        print(f"\n{'='*80}")
        print(f"k = {k}")
        print(f"{'='*80}")

        for loss_type, label in zip(loss_types_full, loss_labels):
            if not stability_data[loss_type]:
                continue

            df_stability = pd.DataFrame(stability_data[loss_type])
            df_k = df_stability[df_stability['aanet_k'] == k]

            print(f"\n{label}:")
            print(f"  {'Latent Bin':<12} {'N runs':<10} {'Median Var':<15} {'Mean Var':<15} {'Max Var':<15}")
            print("  " + "-"*70)

            for bin_label in bin_order:
                bin_data = df_k[df_k['latent_bin'] == bin_label]
                if len(bin_data) > 0:
                    variances = bin_data['variance']
                    print(f"  {bin_label:<12} {len(bin_data):<10} {variances.median():<15.6e} {variances.mean():<15.6e} {variances.max():<15.6e}")
else:
    print(f"Training curves file not found at {training_curves_path}")

#%%
