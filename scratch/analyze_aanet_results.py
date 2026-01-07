# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %% Setup - Load data
# Configure paths
base_dir = Path(__file__).parent.parent / "outputs" / "real_data_analysis_canonical"
n_clusters_list = [128, 256, 512, 768]

# Load all consolidated metrics (corrected versions)
all_metrics = {}
for n in n_clusters_list:
    csv_path = base_dir / f"clusters_{n}" / f"consolidated_metrics_n{n}_corrected.csv"
    if csv_path.exists():
        all_metrics[n] = pd.read_csv(csv_path)
        print(f"Loaded n={n}: {len(all_metrics[n])} rows")
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

# Note: activation_pca_rank is all NaN in the data, so we can't plot that
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
def calculate_elbow_score(x, y):
    """
    Calculate elbow scores using perpendicular distance from line.
    Returns the k value at the elbow and the elbow strength.

    Based on the Kneedle algorithm idea: find point of maximum distance
    from the line connecting first and last points.
    """
    if len(x) < 3:
        return None, 0.0

    # Normalize to [0, 1] range for both x and y
    x_norm = (np.array(x) - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
    y_norm = (np.array(y) - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else np.zeros_like(y)

    # Calculate perpendicular distance from line connecting first to last point
    # Line from (0,1) to (1,0) in normalized space
    # Distance = |x + y - 1| / sqrt(2)
    distances = np.abs(x_norm + y_norm - 1) / np.sqrt(2)

    # Find elbow (maximum distance)
    elbow_idx = np.argmax(distances)
    elbow_k = x[elbow_idx]
    elbow_strength = distances[elbow_idx]

    return elbow_k, elbow_strength

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

def select_promising_clusters(df, n_clusters_val, delta_k_threshold=1,
                               sd_outlier=3, sd_strong=1):
    """
    Select promising clusters based on multi-criteria approach.

    Takes ALL clusters that meet the criteria (no arbitrary top-N limits).

    All categories now use standard deviation cutoffs:
    - Categories B & C (outliers): mean + sd_outlier*SD (default 3)
    - Categories A & D (strong): mean + sd_strong*SD (default 1)

    Quality filters applied:
    - n_latents >= 2 (no single-latent clusters)
    - recon_is_monotonic == True (monotonic decrease to elbow)
    - arch_is_monotonic == True (monotonic decrease to elbow)
    - recon_pct_decrease >= 20 (at least 20% decrease from max to min)
    - arch_pct_decrease >= 20 (at least 20% decrease from max to min)
    - k_differential <= 1 (elbow agreement within 1 k value)

    Returns: set of cluster_ids that are selected
    """
    # Filter for this n_clusters value
    group = df[df['n_clusters_total'] == n_clusters_val].copy()

    # Apply quality filters with detailed statistics
    def print_filter_stats(group, label):
        recon_mean = group['aanet_recon_loss_elbow_strength'].mean()
        recon_std = group['aanet_recon_loss_elbow_strength'].std()
        arch_mean = group['aanet_archetypal_loss_elbow_strength'].mean()
        arch_std = group['aanet_archetypal_loss_elbow_strength'].std()
        print(f"  {label}: {len(group)} clusters | "
              f"recon μ={recon_mean:.4f} σ={recon_std:.4f} | "
              f"arch μ={arch_mean:.4f} σ={arch_std:.4f}")

    print_filter_stats(group, "Before filters")
    group = group[group['n_latents'] >= 2].copy()
    print_filter_stats(group, "After n_latents >= 2")
    group = group[group['recon_is_monotonic'] == True].copy()
    print_filter_stats(group, "After recon_is_monotonic")
    group = group[group['arch_is_monotonic'] == True].copy()
    print_filter_stats(group, "After arch_is_monotonic")
    group = group[group['recon_pct_decrease'] >= 20].copy()
    print_filter_stats(group, "After recon_pct_decrease >= 20")
    group = group[group['arch_pct_decrease'] >= 20].copy()
    print_filter_stats(group, "After arch_pct_decrease >= 20")

    # Filter: Delta K constraint (using k_differential from CSV if available, else calculate)
    if 'k_differential' not in group.columns:
        group['k_differential'] = (group['aanet_recon_loss_elbow_k'] -
                                    group['aanet_archetypal_loss_elbow_k'])
    group = group[group['k_differential'].abs() <= delta_k_threshold].copy()
    print_filter_stats(group, f"After k_differential <= {delta_k_threshold}")

    if len(group) == 0:
        return set(), {}

    # Calculate distance from origin for ranking
    group['distance_from_origin'] = np.sqrt(
        group['aanet_recon_loss_elbow_strength']**2 +
        group['aanet_archetypal_loss_elbow_strength']**2
    )

    # Calculate SD-based thresholds
    recon_mean = group['aanet_recon_loss_elbow_strength'].mean()
    recon_std = group['aanet_recon_loss_elbow_strength'].std()
    arch_mean = group['aanet_archetypal_loss_elbow_strength'].mean()
    arch_std = group['aanet_archetypal_loss_elbow_strength'].std()

    # Thresholds for outliers (categories B & C)
    recon_outlier_threshold = recon_mean + sd_outlier * recon_std
    arch_outlier_threshold = arch_mean + sd_outlier * arch_std

    # Thresholds for strong values (categories A & D)
    recon_strong_threshold = recon_mean + sd_strong * recon_std
    arch_strong_threshold = arch_mean + sd_strong * arch_std

    selected_clusters = set()
    category_stats = {
        'A_strong_both': [],
        'B_recon_outliers': [],
        'C_arch_outliers': [],
        'D_agreement': []
    }

    # Category A: Strong on Both Axes
    # Both above mean+1SD
    cat_a = group[
        (group['aanet_recon_loss_elbow_strength'] > recon_strong_threshold) &
        (group['aanet_archetypal_loss_elbow_strength'] > arch_strong_threshold)
    ].copy()
    category_stats['A_strong_both'] = cat_a['cluster_id'].tolist()
    selected_clusters.update(cat_a['cluster_id'])

    # Category B: Reconstruction Outliers (mean + 3*SD)
    cat_b = group[
        group['aanet_recon_loss_elbow_strength'] > recon_outlier_threshold
    ].copy()
    category_stats['B_recon_outliers'] = cat_b['cluster_id'].tolist()
    selected_clusters.update(cat_b['cluster_id'])

    # Category C: Archetypal Outliers (mean + 3*SD)
    cat_c = group[
        group['aanet_archetypal_loss_elbow_strength'] > arch_outlier_threshold
    ].copy()
    category_stats['C_arch_outliers'] = cat_c['cluster_id'].tolist()
    selected_clusters.update(cat_c['cluster_id'])

    # Category D: Perfect Agreement Standouts
    # Delta k = 0 AND both metrics above their means
    cat_d = group[
        (group['k_differential'] == 0) &
        (((group['aanet_recon_loss_elbow_strength'] > recon_mean) & (group['aanet_archetypal_loss_elbow_strength'] > arch_mean)) |
         ((group['aanet_archetypal_loss_elbow_strength'] > arch_mean) & (group['aanet_recon_loss_elbow_strength'] > recon_mean)))
    ].copy()
    category_stats['D_agreement'] = cat_d['cluster_id'].tolist()
    selected_clusters.update(cat_d['cluster_id'])

    return selected_clusters, category_stats


# Select promising clusters for each n_clusters value
all_selected = {}
all_category_stats = {}

print("\n" + "="*80)
print("CLUSTER SELECTION SUMMARY")
print("="*80)

for n_clust in sorted(all_elbow_df['n_clusters_total'].unique()):
    selected_ids, cat_stats = select_promising_clusters(all_elbow_df, n_clust)
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

# %% Analysis 10b: Side-by-side visualization

fig, axes = plt.subplots(4, 2, figsize=(20, 24))

for idx, n_clust in enumerate(sorted(all_elbow_df['n_clusters_total'].unique())):
    if idx >= 4:
        break

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

    # Mark selected vs rejected
    selected_set = all_selected[n_clust]
    plot_data['is_selected'] = plot_data['cluster_id'].isin(selected_set)

    # Plot rejected first (gray)
    rejected = plot_data[~plot_data['is_selected']]
    ax_right.scatter(
        rejected['aanet_recon_loss_elbow_strength'],
        rejected['aanet_archetypal_loss_elbow_strength'],
        c='lightgray',
        s=30,
        alpha=0.3,
        label=f'Rejected ({len(rejected)})'
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

fig, axes = plt.subplots(4, 2, figsize=(20, 24))

for idx, n_clust in enumerate(sorted(all_elbow_df['n_clusters_total'].unique())):
    if idx >= 4:
        break

    # Get data for this n_clusters
    plot_data = all_elbow_df[all_elbow_df['n_clusters_total'] == n_clust].copy()

    # Apply quality filters to get the filtered dataset
    quality_filtered = plot_data.copy()
    quality_filtered = quality_filtered[quality_filtered['n_latents'] >= 2].copy()
    quality_filtered = quality_filtered[quality_filtered['recon_is_monotonic'] == True].copy()
    quality_filtered = quality_filtered[quality_filtered['arch_is_monotonic'] == True].copy()
    quality_filtered = quality_filtered[quality_filtered['recon_pct_decrease'] >= 20].copy()
    quality_filtered = quality_filtered[quality_filtered['arch_pct_decrease'] >= 20].copy()
    quality_filtered = quality_filtered[quality_filtered['k_differential'].abs() <= 1].copy()

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

    # Mark selected vs rejected
    selected_set = all_selected[n_clust]
    plot_data_filtered['is_selected'] = plot_data_filtered['cluster_id'].isin(selected_set)

    # Plot rejected first (gray)
    rejected = plot_data_filtered[~plot_data_filtered['is_selected']]
    ax_right.scatter(
        rejected['aanet_recon_loss_elbow_strength'],
        rejected['aanet_archetypal_loss_elbow_strength'],
        c='lightgray',
        s=30,
        alpha=0.3,
        label=f'Rejected ({len(rejected)})'
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
    group = group[group['k_differential'].abs() <= 1].copy()

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
