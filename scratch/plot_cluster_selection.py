#!/usr/bin/env python3
"""
Generate cluster selection visualization plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use('Agg')  # Use non-interactive backend

# Configuration
DATA_DIR = Path("/home/mattylev/projects/simplex/SAEs/mess3_sae/outputs/real_data_analysis_canonical")
OUTPUT_FILE = Path("/home/mattylev/projects/simplex/SAEs/mess3_sae/scratch/cluster_selection_viz.png")

# Load all elbow data
print("Loading data...")
all_elbow_df = pd.DataFrame()

for clusters_dir in sorted(DATA_DIR.glob("clusters_*")):
    n_str = clusters_dir.name.split("_")[1]
    n_clusters = int(n_str)

    csv_path = clusters_dir / f"consolidated_metrics_n{n_str}_corrected.csv"
    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)
    df['n_clusters_total'] = n_clusters
    all_elbow_df = pd.concat([all_elbow_df, df], ignore_index=True)

print(f"Loaded {len(all_elbow_df)} rows from {len(all_elbow_df['n_clusters_total'].unique())} different n_clusters values")

# Calculate elbow metrics
def calculate_elbow_score(x, y):
    if len(x) < 3:
        return None, 0.0
    x_norm = (np.array(x) - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
    y_norm = (np.array(y) - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else np.zeros_like(y)
    distances = np.abs(x_norm + y_norm - 1) / np.sqrt(2)
    elbow_idx = np.argmax(distances)
    elbow_k = x[elbow_idx]
    elbow_strength = distances[elbow_idx]
    return elbow_k, elbow_strength

print("\nCalculating elbow metrics...")
elbow_results = []

for (n_clust, cluster_id), group in all_elbow_df.groupby(['n_clusters_total', 'cluster_id']):
    group = group.sort_values('aanet_k')
    if len(group) < 3:
        continue

    k_values = group['aanet_k'].tolist()
    recon_elbow_k, recon_strength = calculate_elbow_score(k_values, group['aanet_recon_loss'].tolist())
    arch_elbow_k, arch_strength = calculate_elbow_score(k_values, group['aanet_archetypal_loss'].tolist())
    extrema_elbow_k, extrema_strength = calculate_elbow_score(k_values, group['aanet_extrema_loss'].tolist())

    # Recalculate k_differential from the recalculated elbow_k values
    k_diff = abs(recon_elbow_k - arch_elbow_k) if (recon_elbow_k and arch_elbow_k) else np.nan

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
        'recon_is_monotonic': group['recon_is_monotonic'].iloc[0],
        'arch_is_monotonic': group['arch_is_monotonic'].iloc[0],
        'recon_pct_decrease': group['recon_pct_decrease'].iloc[0],
        'arch_pct_decrease': group['arch_pct_decrease'].iloc[0],
        'k_differential': k_diff,
    })

all_elbow_df_processed = pd.DataFrame(elbow_results)
print(f"Calculated elbows for {len(all_elbow_df_processed)} clusters")

# Selection function
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
    """
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

    # Filter: Delta K constraint (using k_differential from CSV)
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
        (group['aanet_recon_loss_elbow_strength'] > recon_mean) &
        (group['aanet_archetypal_loss_elbow_strength'] > arch_mean)
    ].copy()
    category_stats['D_agreement'] = cat_d['cluster_id'].tolist()
    selected_clusters.update(cat_d['cluster_id'])

    return selected_clusters, category_stats

# Select promising clusters for each n_clusters value
print("\n" + "="*80)
print("CLUSTER SELECTION SUMMARY")
print("="*80)

all_selected = {}
all_category_stats = {}

for n_clust in sorted(all_elbow_df_processed['n_clusters_total'].unique()):
    selected_ids, cat_stats = select_promising_clusters(all_elbow_df_processed, n_clust)
    all_selected[n_clust] = selected_ids
    all_category_stats[n_clust] = cat_stats

    print(f"\nn_clusters={n_clust}:")
    print(f"  Total clusters (after Δk filter): {len(all_elbow_df_processed[(all_elbow_df_processed['n_clusters_total'] == n_clust) & (abs(all_elbow_df_processed['aanet_recon_loss_elbow_k'] - all_elbow_df_processed['aanet_archetypal_loss_elbow_k']) <= 1)])}")
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
selected_rows = all_elbow_df_processed[
    all_elbow_df_processed.apply(lambda row: row['cluster_id'] in all_selected.get(row['n_clusters_total'], set()), axis=1)
]
total_latents = selected_rows['n_latents'].sum()
print(f"\nEstimated latents to interpret: {total_latents:,}")
print(f"Estimated cost at $0.003/latent: ${total_latents * 0.003:.2f}")

# Create visualization
print("\nGenerating plots...")
fig, axes = plt.subplots(4, 2, figsize=(20, 24))

for idx, n_clust in enumerate(sorted(all_elbow_df_processed['n_clusters_total'].unique())):
    if idx >= 4:
        break

    # Get data for this n_clusters
    plot_data = all_elbow_df_processed[all_elbow_df_processed['n_clusters_total'] == n_clust].copy()
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

print(f"Saving plot to {OUTPUT_FILE}")
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')

# Create zoomed version
print("\nGenerating zoomed plots (excluding top 2 outliers)...")
fig, axes = plt.subplots(4, 2, figsize=(20, 24))

for idx, n_clust in enumerate(sorted(all_elbow_df_processed['n_clusters_total'].unique())):
    if idx >= 4:
        break

    # Get data for this n_clusters
    plot_data = all_elbow_df_processed[all_elbow_df_processed['n_clusters_total'] == n_clust].copy()

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
    xlim_max = recon_mean + 10 * recon_std
    ylim_max = arch_mean + 10 * arch_std

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

OUTPUT_FILE_ZOOMED = Path("/home/mattylev/projects/simplex/SAEs/mess3_sae/scratch/cluster_selection_viz_zoomed.png")
print(f"Saving zoomed plot to {OUTPUT_FILE_ZOOMED}")
plt.savefig(OUTPUT_FILE_ZOOMED, dpi=150, bbox_inches='tight')

# Print detailed information about selected latents
print("\n" + "="*100)
print("DETAILED SELECTED LATENT INFORMATION")
print("="*100)

for n_clust in sorted(all_elbow_df_processed['n_clusters_total'].unique()):
    print(f"\n{'='*100}")
    print(f"n_clusters = {n_clust}")
    print(f"{'='*100}")

    # Get data for this n_clusters
    group = all_elbow_df_processed[all_elbow_df_processed['n_clusters_total'] == n_clust].copy()

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
    print(f"\n{'Category':<10} {'Cluster ID':<12} {'N_Latents':<12} {'Recon Strength (SDs)':<25} {'Arch Strength (SDs)':<25}")
    print("-" * 84)

    # Print each selected cluster
    for _, row in selected_df.iterrows():
        cat = row['category']
        cluster_id = row['cluster_id']
        n_latents = row['n_latents']
        recon_strength = row['aanet_recon_loss_elbow_strength']
        recon_sds = row['recon_sds']
        arch_strength = row['aanet_archetypal_loss_elbow_strength']
        arch_sds = row['arch_sds']

        # Print main row
        print(f"{cat:<10} {cluster_id:<12} {n_latents:<12} {recon_strength:.4f} ({recon_sds:+.2f} SD){'':<8} {arch_strength:.4f} ({arch_sds:+.2f} SD)")

        # Print latent indices on next line if available
        if 'latent_indices' in row.index:
            latent_indices = row['latent_indices']
            print(f"{'':>10} Latent indices: {latent_indices}")
        print()  # Empty line between clusters

print("\n" + "="*100)
print("Done!")
