"""
Shared cluster selection utilities.

This module contains functions used by multiple scripts for selecting
promising clusters based on AANet elbow analysis.
"""

import numpy as np


def calculate_elbow_score(x, y):
    """
    Calculate elbow scores using perpendicular distance from line.
    Returns the k value at the elbow and the elbow strength.

    Based on the Kneedle algorithm idea: find point of maximum distance
    from the line connecting first and last points.

    Args:
        x: Array of k values (sorted ascending)
        y: Array of loss values corresponding to each k

    Returns:
        (elbow_k, elbow_strength): The k value at elbow and the strength score
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


def select_promising_clusters(df, n_clusters_val, delta_k_threshold=1,
                              sd_outlier=3, sd_strong=1, verbose=False):
    """
    Select promising clusters based on multi-criteria approach.

    Takes ALL clusters that meet the criteria (no arbitrary top-N limits).

    All categories use standard deviation cutoffs:
    - Categories B & C (outliers): mean + sd_outlier*SD (default 3)
    - Categories A & D (strong): mean + sd_strong*SD (default 1)

    Quality filters applied:
    - n_latents >= 2 (no single-latent clusters)
    - recon_is_monotonic == True (monotonic decrease to elbow)
    - arch_is_monotonic == True (monotonic decrease to elbow)
    - recon_pct_decrease >= 20 (at least 20% decrease from max to min)
    - arch_pct_decrease >= 20 (at least 20% decrease from max to min)
    - k_differential <= delta_k_threshold (elbow agreement)

    Args:
        df: DataFrame with cluster metrics
        n_clusters_val: Which n_clusters value to filter for
        delta_k_threshold: Maximum allowed |k_differential| for quality filter
        sd_outlier: SD multiplier for outlier categories (B, C)
        sd_strong: SD multiplier for strong categories (A, D)
        verbose: If True, print filter statistics at each step

    Returns:
        (selected_clusters, category_stats): Set of cluster_ids and dict of category lists
    """
    # Filter for this n_clusters value
    group = df[df['n_clusters_total'] == n_clusters_val].copy()

    def print_filter_stats(group, label):
        if not verbose:
            return
        recon_mean = group['aanet_recon_loss_elbow_strength'].mean()
        recon_std = group['aanet_recon_loss_elbow_strength'].std()
        arch_mean = group['aanet_archetypal_loss_elbow_strength'].mean()
        arch_std = group['aanet_archetypal_loss_elbow_strength'].std()
        print(f"  {label}: {len(group)} clusters | "
              f"recon μ={recon_mean:.4f} σ={recon_std:.4f} | "
              f"arch μ={arch_mean:.4f} σ={arch_std:.4f}")

    # Apply quality filters
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
    # Both above mean + sd_strong * SD
    cat_a = group[
        (group['aanet_recon_loss_elbow_strength'] > recon_strong_threshold) &
        (group['aanet_archetypal_loss_elbow_strength'] > arch_strong_threshold)
    ].copy()
    category_stats['A_strong_both'] = cat_a['cluster_id'].tolist()
    selected_clusters.update(cat_a['cluster_id'])

    # Category B: Reconstruction Outliers (mean + sd_outlier * SD)
    cat_b = group[
        group['aanet_recon_loss_elbow_strength'] > recon_outlier_threshold
    ].copy()
    category_stats['B_recon_outliers'] = cat_b['cluster_id'].tolist()
    selected_clusters.update(cat_b['cluster_id'])

    # Category C: Archetypal Outliers (mean + sd_outlier * SD)
    cat_c = group[
        group['aanet_archetypal_loss_elbow_strength'] > arch_outlier_threshold
    ].copy()
    category_stats['C_arch_outliers'] = cat_c['cluster_id'].tolist()
    selected_clusters.update(cat_c['cluster_id'])

    # Category D: Perfect Agreement Standouts
    # Delta k = 0 AND both metrics above their means
    cat_d = group[
        (group['k_differential'] == 0) &
        (((group['aanet_recon_loss_elbow_strength'] > recon_mean) &
          (group['aanet_archetypal_loss_elbow_strength'] > arch_mean)) |
         ((group['aanet_archetypal_loss_elbow_strength'] > arch_mean) &
          (group['aanet_recon_loss_elbow_strength'] > recon_mean)))
    ].copy()
    category_stats['D_agreement'] = cat_d['cluster_id'].tolist()
    selected_clusters.update(cat_d['cluster_id'])

    return selected_clusters, category_stats
