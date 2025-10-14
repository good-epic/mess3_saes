"""Cluster analysis: PCA, R², reconstructions."""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from sklearn.linear_model import Ridge

from mess3_gmg_analysis_utils import (
    collect_cluster_reconstructions,
    fit_pca_for_clusters,
    plot_cluster_pca,
    project_decoder_directions_to_pca,
)

from .config import AnalysisConfig, BeliefSeedingConfig


class ClusterAnalyzer:
    """Handles post-clustering analysis: PCA, R², reconstruction collection."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def collect_reconstructions(
        self,
        acts_flat: torch.Tensor,
        sae,
        cluster_labels: np.ndarray,
        cluster_activation_threshold: float,
        encoded_cache: Optional[Tuple] = None,
    ) -> Tuple[Dict, Dict]:
        """Collect cluster reconstructions.

        Args:
            acts_flat: Flattened activations
            sae: SAE model
            cluster_labels: Cluster labels for all latents
            cluster_activation_threshold: Minimum activation threshold
            encoded_cache: Optional pre-encoded features

        Returns:
            Tuple of (cluster_recons, cluster_stats)
        """
        with torch.no_grad():
            cluster_recons, cluster_stats = collect_cluster_reconstructions(
                acts_flat,
                sae,
                cluster_labels.tolist(),
                min_activation=cluster_activation_threshold,
                encoded_cache=encoded_cache,
            )
        return cluster_recons, cluster_stats

    def add_activity_rates_to_stats(
        self,
        cluster_stats: Dict[int, Dict],
        activity_rates: np.ndarray,
        mean_abs_activation: np.ndarray,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Add activity rate information to cluster stats.

        Args:
            cluster_stats: Cluster statistics
            activity_rates: Activity rates for all latents
            mean_abs_activation: Mean absolute activation for all latents

        Returns:
            Dict mapping site -> cluster_id -> rates array
        """
        per_site_cluster_rates = {}

        for cid, stats in cluster_stats.items():
            latent_ids = stats.get("latent_indices", [])
            stats["activity_rates"] = {int(idx): float(activity_rates[idx]) for idx in latent_ids}
            stats["mean_abs_activation"] = {int(idx): float(mean_abs_activation[idx]) for idx in latent_ids}

            rates_array = np.array([activity_rates[idx] for idx in latent_ids], dtype=float)
            if rates_array.size > 0:
                per_site_cluster_rates[int(cid)] = rates_array

        return per_site_cluster_rates

    def fit_pca_and_project(
        self,
        cluster_recons: Dict,
        cluster_stats: Dict,
        sae,
        min_cluster_samples: int,
    ) -> Optional[Dict[int, Any]]:
        """Fit PCA for clusters and project decoder directions.

        Note: Plotting is deferred to save_to_directory() where we have the output path.

        Args:
            cluster_recons: Cluster reconstructions
            cluster_stats: Cluster statistics
            sae: SAE model
            min_cluster_samples: Minimum samples required per cluster

        Returns:
            PCA results dict, or None if skipped
        """
        if self.config.skip_pca_plots:
            return None

        pca_results = fit_pca_for_clusters(
            cluster_recons,
            n_components=self.config.pca_components,
            min_samples=min_cluster_samples,
        )

        project_decoder_directions_to_pca(sae, pca_results, cluster_stats)

        # Store metadata in cluster stats
        for cid, result in pca_results.items():
            stats = cluster_stats.get(cid)
            if stats is not None:
                stats["explained_variance_ratio"] = [float(x) for x in result.pca.explained_variance_ratio_]
                stats["decoder_scale_factor"] = float(result.scale_factor)
                stats["decoder_projections_selected_pcs"] = {
                    int(latent_idx): [float(x) for x in vec.tolist()]
                    for latent_idx, vec in result.decoder_coords.items()
                }

        return pca_results

    def generate_pca_plots(
        self,
        pca_results: Dict[int, Any],
        site: str,
        site_selected_k: int,
        site_dir: str,
        seed: int,
    ) -> None:
        """Generate PCA plots for clusters.

        Args:
            pca_results: PCA results from fit_pca_and_project
            site: Site name
            site_selected_k: Selected k for this site
            site_dir: Output directory
            seed: Random seed
        """
        for cid, result in pca_results.items():
            plot_path = os.path.join(site_dir, f"cluster_{cid}_pca.png")
            try:
                plot_cluster_pca(
                    site,
                    site_selected_k,
                    cid,
                    result,
                    plot_path,
                    max_points=self.config.plot_max_points,
                    random_state=seed,
                )
            except ValueError as exc:
                print(f"Skipping PCA plot for {site} cluster {cid}: {exc}")

    def compute_belief_r2(
        self,
        acts_flat: torch.Tensor,
        feature_acts_np: np.ndarray,
        decoder_dirs: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        component_beliefs_flat: Dict[str, np.ndarray],
        component_order: List[str],
        ridge_alpha: float,
        site: str,
        soft_assignments: Optional[np.ndarray] = None,
        assignment_name: str = "hard",
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        """Compute R² scores for belief prediction per cluster.

        Args:
            acts_flat: Flattened activations
            feature_acts_np: Feature activations (numpy)
            decoder_dirs: All decoder directions
            cluster_labels: Cluster labels for all latents (used if soft_assignments is None)
            n_clusters: Number of clusters
            component_beliefs_flat: Component beliefs
            component_order: Ordered component names
            ridge_alpha: Ridge regularization
            site: Site name for logging
            soft_assignments: Optional (n_latents, n_clusters) soft assignment matrix
            assignment_name: Name for logging (e.g., "hard", "soft", "refined")

        Returns:
            Dict mapping cluster_id -> component -> R² scores, or None on failure
        """
        if not component_beliefs_flat:
            return None

        try:
            def _prepare_targets(comp_name: str, matrix: np.ndarray) -> Tuple[np.ndarray, int]:
                if comp_name.startswith("tom_quantum"):
                    if matrix.shape[1] <= 1:
                        return np.empty((matrix.shape[0], 0), dtype=matrix.dtype), 1
                    return matrix[:, 1:], 1
                return matrix, 0

            cluster_r2_summary = {}
            for cluster_id in range(int(n_clusters)):
                # Get cluster members and their weights
                if soft_assignments is not None:
                    # Use soft assignment weights
                    soft_weights = soft_assignments[:, cluster_id]
                    latent_ids = np.where(soft_weights > 0)[0]
                    if latent_ids.size == 0:
                        continue
                    weights = soft_weights[latent_ids]
                else:
                    # Use hard labels
                    latent_ids = np.where(cluster_labels == cluster_id)[0]
                    if latent_ids.size == 0:
                        continue
                    weights = None

                cluster_entry: Dict[str, Any] = {}

                # Get cluster features (potentially weighted)
                if weights is not None:
                    # Soft assignment: weight each latent's features
                    cluster_features = feature_acts_np[:, latent_ids] * weights[None, :]
                else:
                    # Hard assignment: unweighted features
                    cluster_features = feature_acts_np[:, latent_ids]

                # Restrict scoring to samples where the cluster actually activates.
                active_mask = np.any(cluster_features != 0.0, axis=1)
                if not np.any(active_mask):
                    continue

                cluster_features_active = cluster_features[active_mask]
                if cluster_features_active.shape[0] < 2:
                    continue  # Not enough data to fit a regression

                for comp_name in component_order:
                    comp_targets_raw = component_beliefs_flat[comp_name]
                    comp_targets_prepped, dropped_dims = _prepare_targets(comp_name, comp_targets_raw)

                    if comp_targets_prepped.shape[1] == 0:
                        continue

                    comp_targets_active = comp_targets_prepped[active_mask]

                    try:
                        ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
                        ridge.fit(cluster_features_active, comp_targets_active)
                        preds_refit = ridge.predict(cluster_features_active)
                    except Exception as exc_refit:
                        print(f"{site}: ridge refit failed for cluster {cluster_id} component {comp_name} ({exc_refit})")
                        continue

                    target_mean = comp_targets_active.mean(axis=0, keepdims=True)
                    ss_res = np.sum((comp_targets_active - preds_refit) ** 2, axis=0)
                    ss_tot = np.sum((comp_targets_active - target_mean) ** 2, axis=0)
                    denom = np.where(ss_tot == 0.0, 1.0, ss_tot)
                    r2 = 1.0 - ss_res / denom

                    cluster_entry[comp_name] = {
                        "per_dimension": r2.astype(float).tolist(),
                        "mean_r2": float(np.mean(r2)),
                        "n_active_samples": int(active_mask.sum()),
                        "dropped_constant_dims": int(dropped_dims),
                    }

                if cluster_entry:
                    cluster_r2_summary[int(cluster_id)] = cluster_entry

            return cluster_r2_summary

        except Exception as exc:
            print(f"{site}: failed to compute belief R^2 diagnostics ({assignment_name}) ({exc})")
            return None

    def compute_optimal_component_assignment(
        self,
        belief_r2_summary: Dict[int, Dict[str, Any]],
        component_order: List[str],
        n_clusters: int,
    ) -> Optional[Dict[str, Any]]:
        """Find optimal 1-to-1 assignment of components to clusters using Hungarian algorithm.

        Args:
            belief_r2_summary: Dict mapping cluster_id -> component -> R² scores
            component_order: Ordered list of component names
            n_clusters: Number of clusters

        Returns:
            Dict with assignment details:
                - r2_matrix: Full R² matrix (n_clusters × n_components)
                - assignments: {comp_name: cluster_id} for optimal mapping
                - assignment_scores: {comp_name: r2_score} for assigned pairs
                - noise_clusters: List of unassigned cluster IDs
                - total_r2: Sum of assigned R² scores
                - mean_assigned_r2: Mean R² for assigned pairs
        """
        if not belief_r2_summary or not component_order:
            return None

        from scipy.optimize import linear_sum_assignment

        n_components = len(component_order)

        # Build R² matrix (n_clusters × n_components)
        r2_matrix = np.zeros((n_clusters, n_components), dtype=float)

        for cluster_id in range(n_clusters):
            cluster_r2 = belief_r2_summary.get(cluster_id, {})
            for comp_idx, comp_name in enumerate(component_order):
                comp_r2 = cluster_r2.get(comp_name, {})
                r2_matrix[cluster_id, comp_idx] = comp_r2.get("mean_r2", 0.0)

        # Solve assignment problem (maximize R²)
        # linear_sum_assignment minimizes, so negate the matrix
        row_indices, col_indices = linear_sum_assignment(-r2_matrix)

        # Build assignment mappings
        assignments = {}
        assignment_scores = {}
        assigned_clusters = set()

        for row_idx, col_idx in zip(row_indices, col_indices):
            comp_name = component_order[col_idx]
            r2_score = r2_matrix[row_idx, col_idx]
            assignments[comp_name] = int(row_idx)
            assignment_scores[comp_name] = float(r2_score)
            assigned_clusters.add(int(row_idx))

        # Identify noise clusters (unassigned)
        all_clusters = set(range(n_clusters))
        noise_clusters = sorted(list(all_clusters - assigned_clusters))

        # Aggregate metrics
        total_r2 = sum(assignment_scores.values())
        mean_assigned_r2 = total_r2 / len(assignment_scores) if assignment_scores else 0.0

        return {
            "r2_matrix": r2_matrix.tolist(),
            "assignments": assignments,
            "assignment_scores": assignment_scores,
            "noise_clusters": noise_clusters,
            "total_r2": float(total_r2),
            "mean_assigned_r2": float(mean_assigned_r2),
        }

    def compute_activation_coherence_metrics(
        self,
        feature_acts: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        soft_assignments: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute activation coherence metrics for clusters.

        Measures how coherently latents within clusters co-activate, and how
        independent different clusters are.

        Args:
            feature_acts: Feature activations (n_samples, n_latents)
            cluster_labels: Hard cluster labels (n_latents,)
            n_clusters: Number of clusters
            soft_assignments: Optional soft assignment matrix (n_latents, n_clusters)

        Returns:
            Dict with coherence metrics:
                - within_cluster_correlation_{cluster_id}: Mean pairwise correlation within cluster
                - within_cluster_correlation_mean: Average across all clusters
                - between_cluster_correlation_mean: Mean correlation between cluster activations
                - cluster_{id}_activation_sparsity: Fraction of samples activating cluster
        """
        metrics = {}

        # Use soft or hard assignments
        if soft_assignments is not None:
            assignment_matrix = soft_assignments
        else:
            # Convert hard labels to one-hot matrix
            assignment_matrix = np.zeros((len(cluster_labels), n_clusters), dtype=float)
            for i, label in enumerate(cluster_labels):
                if label >= 0:  # Skip noise (-1)
                    assignment_matrix[i, label] = 1.0

        # 1. Within-cluster correlation
        within_cluster_corrs = []
        for cluster_id in range(n_clusters):
            # Get latents in this cluster
            cluster_weights = assignment_matrix[:, cluster_id]
            cluster_members = np.where(cluster_weights > 0)[0]

            if len(cluster_members) < 2:
                continue

            # Get activations for these latents
            cluster_acts = feature_acts[:, cluster_members]

            # Compute pairwise correlations
            try:
                corr_matrix = np.corrcoef(cluster_acts.T)
                # Get upper triangle (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                pairwise_corrs = corr_matrix[upper_tri_indices]
                mean_corr = float(np.mean(pairwise_corrs))
                metrics[f"within_cluster_correlation_{cluster_id}"] = mean_corr
                within_cluster_corrs.append(mean_corr)
            except Exception:
                # Skip if correlation fails (e.g., constant activations)
                continue

        if within_cluster_corrs:
            metrics["within_cluster_correlation_mean"] = float(np.mean(within_cluster_corrs))

        # 2. Between-cluster independence (correlation of cluster-level activations)
        # Compute cluster activation signals: weighted sum of latent activations
        cluster_activation_signals = []
        for cluster_id in range(n_clusters):
            cluster_weights = assignment_matrix[:, cluster_id]
            if cluster_weights.sum() > 0:
                # Weighted sum: samples × latents @ weights = samples
                cluster_signal = (feature_acts * cluster_weights[None, :]).sum(axis=1)
                cluster_activation_signals.append(cluster_signal)

        if len(cluster_activation_signals) >= 2:
            cluster_signals_matrix = np.stack(cluster_activation_signals, axis=1)
            try:
                between_corr_matrix = np.corrcoef(cluster_signals_matrix.T)
                # Get off-diagonal elements
                np.fill_diagonal(between_corr_matrix, 0)
                n_pairs = n_clusters * (n_clusters - 1) / 2
                if n_pairs > 0:
                    between_corr_mean = float(np.sum(np.abs(between_corr_matrix)) / (2 * n_pairs))
                    metrics["between_cluster_correlation_mean"] = between_corr_mean
            except Exception:
                pass

        # 3. Per-cluster activation statistics
        for cluster_id in range(n_clusters):
            cluster_weights = assignment_matrix[:, cluster_id]
            cluster_members = np.where(cluster_weights > 0)[0]

            if len(cluster_members) == 0:
                continue

            # Weighted cluster activations per sample
            cluster_acts_per_sample = (feature_acts[:, cluster_members] * cluster_weights[cluster_members][None, :]).sum(axis=1)

            # Sparsity: fraction of samples where cluster activates
            activation_threshold = 1e-6
            active_samples = cluster_acts_per_sample > activation_threshold
            sparsity = float(active_samples.mean())
            metrics[f"cluster_{cluster_id}_activation_sparsity"] = sparsity

        return metrics
