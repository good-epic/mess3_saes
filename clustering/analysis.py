"""Cluster analysis: PCA, R², reconstructions."""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from mess3_gmg_analysis_utils import (
    collect_cluster_reconstructions,
    fit_pca_for_clusters,
    fit_residual_to_belief_map,
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
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        """Compute R² scores for belief prediction per cluster.

        Args:
            acts_flat: Flattened activations
            feature_acts_np: Feature activations (numpy)
            decoder_dirs: All decoder directions
            cluster_labels: Cluster labels for all latents
            n_clusters: Number of clusters
            component_beliefs_flat: Component beliefs
            component_order: Ordered component names
            ridge_alpha: Ridge regularization
            site: Site name for logging

        Returns:
            Dict mapping cluster_id -> component -> R² scores, or None on failure
        """
        if not component_beliefs_flat:
            return None

        try:
            acts_np = acts_flat.detach().cpu().numpy()
            beliefs_concat = np.concatenate(list(component_beliefs_flat.values()), axis=1)

            readout_coef, readout_intercept = fit_residual_to_belief_map(
                acts_np,
                beliefs_concat,
                alpha=ridge_alpha,
            )

            if decoder_dirs.shape[1] != readout_coef.shape[0]:
                raise ValueError(
                    f"Decoder width {decoder_dirs.shape[1]} must equal residual_to_belief rows {readout_coef.shape[0]}"
                )

            latent_sens_full = decoder_dirs @ readout_coef

            # Build component slices
            component_slices: Dict[str, Tuple[int, int]] = {}
            offset = 0
            for comp_name, comp_matrix in component_beliefs_flat.items():
                comp_dim = comp_matrix.shape[1]
                component_slices[comp_name] = (offset, offset + comp_dim)
                offset += comp_dim

            # Compute R² per cluster
            cluster_r2_summary = {}
            for cluster_id in range(int(n_clusters)):
                latent_ids = np.where(cluster_labels == cluster_id)[0]
                if latent_ids.size == 0:
                    continue

                cluster_entry: Dict[str, Any] = {}
                cluster_features = feature_acts_np[:, latent_ids]

                for comp_name in component_order:
                    start, end = component_slices[comp_name]
                    comp_targets = component_beliefs_flat[comp_name]
                    comp_targets_centered = comp_targets - readout_intercept[start:end]
                    cluster_sens = latent_sens_full[latent_ids][:, start:end]
                    preds = cluster_features @ cluster_sens
                    residual = comp_targets_centered - preds
                    ss_res = np.sum(residual ** 2, axis=0)
                    centered = comp_targets_centered - comp_targets_centered.mean(axis=0, keepdims=True)
                    ss_tot = np.sum(centered ** 2, axis=0)
                    denom = np.where(ss_tot == 0.0, 1.0, ss_tot)
                    r2 = 1.0 - ss_res / denom
                    cluster_entry[comp_name] = {
                        "per_dimension": r2.astype(float).tolist(),
                        "mean_r2": float(np.mean(r2)),
                    }

                if cluster_entry:
                    cluster_r2_summary[int(cluster_id)] = cluster_entry

            return cluster_r2_summary

        except Exception as exc:
            print(f"{site}: failed to compute belief R^2 diagnostics ({exc})")
            return None
