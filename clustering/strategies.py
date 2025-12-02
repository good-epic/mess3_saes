"""Clustering strategy implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# from training_and_analysis_utils import build_similarity_matrix, spectral_clustering_with_eigengap
from subspace_clustering_utils import (
    normalize_and_deduplicate,
    k_subspaces_clustering,
    ensc_clustering,
    grid_search_k_subspaces,
    add_diagnostics_to_result,
)

from .config import ClusteringConfig, SpectralParams, SubspaceParams, ENSCParams


@dataclass
class ClusteringStrategyResult:
    """Result from a clustering strategy."""
    cluster_labels: np.ndarray  # Labels for active latents only
    n_clusters: int
    diagnostics: Dict[str, Any]

    # Soft assignment weights
    soft_weights: Optional[np.ndarray] = None  # (n_latents, n_clusters) soft assignments

    # Subspace-specific fields
    decoder_normalized: Optional[np.ndarray] = None
    normalized_to_full_idx: Optional[np.ndarray] = None
    kept_indices: Optional[np.ndarray] = None  # Indices in active space


class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""

    @abstractmethod
    def cluster(
        self,
        decoder_active: np.ndarray,
        active_indices: np.ndarray,
        config: ClusteringConfig,
        site: str,
        site_dir: str,
        latent_activity_matrix: Optional[np.ndarray] = None,
        belief_seed_clusters: Optional[Dict[int, List[int]]] = None,
        component_order: Optional[List[str]] = None,
    ) -> ClusteringStrategyResult:
        """Run clustering on active decoder directions.

        Args:
            decoder_active: Active decoder directions (n_active, d_model)
            active_indices: Indices of active latents in full decoder
            config: Clustering configuration
            site: Site name
            site_dir: Directory for saving outputs
            latent_activity_matrix: Activity matrix for phi similarity (spectral only)
            belief_seed_clusters: Seed clusters in active index space (subspace only)
            component_order: Ordered component names for belief seeding

        Returns:
            ClusteringStrategyResult with labels, diagnostics, etc.
        """
        pass


class SpectralClusteringStrategy(ClusteringStrategy):
    """Spectral clustering with eigengap selection."""

    def cluster(
        self,
        decoder_active: np.ndarray,
        active_indices: np.ndarray,
        config: ClusteringConfig,
        site: str,
        site_dir: str,
        latent_activity_matrix: Optional[np.ndarray] = None,
        belief_seed_clusters: Optional[Dict[int, List[int]]] = None,
        component_order: Optional[List[str]] = None,
    ) -> ClusteringStrategyResult:
        """Run spectral clustering."""
        import os

        params: SpectralParams = config.spectral_params
        decoder_for_clustering = decoder_active.copy()

        # Center decoder rows if requested
        if params.center_decoder_rows and decoder_active.size > 0:
            row_mean = decoder_for_clustering.mean(axis=0, keepdims=True)
            decoder_for_clustering = decoder_for_clustering - row_mean
            norms = np.linalg.norm(decoder_for_clustering, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            decoder_for_clustering = decoder_for_clustering / norms

        # Build similarity matrix
        if params.sim_metric == "phi":
            if latent_activity_matrix is None:
                raise ValueError("Phi similarity requires latent activation matrix")
            latent_active = latent_activity_matrix[:, active_indices]
            sim_matrix = build_similarity_matrix(
                decoder_for_clustering,
                method="phi",
                latent_acts=latent_active,
            )
            sim_matrix = (sim_matrix + 1.0) / 2.0
            np.fill_diagonal(sim_matrix, 1.0)
        else:
            sim_matrix = build_similarity_matrix(decoder_for_clustering, method=params.sim_metric)

        # Run spectral clustering
        eig_plot = None
        if params.plot_eigengap:
            eig_plot = os.path.join(site_dir, "eigengap.png")

        from training_and_analysis_utils import build_similarity_matrix, spectral_clustering_with_eigengap
        cluster_labels, n_clusters = spectral_clustering_with_eigengap(
            sim_matrix,
            max_clusters=min(params.max_clusters, decoder_active.shape[0]),
            min_clusters=params.min_clusters,
            random_state=config.seed,
            plot=params.plot_eigengap,
            plot_path=eig_plot,
        )
        cluster_labels = np.asarray(cluster_labels, dtype=int)

        # Compute soft weights from eigenvector distances to cluster centers
        soft_weights = self._compute_spectral_soft_weights(
            sim_matrix, cluster_labels, n_clusters, config.seed
        )

        diagnostics = {
            "clustering_method": "spectral",
            "decoder_rows_centered": bool(params.center_decoder_rows),
            "sim_metric": params.sim_metric,
        }

        return ClusteringStrategyResult(
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            diagnostics=diagnostics,
            soft_weights=soft_weights,
        )

    def _compute_spectral_soft_weights(
        self,
        sim_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        seed: int
    ) -> np.ndarray:
        """Compute soft assignment weights from spectral embedding.

        Args:
            sim_matrix: Similarity matrix
            cluster_labels: Hard cluster labels
            n_clusters: Number of clusters
            seed: Random seed

        Returns:
            Soft weights (n_points, n_clusters)
        """
        from sklearn.cluster import KMeans

        # Reproduce spectral embedding
        diag = np.diag(sim_matrix.sum(axis=1))
        laplacian = diag - sim_matrix
        sqrt_deg = np.diag(1.0 / np.sqrt(np.maximum(diag.diagonal(), 1e-12)))
        norm_lap = sqrt_deg @ laplacian @ sqrt_deg

        # Get eigenvectors
        eigvals, eigvecs = np.linalg.eigh(norm_lap)
        eigvals = np.real(eigvals)
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        # Use first n_clusters eigenvectors
        embedding = eigvecs[:, :n_clusters]

        # Fit k-means to get cluster centers
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        kmeans.fit(embedding)

        # Compute distances to each cluster center
        distances = kmeans.transform(embedding)  # (n_points, n_clusters)

        # Convert distances to soft weights via softmax
        # Use negative distances so closer = higher weight
        soft_weights = np.exp(-distances)
        soft_weights = soft_weights / (soft_weights.sum(axis=1, keepdims=True) + 1e-10)

        return soft_weights


class KSubspacesClusteringStrategy(ClusteringStrategy):
    """K-subspaces clustering with optional belief seeding."""

    def cluster(
        self,
        decoder_active: np.ndarray,
        active_indices: np.ndarray,
        config: ClusteringConfig,
        site: str,
        site_dir: str,
        latent_activity_matrix: Optional[np.ndarray] = None,
        belief_seed_clusters: Optional[Dict[int, List[int]]] = None,
        component_order: Optional[List[str]] = None,
    ) -> ClusteringStrategyResult:
        """Run k-subspaces clustering."""
        params: SubspaceParams = config.subspace_params

        # decoder_active is already normalized and deduplicated by the pipeline
        decoder_normalized = decoder_active
        kept_indices = np.arange(len(decoder_active))  # All indices kept (already deduped)
        normalized_to_full_idx = active_indices

        # Belief seed clusters are already in correct space (no remapping needed)
        initial_clusters_normalized: Dict[int, List[int]] = {}
        if belief_seed_clusters:
            # Clusters are already using indices in the deduplicated active space
            initial_clusters_normalized = belief_seed_clusters

        # Handle trivial cases
        if len(decoder_normalized) == 0:
            return ClusteringStrategyResult(
                cluster_labels=np.array([], dtype=int),
                n_clusters=0,
                diagnostics={"clustering_method": "k_subspaces"},
                decoder_normalized=decoder_normalized,
                normalized_to_full_idx=normalized_to_full_idx,
                kept_indices=kept_indices,
            )
        elif len(decoder_normalized) == 1:
            cluster_labels_active = np.array([0], dtype=int)
            return ClusteringStrategyResult(
                cluster_labels=cluster_labels_active,
                n_clusters=1,
                diagnostics={"clustering_method": "k_subspaces"},
                decoder_normalized=decoder_normalized,
                normalized_to_full_idx=normalized_to_full_idx,
                kept_indices=kept_indices,
            )

        # Determine number of clusters
        effective_n_clusters = params.n_clusters
        override_reason = None

        # Only override auto-detection (None), not explicit user specification
        if params.n_clusters is None and initial_clusters_normalized and component_order:
            num_seed_clusters = len(initial_clusters_normalized)
            seeded_latent_total = sum(len(v) for v in initial_clusters_normalized.values())
            total_components = len(component_order)

            if total_components and num_seed_clusters == total_components:
                effective_n_clusters = num_seed_clusters
                override_reason = "all components seeded (auto)"
            elif (
                seeded_latent_total >= len(decoder_normalized)
                and total_components
                and num_seed_clusters < total_components
            ):
                effective_n_clusters = num_seed_clusters
                override_reason = "all latents covered by seeded components (auto)"

            if override_reason is not None:
                print(f"{site}: auto-detected k_subspaces cluster count to {effective_n_clusters} ({override_reason})")

        # Run k-subspaces
        if params.use_grid_search:
            if initial_clusters_normalized:
                raise ValueError("Belief-aligned seeding is not compatible with --use_grid_search")
            subspace_result, grid_results = grid_search_k_subspaces(
                decoder_normalized,
                k_values=params.k_values,
                r_values=params.r_values,
                random_state=config.seed,
                bic_penalty_weight=0.1,
            )
            print(
                f"{site}: grid search selected K={subspace_result.n_clusters}, "
                f"r={subspace_result.subspace_rank} "
                f"(total error={subspace_result.total_reconstruction_error:.4f})"
            )
        else:
            subspace_result = k_subspaces_clustering(
                decoder_normalized,
                n_clusters=effective_n_clusters,
                subspace_rank=params.subspace_rank,
                max_iters=20,
                random_state=config.seed,
                initial_clusters=initial_clusters_normalized if initial_clusters_normalized else None,
                lock_mode=params.seed_lock_mode,
                variance_threshold=params.variance_threshold,
                gap_threshold=params.gap_threshold,
            )

            if params.n_clusters is None or params.subspace_rank is None:
                auto_info = []
                if params.n_clusters is None:
                    auto_info.append(f"K={subspace_result.n_clusters} (auto)")
                else:
                    auto_info.append(f"K={subspace_result.n_clusters}")
                if params.subspace_rank is None and subspace_result.cluster_ranks:
                    ranks_str = ",".join(str(subspace_result.cluster_ranks[i]) for i in sorted(subspace_result.cluster_ranks.keys()))
                    auto_info.append(f"r=[{ranks_str}] (auto per-cluster)")
                elif params.subspace_rank is not None:
                    auto_info.append(f"r={subspace_result.subspace_rank}")
                print(f"{site}: {', '.join(auto_info)}, error={subspace_result.total_reconstruction_error:.4f}")

                # Print rank estimation details if available
                if params.subspace_rank is None and subspace_result.rank_estimation_details:
                    for cluster_id in sorted(subspace_result.rank_estimation_details.keys()):
                        details = subspace_result.rank_estimation_details[cluster_id]
                        print(f"  Cluster {cluster_id}: variance→r={details.variance_rank}, "
                              f"gap→r={details.gap_rank}, final→r={details.final_rank} "
                              f"(limited by {details.limiting_method})")

        # Add diagnostics
        subspace_result = add_diagnostics_to_result(subspace_result, decoder_normalized)

        # decoder_active is already deduplicated, so labels map directly
        cluster_labels_active = subspace_result.cluster_labels

        # Build diagnostics
        diagnostics = self._build_diagnostics(subspace_result, params, config)

        # Print diagnostics
        if subspace_result.principal_angles is not None:
            min_angles_deg = {
                pair: float(np.rad2deg(angles.min()))
                for pair, angles in subspace_result.principal_angles.items()
            }
            overall_min = min(min_angles_deg.values()) if min_angles_deg else 0.0
            print(f"{site}: minimum principal angle = {overall_min:.1f}°")

        if subspace_result.within_projection_energy is not None and subspace_result.between_projection_energy is not None:
            mean_within = np.mean(list(subspace_result.within_projection_energy.values()))
            ratio = mean_within / subspace_result.between_projection_energy if subspace_result.between_projection_energy > 0 else float('inf')
            print(f"{site}: within/between energy ratio = {ratio:.2f}")

        # Compute soft weights from reconstruction errors
        soft_weights = self._compute_subspace_soft_weights(
            decoder_active, decoder_normalized, kept_indices,
            subspace_result, cluster_labels_active
        )

        return ClusteringStrategyResult(
            cluster_labels=cluster_labels_active,
            n_clusters=subspace_result.n_clusters,
            diagnostics=diagnostics,
            soft_weights=soft_weights,
            decoder_normalized=decoder_normalized,
            normalized_to_full_idx=normalized_to_full_idx,
            kept_indices=kept_indices,
        )

    def _compute_subspace_soft_weights(
        self,
        decoder_active: np.ndarray,
        decoder_normalized: np.ndarray,
        kept_indices: np.ndarray,
        subspace_result,
        cluster_labels: np.ndarray
    ) -> np.ndarray:
        """Compute soft weights from subspace reconstruction errors.

        Lower reconstruction error = higher affinity to that cluster.

        Args:
            decoder_active: Full active decoder
            decoder_normalized: Deduplicated decoder
            kept_indices: Indices of kept points
            subspace_result: Subspace clustering result with bases
            cluster_labels: Hard cluster assignments

        Returns:
            Soft weights (n_active, n_clusters)
        """
        n_active = len(decoder_active)
        n_clusters = subspace_result.n_clusters

        soft_weights = np.zeros((n_active, n_clusters))

        # Get subspace bases
        if not hasattr(subspace_result, 'bases') or subspace_result.bases is None:
            # Fall back to hard assignment
            for i in range(n_active):
                soft_weights[i, cluster_labels[i]] = 1.0
            return soft_weights

        bases = subspace_result.bases

        # Compute reconstruction error for each point in each subspace
        reconstruction_errors = np.zeros((n_active, n_clusters))

        for cluster_id, basis in bases.items():
            # Project all points onto this subspace
            if basis.shape[1] == 0:
                # Empty basis, high error
                reconstruction_errors[:, cluster_id] = 1e10
                continue

            projections = decoder_active @ basis @ basis.T
            errors = np.linalg.norm(decoder_active - projections, axis=1)
            reconstruction_errors[:, cluster_id] = errors

        # Convert errors to soft weights via softmax
        # Use negative errors so lower error = higher weight
        soft_weights = np.exp(-reconstruction_errors)
        soft_weights = soft_weights / (soft_weights.sum(axis=1, keepdims=True) + 1e-10)

        return soft_weights

    def _build_diagnostics(self, subspace_result, params: SubspaceParams, config: ClusteringConfig) -> Dict[str, Any]:
        """Build diagnostics dictionary from subspace result."""
        diagnostics = {
            "clustering_method": "k_subspaces",
            "subspace_rank": int(subspace_result.subspace_rank) if subspace_result.subspace_rank is not None else None,
            "subspace_reconstruction_error": float(subspace_result.total_reconstruction_error),
            "n_clusters_auto": params.n_clusters is None and not params.use_grid_search,
            "subspace_rank_auto": params.subspace_rank is None and not params.use_grid_search,
            "used_grid_search": params.use_grid_search,
        }

        if subspace_result.cluster_ranks is not None:
            diagnostics["cluster_ranks"] = {
                int(k): int(v) for k, v in subspace_result.cluster_ranks.items()
            }

        if subspace_result.rank_estimation_details is not None:
            diagnostics["rank_estimation_details"] = {
                int(k): {
                    "variance_rank": v.variance_rank,
                    "gap_rank": v.gap_rank,
                    "final_rank": v.final_rank,
                    "limiting_method": v.limiting_method,
                }
                for k, v in subspace_result.rank_estimation_details.items()
            }
            diagnostics["variance_threshold"] = params.variance_threshold
            diagnostics["gap_threshold"] = params.gap_threshold

        if subspace_result.principal_angles is not None:
            principal_angles_deg = {}
            min_principal_angles = {}
            for (ci, cj), angles in subspace_result.principal_angles.items():
                key = f"({ci},{cj})"
                principal_angles_deg[key] = [float(np.rad2deg(a)) for a in angles]
                min_principal_angles[key] = float(np.rad2deg(angles.min()))

            diagnostics["principal_angles_deg"] = principal_angles_deg
            diagnostics["min_principal_angles_deg"] = min_principal_angles
            diagnostics["overall_min_principal_angle_deg"] = float(min(min_principal_angles.values())) if min_principal_angles else None

        if subspace_result.within_projection_energy is not None:
            diagnostics["within_projection_energy"] = {
                int(k): float(v) for k, v in subspace_result.within_projection_energy.items()
            }

        if subspace_result.between_projection_energy is not None:
            diagnostics["between_projection_energy"] = float(subspace_result.between_projection_energy)

            if subspace_result.within_projection_energy is not None:
                mean_within = np.mean(list(subspace_result.within_projection_energy.values()))
                ratio = mean_within / subspace_result.between_projection_energy if subspace_result.between_projection_energy > 0 else float('inf')
                diagnostics["energy_contrast_ratio"] = float(ratio)

        return diagnostics


class ENSCClusteringStrategy(ClusteringStrategy):
    """Elastic Net Subspace Clustering."""

    def cluster(
        self,
        decoder_active: np.ndarray,
        active_indices: np.ndarray,
        config: ClusteringConfig,
        site: str,
        site_dir: str,
        latent_activity_matrix: Optional[np.ndarray] = None,
        belief_seed_clusters: Optional[Dict[int, List[int]]] = None,
        component_order: Optional[List[str]] = None,
    ) -> ClusteringStrategyResult:
        """Run ENSC clustering."""
        params: ENSCParams = config.ensc_params

        # decoder_active is already normalized and deduplicated by the pipeline
        decoder_normalized = decoder_active
        kept_indices = np.arange(len(decoder_active))  # All indices kept (already deduped)
        normalized_to_full_idx = active_indices

        # Handle trivial cases
        if len(decoder_normalized) == 0:
            return ClusteringStrategyResult(
                cluster_labels=np.array([], dtype=int),
                n_clusters=0,
                diagnostics={"clustering_method": "ensc"},
                decoder_normalized=decoder_normalized,
                normalized_to_full_idx=normalized_to_full_idx,
                kept_indices=kept_indices,
            )
        elif len(decoder_normalized) == 1:
            cluster_labels_active = np.array([0], dtype=int)
            return ClusteringStrategyResult(
                cluster_labels=cluster_labels_active,
                n_clusters=1,
                diagnostics={"clustering_method": "ensc"},
                decoder_normalized=decoder_normalized,
                normalized_to_full_idx=normalized_to_full_idx,
                kept_indices=kept_indices,
            )

        # Run ENSC
        subspace_result = ensc_clustering(
            decoder_normalized,
            n_clusters=params.n_clusters,
            subspace_rank=params.subspace_rank,
            lambda_1=params.lambda1,
            lambda_2=params.lambda2,
            random_state=config.seed,
            variance_threshold=params.variance_threshold,
            gap_threshold=params.gap_threshold,
        )

        if params.n_clusters is None or params.subspace_rank is None:
            auto_info = []
            if params.n_clusters is None:
                auto_info.append(f"K={subspace_result.n_clusters} (auto eigengap)")
            else:
                auto_info.append(f"K={subspace_result.n_clusters}")
            if params.subspace_rank is None and subspace_result.cluster_ranks:
                ranks_str = ",".join(str(subspace_result.cluster_ranks[i]) for i in sorted(subspace_result.cluster_ranks.keys()))
                auto_info.append(f"r=[{ranks_str}] (auto per-cluster)")
            elif params.subspace_rank is not None:
                auto_info.append(f"r={subspace_result.subspace_rank}")
            print(f"{site}: {', '.join(auto_info)}, error={subspace_result.total_reconstruction_error:.4f}")

            # Print rank estimation details if available
            if params.subspace_rank is None and subspace_result.rank_estimation_details:
                for cluster_id in sorted(subspace_result.rank_estimation_details.keys()):
                    details = subspace_result.rank_estimation_details[cluster_id]
                    print(f"  Cluster {cluster_id}: variance→r={details.variance_rank}, "
                          f"gap→r={details.gap_rank}, final→r={details.final_rank} "
                          f"(limited by {details.limiting_method})")

        # Add diagnostics
        subspace_result = add_diagnostics_to_result(subspace_result, decoder_normalized)

        # decoder_active is already deduplicated, so labels map directly
        cluster_labels_active = subspace_result.cluster_labels

        # Build diagnostics (similar to k-subspaces)
        diagnostics = {
            "clustering_method": "ensc",
            "subspace_rank": int(subspace_result.subspace_rank) if subspace_result.subspace_rank is not None else None,
            "subspace_reconstruction_error": float(subspace_result.total_reconstruction_error),
            "n_clusters_auto": params.n_clusters is None,
            "subspace_rank_auto": params.subspace_rank is None,
        }

        if subspace_result.cluster_ranks is not None:
            diagnostics["cluster_ranks"] = {
                int(k): int(v) for k, v in subspace_result.cluster_ranks.items()
            }

        if subspace_result.rank_estimation_details is not None:
            diagnostics["rank_estimation_details"] = {
                int(k): {
                    "variance_rank": v.variance_rank,
                    "gap_rank": v.gap_rank,
                    "final_rank": v.final_rank,
                    "limiting_method": v.limiting_method,
                }
                for k, v in subspace_result.rank_estimation_details.items()
            }
            diagnostics["variance_threshold"] = params.variance_threshold
            diagnostics["gap_threshold"] = params.gap_threshold

        # Compute soft weights from self-representation matrix or reconstruction errors
        soft_weights = self._compute_ensc_soft_weights(
            decoder_active, decoder_normalized, kept_indices,
            subspace_result, cluster_labels_active
        )

        return ClusteringStrategyResult(
            cluster_labels=cluster_labels_active,
            n_clusters=subspace_result.n_clusters,
            diagnostics=diagnostics,
            soft_weights=soft_weights,
            decoder_normalized=decoder_normalized,
            normalized_to_full_idx=normalized_to_full_idx,
            kept_indices=kept_indices,
        )

    def _compute_ensc_soft_weights(
        self,
        decoder_active: np.ndarray,
        decoder_normalized: np.ndarray,
        kept_indices: np.ndarray,
        subspace_result,
        cluster_labels: np.ndarray
    ) -> np.ndarray:
        """Compute soft weights for ENSC clustering.

        Try to use self-representation matrix if available, otherwise
        fall back to reconstruction error-based approach.

        Args:
            decoder_active: Full active decoder
            decoder_normalized: Deduplicated decoder
            kept_indices: Indices of kept points
            subspace_result: ENSC result
            cluster_labels: Hard cluster assignments

        Returns:
            Soft weights (n_active, n_clusters)
        """
        n_active = len(decoder_active)
        n_clusters = subspace_result.n_clusters

        # Check if self-representation matrix is available
        if hasattr(subspace_result, 'self_representation_matrix') and subspace_result.self_representation_matrix is not None:
            # Use self-representation coefficients as soft affinities
            C = subspace_result.self_representation_matrix  # (n_normalized, n_normalized)

            # Build affinity within each cluster
            soft_weights = np.zeros((n_active, n_clusters))

            for cluster_id in range(n_clusters):
                cluster_mask_norm = subspace_result.cluster_labels == cluster_id

                # For each point (in active space), compute affinity to this cluster
                for i in range(n_active):
                    if i in kept_indices:
                        # This point is in the normalized set
                        norm_idx = np.where(kept_indices == i)[0][0]
                        # Sum of affinities to cluster members
                        affinity = np.abs(C[norm_idx, cluster_mask_norm]).sum()
                        soft_weights[i, cluster_id] = affinity
                    else:
                        # Duplicate point - use nearest kept point's affinity
                        dup_vec = decoder_active[i]
                        similarities = np.abs(np.dot(decoder_normalized, dup_vec))
                        nearest_kept_idx = kept_indices[np.argmax(similarities)]
                        soft_weights[i, :] = soft_weights[nearest_kept_idx, :]

            # Normalize to sum to 1
            soft_weights = soft_weights / (soft_weights.sum(axis=1, keepdims=True) + 1e-10)

        else:
            # Fall back to reconstruction error approach (same as k-subspaces)
            soft_weights = np.zeros((n_active, n_clusters))

            # Get subspace bases
            if not hasattr(subspace_result, 'bases') or subspace_result.bases is None:
                # Hard assignment fallback
                for i in range(n_active):
                    soft_weights[i, cluster_labels[i]] = 1.0
                return soft_weights

            bases = subspace_result.bases
            reconstruction_errors = np.zeros((n_active, n_clusters))

            for cluster_id, basis in bases.items():
                if basis.shape[1] == 0:
                    reconstruction_errors[:, cluster_id] = 1e10
                    continue

                projections = decoder_active @ basis @ basis.T
                errors = np.linalg.norm(decoder_active - projections, axis=1)
                reconstruction_errors[:, cluster_id] = errors

            # Convert to soft weights
            soft_weights = np.exp(-reconstruction_errors)
            soft_weights = soft_weights / (soft_weights.sum(axis=1, keepdims=True) + 1e-10)

        return soft_weights


def create_clustering_strategy(config: ClusteringConfig) -> ClusteringStrategy:
    """Factory function to create clustering strategy from config."""
    if config.method == "spectral":
        return SpectralClusteringStrategy()
    elif config.method == "k_subspaces":
        return KSubspacesClusteringStrategy()
    elif config.method == "ensc":
        return ENSCClusteringStrategy()
    else:
        raise ValueError(f"Unknown clustering method: {config.method}")
