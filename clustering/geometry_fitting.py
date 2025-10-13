"""Geometry-guided clustering refinement using Gromov-Wasserstein distance.

This module implements post-processing of soft cluster assignments by:
1. Fitting clusters to candidate belief geometries (K-simplices, circles, etc.)
2. Computing GW optimal distance via entropic Gromov-Wasserstein transport
3. Extracting per-point distortion contributions for filtering/refinement
4. Refining cluster memberships based on geometry fit quality

The core idea: Latents whose decoder directions align well with a belief geometry
(low GW distance) are kept in that cluster; high-distortion outliers are filtered.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any
import numpy as np
import warnings

try:
    import ot
    from ot.gromov import entropic_gromov_wasserstein
except ImportError:
    raise ImportError(
        "POT library required for geometry fitting. Install with: pip install POT"
    )

from gromov_monge_gap import (
    cosine_cost_matrix,
    euclidean_cost_matrix,
)
from .geometries import BeliefGeometry, SimplexGeometry, CircleGeometry


@dataclass
class GeometryFittingConfig:
    """Configuration for geometry-guided clustering refinement."""

    # Soft assignment extraction
    soft_assignment_method: Literal["top_m", "threshold", "both"] = "top_m"
    top_m: int = 3  # Keep top m clusters per latent
    soft_threshold: float = 0.1  # Minimum soft weight to keep

    # Geometry candidates
    simplex_k_range: Tuple[int, int] = (1, 8)
    include_circle: bool = True
    include_hypersphere: bool = False
    hypersphere_dims: List[int] = field(default_factory=lambda: [2, 3, 4])

    # Gromov-Wasserstein parameters
    gw_epsilon: float = 0.1
    gw_solver: Literal["PPA", "PGD"] = "PPA"
    gw_max_iter: int = 100
    gw_tol: float = 1e-9

    # Sinkhorn parameters
    sinkhorn_epsilon: float = 0.1
    sinkhorn_max_iter: int = 1000
    n_target_samples: int = 1000  # Number of uniform samples from target geometry

    # Cost function
    cost_fn: Literal["cosine", "euclidean"] = "cosine"
    normalize_vectors: bool = True

    # Filtering/refinement thresholds
    threshold_mode: Literal["normalized", "raw"] = "normalized"
    per_point_threshold: float = 0.5  # Threshold value (interpretation depends on threshold_mode)
    optimal_distortion_threshold: float = 1.0  # Optimal GW distortion above this = poor fit

    # Which metrics to use for filtering
    filter_metrics: List[Literal["gw_full"]] = field(
        default_factory=lambda: ["gw_full"]
    )


@dataclass
class GeometryFitResult:
    """Results from fitting a cluster to a single geometry via Gromov-Wasserstein."""

    geometry_name: str
    optimal_distance: float  # GW optimal distance
    gw_distortion_contributions: np.ndarray  # Per-point contributions to GW distance (shape: n_points)

    # Optional: The GW coupling matrix
    gw_coupling: Optional[np.ndarray] = None

    def get_metric(self, metric_name: str) -> np.ndarray:
        """Get per-point metric by name."""
        if metric_name == "gw_full":
            return self.gw_distortion_contributions
        raise ValueError(f"Unknown metric: {metric_name}")

    def to_summary_dict(self, include_raw_arrays: bool = False) -> Dict[str, Any]:
        """Convert to summary dictionary for JSON serialization.

        Args:
            include_raw_arrays: If True, include full per-point arrays (can be large)
        """
        summary = {
            "geometry_name": self.geometry_name,
            "optimal_distance": float(self.optimal_distance),
            "per_point_stats": {
                "gw_distortion_contributions": {
                    "mean": float(np.mean(self.gw_distortion_contributions)),
                    "std": float(np.std(self.gw_distortion_contributions)),
                    "min": float(np.min(self.gw_distortion_contributions)),
                    "max": float(np.max(self.gw_distortion_contributions)),
                    "median": float(np.median(self.gw_distortion_contributions)),
                },
            }
        }

        # Optionally include full raw arrays
        if include_raw_arrays:
            summary["per_point_raw_values"] = {
                "gw_distortion_contributions": self.gw_distortion_contributions.astype(float).tolist(),
            }

        return summary


@dataclass
class ClusterGeometryFits:
    """Geometry fit results for a single cluster."""

    cluster_id: int
    initial_member_mask: np.ndarray  # Boolean mask for initial members
    best_geometry: str
    best_optimal_distance: float  # GW optimal distance of best fit
    all_fits: Dict[str, GeometryFitResult]  # geometry_name -> fit result

    # Two-stage filtering details
    preliminary_keep_mask: Optional[np.ndarray] = None  # Stage 1: Points kept after consensus filtering

    # Final filtering details (populated during refinement)
    filtered_member_indices: Optional[np.ndarray] = None  # Indices that passed filtering
    removed_member_indices: Optional[np.ndarray] = None  # Indices that were removed
    filter_scores_raw: Optional[Dict[str, np.ndarray]] = None  # Raw per-point scores per metric
    filter_scores_normalized: Optional[Dict[str, np.ndarray]] = None  # Normalized [0,1] scores
    filter_scores_averaged: Optional[np.ndarray] = None  # Final averaged scores used for filtering
    raw_thresholds: Optional[Dict[str, float]] = None  # Raw threshold values per metric
    normalized_thresholds: Optional[Dict[str, float]] = None  # Normalized threshold values per metric
    threshold_mode: Optional[str] = None  # "normalized" or "raw"
    threshold_applied: Optional[float] = None  # The actual threshold value applied

    def to_summary_dict(self, include_raw_arrays: bool = True) -> Dict[str, Any]:
        """Convert to summary dictionary with all geometry fits.

        Args:
            include_raw_arrays: If True, include per-point raw values (default: True for analysis)
        """
        summary = {
            "cluster_id": int(self.cluster_id),
            "n_initial_members": int(self.initial_member_mask.sum()),
            "best_geometry": self.best_geometry,
            "best_optimal_distance": float(self.best_optimal_distance),
            "all_geometry_fits": {
                geom_name: fit.to_summary_dict(include_raw_arrays=include_raw_arrays)
                for geom_name, fit in self.all_fits.items()
            }
        }

        # Add preliminary filtering (stage 1) details
        if self.preliminary_keep_mask is not None:
            n_preliminary_kept = int(self.preliminary_keep_mask.sum())
            n_preliminary_removed = int((~self.preliminary_keep_mask).sum())
            summary["preliminary_filtering"] = {
                "n_kept": n_preliminary_kept,
                "n_removed": n_preliminary_removed,
                "description": "Stage 1: Removed points with high distortion in all top 3 geometries"
            }

        # Add final filtering (stage 2) details if available
        if self.filtered_member_indices is not None:
            summary["final_filtering"] = {
                "n_kept": int(len(self.filtered_member_indices)),
                "n_removed": int(len(self.removed_member_indices)) if self.removed_member_indices is not None else 0,
                "kept_indices": self.filtered_member_indices.tolist(),
                "removed_indices": self.removed_member_indices.tolist() if self.removed_member_indices is not None else [],
                "description": "Stage 2: Removed points with high distortion in final best geometry"
            }

            # Add threshold values and mode
            if self.threshold_mode is not None:
                summary["final_filtering"]["threshold_mode"] = self.threshold_mode
            if self.threshold_applied is not None:
                summary["final_filtering"]["threshold_applied"] = float(self.threshold_applied)
            if self.normalized_thresholds is not None:
                summary["final_filtering"]["normalized_thresholds"] = {
                    metric_name: float(threshold)
                    for metric_name, threshold in self.normalized_thresholds.items()
                }
            if self.raw_thresholds is not None:
                summary["final_filtering"]["raw_thresholds"] = {
                    metric_name: float(threshold)
                    for metric_name, threshold in self.raw_thresholds.items()
                }

            # Add per-point filter scores
            if self.filter_scores_raw is not None:
                summary["final_filtering"]["per_point_scores"] = {
                    "raw": {
                        metric_name: scores.astype(float).tolist()
                        for metric_name, scores in self.filter_scores_raw.items()
                    },
                    "normalized": {
                        metric_name: scores.astype(float).tolist()
                        for metric_name, scores in self.filter_scores_normalized.items()
                    } if self.filter_scores_normalized is not None else {},
                    "averaged": self.filter_scores_averaged.astype(float).tolist() if self.filter_scores_averaged is not None else [],
                }

        return summary


@dataclass
class RefinedClusteringResult:
    """Results from geometry-guided clustering refinement."""

    cluster_fits: Dict[int, ClusterGeometryFits]
    refined_assignments: np.ndarray  # (n_latents, n_clusters) soft assignments
    noise_mask: np.ndarray  # Boolean mask for noise latents
    config: GeometryFittingConfig


class GeometryFitter:
    """Fits a set of points to belief geometries using GMG analysis."""

    def __init__(self, config: GeometryFittingConfig):
        self.config = config

    def fit_cluster_to_geometries(
        self,
        cluster_points: np.ndarray,
        geometries: List[BeliefGeometry],
        verbose: bool = False
    ) -> Dict[str, GeometryFitResult]:
        """Test cluster against multiple geometries.

        Args:
            cluster_points: Array of shape (n_points, d_model)
            geometries: List of BeliefGeometry instances to test
            verbose: Print progress

        Returns:
            Dict mapping geometry_name -> GeometryFitResult
        """
        results = {}

        for geom in geometries:
            if verbose:
                print(f"  Testing {geom.get_name()}...")

            try:
                result = self._fit_single_geometry(cluster_points, geom)
                results[geom.get_name()] = result
            except Exception as e:
                print(f"  Warning: Failed to fit {geom.get_name()}: {e}")
                continue

        return results

    def _fit_single_geometry(
        self,
        points: np.ndarray,
        geometry: BeliefGeometry,
        verbose: bool = False
    ) -> GeometryFitResult:
        """Compute GW distance and per-point contributions for a single geometry.

        Args:
            points: Array of shape (n_points, d_model) - cluster latent decoder directions
            geometry: Target belief geometry
            verbose: Print progress
        Returns:
            GeometryFitResult with GW distance and per-point contributions
        """
        n = len(points)

        # 1. Sample uniform points from target geometry
        target_samples = geometry.sample_uniform(self.config.n_target_samples, seed=42)
        m = len(target_samples)

        if verbose:
            print("Computing cost matrices")
        # 2. Compute cost matrices
        if self.config.cost_fn == "cosine":
            C_source = cosine_cost_matrix(points, normalize=self.config.normalize_vectors)
            C_target = cosine_cost_matrix(target_samples, normalize=False)
        else:
            C_source = euclidean_cost_matrix(points)
            C_target = euclidean_cost_matrix(target_samples)

        # 3. Compute Gromov-Wasserstein optimal transport
        p = np.ones(n) / n
        q = np.ones(m) / m

        if verbose:
            print("Computing Gromov-Wasserstein optimal transport")

        # Try with configured epsilon, retry once with 50% higher epsilon if convergence fails
        epsilon = self.config.sinkhorn_epsilon
        for attempt in range(2):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                gw_coupling, gw_log = entropic_gromov_wasserstein(
                    C_source,
                    C_target,
                    p,
                    q,
                    loss_fun='square_loss',
                    epsilon=epsilon,
                    solver=self.config.gw_solver,
                    max_iter=self.config.gw_max_iter,
                    tol_rel=self.config.gw_tol,
                    tol_abs=self.config.gw_tol,
                    log=True,
                    verbose=False,
                    numItermax=self.config.sinkhorn_max_iter
                )

                # Check if Sinkhorn convergence warning was raised
                sinkhorn_warning = any(
                    "Sinkhorn did not converge" in str(warning.message)
                    for warning in w
                )

                if sinkhorn_warning and attempt == 0:
                    # Retry with 50% higher epsilon
                    epsilon = self.config.sinkhorn_epsilon * 1.5
                    if verbose:
                        print(f"  Sinkhorn convergence warning, retrying with epsilon={epsilon:.3f}")
                else:
                    # Success or second attempt - break
                    break

        # 4. Extract GW distance
        optimal_distance = gw_log['gw_dist']
        
        if verbose:
            print("Computing per-point contributions to GW distance")
        # 5. Compute per-point contributions to GW distance
        gw_distortions = self._compute_gw_point_distortions(
            C_source, C_target, gw_coupling, p
        )

        return GeometryFitResult(
            geometry_name=geometry.get_name(),
            optimal_distance=optimal_distance,
            gw_distortion_contributions=gw_distortions,
            gw_coupling=gw_coupling
        )

    def _compute_gw_point_distortions(
        self,
        C_source: np.ndarray,
        C_target: np.ndarray,
        T: np.ndarray,
        p: np.ndarray = None
    ) -> np.ndarray:
        """Compute full GW point-level distortion contributions.

        This is the exact formulation with nested loops over all point pairs.

        Args:
            C_source: Source cost matrix (n, n)
            C_target: Target cost matrix (m, m)
            T: GW coupling matrix (n, m)
            p: Source distribution (n,)

        Returns:
            Per-point distortion contributions (n,)
        """
        n = C_source.shape[0]
        m = C_target.shape[0]
        point_distortions = np.zeros(n)

        ## VERY slow version
        # for i in range(n):
        #     print(f"Computing per-point contributions to GW distance for point {i}")
        #     for k in range(n):
        #         print(f"Computing per-point contributions to GW distance for point {i}, k={k}")
        #         for j in range(m):
        #             for l in range(m):
        #                 distortion_term = (C_source[i, k] - C_target[j, l]) ** 2
        #                 point_distortions[i] += T[i, j] * T[k, l] * distortion_term

        C_s_squared = C_source ** 2
        C_t_squared = C_target ** 2

        T_row_sums = T.sum(axis=1)


        ## Term 1
        term_1 = T_row_sums * (C_s_squared @ T_row_sums)
        ## Term 2

        term_2 = np.sum(C_source * (T @ C_target @ T.T), axis=1)

        ## Term 3
        term_3 = (T @ C_t_squared @ T.T).sum(axis=1)

        point_distortions_fast = term_1 - 2 * term_2 + term_3

        # Normalize by point mass
        point_distortions_fast = point_distortions_fast * (p + 1e-9) if p is not None else point_distortions_fast

        return point_distortions_fast


class GeometryRefinementPipeline:
    """Main pipeline for geometry-guided clustering refinement."""

    def __init__(self, config: GeometryFittingConfig):
        self.config = config
        self.fitter = GeometryFitter(config)

    def refine_clusters(
        self,
        decoder_dirs: np.ndarray,
        soft_assignments: np.ndarray,
        verbose: bool = False
    ) -> RefinedClusteringResult:
        """Refine soft cluster assignments using geometry fitting.

        Args:
            decoder_dirs: SAE decoder directions (n_latents, d_model)
            soft_assignments: Soft cluster weights (n_latents, n_clusters)
            verbose: Print progress

        Returns:
            RefinedClusteringResult with geometry fits and refined assignments
        """
        n_latents, n_clusters = soft_assignments.shape

        if verbose:
            print(f"Refining {n_clusters} clusters with {n_latents} latents")

        # 1. Extract cluster memberships from soft assignments
        cluster_members = self._extract_soft_clusters(soft_assignments)

        # 2. Build geometry catalog
        from .geometries import create_geometry_catalog
        geometries = create_geometry_catalog(
            simplex_k_range=self.config.simplex_k_range,
            include_circle=self.config.include_circle,
            include_hypersphere=self.config.include_hypersphere,
            hypersphere_dims=self.config.hypersphere_dims
        )

        if verbose:
            print(f"Testing {len(geometries)} candidate geometries")

        # 3. Fit each cluster to each geometry
        cluster_fits = {}

        for cluster_id, member_mask in cluster_members.items():
            if verbose:
                print(f"\nCluster {cluster_id}: {member_mask.sum()} members")

            cluster_points = decoder_dirs[member_mask]

            if len(cluster_points) < 2:
                if verbose:
                    print(f"  Skipping (too few points)")
                continue

            # Fit to all geometries (initial fit with full cluster)
            fits = self.fitter.fit_cluster_to_geometries(
                cluster_points, geometries, verbose=verbose
            )

            if not fits:
                if verbose:
                    print(f"  No valid fits")
                continue

            # Rank geometries by optimal distance
            sorted_geometries = sorted(fits.keys(), key=lambda g: fits[g].optimal_distance)

            if verbose:
                print(f"  Top 3 initial fits: {', '.join([f'{g}={fits[g].optimal_distance:.4f}' for g in sorted_geometries[:3]])}")

            # Stage 1: Preliminary filtering using top 3 geometries
            top_k = min(3, len(sorted_geometries))
            top_geoms = sorted_geometries[:top_k]

            # Identify points with high distortion in ALL top geometries
            preliminary_keep_mask = self._preliminary_filter(
                cluster_points, fits, top_geoms, verbose=verbose
            )

            if verbose:
                n_removed = (~preliminary_keep_mask).sum()
                if n_removed > 0:
                    print(f"  Stage 1: Removed {n_removed} points (high distortion in all top {top_k} geometries)")

            # Get filtered cluster points
            filtered_cluster_points = cluster_points[preliminary_keep_mask]

            # Stage 2: Refit top geometries with filtered points
            if len(filtered_cluster_points) >= 2 and filtered_cluster_points.shape[0] < cluster_points.shape[0]:
                if verbose:
                    print(f"  Stage 2: Refitting top {top_k} geometries with {len(filtered_cluster_points)} filtered points")

                refits = {}
                for geom_name in top_geoms:
                    geom = next(g for g in geometries if g.get_name() == geom_name)
                    try:
                        refit_result = self.fitter._fit_single_geometry(filtered_cluster_points, geom)
                        refits[geom_name] = refit_result
                    except Exception as e:
                        if verbose:
                            print(f"    Warning: Refit failed for {geom_name}: {e}")
                        # Keep original fit if refit fails
                        refits[geom_name] = fits[geom_name]

                # Update fits with refits
                for geom_name, refit in refits.items():
                    fits[geom_name] = refit

                if verbose:
                    print(f"  Refit results: {', '.join([f'{g}={refits[g].optimal_distance:.4f}' for g in refits.keys()])}")

            # Select final best geometry (from refits if available, otherwise from initial fits)
            best_geom = min(fits.keys(), key=lambda g: fits[g].optimal_distance)
            best_fit = fits[best_geom]
            best_optimal_distance = best_fit.optimal_distance

            cluster_fits[cluster_id] = ClusterGeometryFits(
                cluster_id=cluster_id,
                initial_member_mask=member_mask,
                best_geometry=best_geom,
                best_optimal_distance=best_optimal_distance,
                all_fits=fits,
                preliminary_keep_mask=preliminary_keep_mask  # Store for reference
            )

            if verbose:
                print(f"  Final best: {best_geom}, GW_dist={best_optimal_distance:.4f}")

        # 4. Refine memberships using per-point distortion metrics
        refined_assignments = self._refine_memberships(
            decoder_dirs,
            soft_assignments,
            cluster_fits,
            verbose=verbose
        )

        # 5. Identify noise latents
        noise_mask = self._identify_noise_latents(
            refined_assignments,
            cluster_fits
        )

        if verbose:
            print(f"\nNoise latents: {noise_mask.sum()} / {n_latents}")

        return RefinedClusteringResult(
            cluster_fits=cluster_fits,
            refined_assignments=refined_assignments,
            noise_mask=noise_mask,
            config=self.config
        )

    def _extract_soft_clusters(
        self,
        soft_assignments: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Extract cluster memberships from soft assignment matrix.

        Uses config.soft_assignment_method to decide which latents belong
        to which clusters.

        Args:
            soft_assignments: (n_latents, n_clusters) soft weights

        Returns:
            Dict mapping cluster_id -> boolean mask of members
        """
        n_latents, n_clusters = soft_assignments.shape
        cluster_members = {}

        for cluster_id in range(n_clusters):
            cluster_weights = soft_assignments[:, cluster_id]

            if self.config.soft_assignment_method == "top_m":
                # Keep latents where this cluster is in top m
                top_m_clusters = np.argsort(soft_assignments, axis=1)[:, -self.config.top_m:]
                member_mask = np.any(top_m_clusters == cluster_id, axis=1)

            elif self.config.soft_assignment_method == "threshold":
                # Keep latents above threshold
                member_mask = cluster_weights >= self.config.soft_threshold

            else:  # "both"
                # Keep if in top m OR above threshold
                top_m_clusters = np.argsort(soft_assignments, axis=1)[:, -self.config.top_m:]
                top_m_mask = np.any(top_m_clusters == cluster_id, axis=1)
                threshold_mask = cluster_weights >= self.config.soft_threshold
                member_mask = top_m_mask | threshold_mask

            cluster_members[cluster_id] = member_mask

        return cluster_members

    def _preliminary_filter(
        self,
        cluster_points: np.ndarray,
        fits: Dict[str, GeometryFitResult],
        top_geoms: List[str],
        verbose: bool = False
    ) -> np.ndarray:
        """Stage 1 filtering: Remove points with high distortion in ALL top geometries.

        Args:
            cluster_points: Original cluster points (n_points, d_model)
            fits: All geometry fit results
            top_geoms: List of top geometry names (usually top 3)
            verbose: Print progress

        Returns:
            Boolean mask of points to keep
        """
        n_points = len(cluster_points)

        # For each point, check if it has high distortion in ALL top geometries
        high_in_all = np.ones(n_points, dtype=bool)  # Start assuming all are bad

        for geom_name in top_geoms:
            fit = fits[geom_name]

            # Get distortion scores for this geometry
            distortions = fit.gw_distortion_contributions

            # Apply threshold based on mode
            if self.config.threshold_mode == "normalized":
                # Normalize to [0, 1]
                distortions_norm = (distortions - distortions.min()) / (distortions.max() - distortions.min() + 1e-10)
                high_distortion = distortions_norm > self.config.per_point_threshold
            else:  # "raw"
                high_distortion = distortions > self.config.per_point_threshold

            # Point only removed if high in ALL top geometries
            high_in_all = high_in_all & high_distortion

        # Keep points that are NOT high in all top geometries
        keep_mask = ~high_in_all

        return keep_mask

    def _refine_memberships(
        self,
        decoder_dirs: np.ndarray,
        soft_assignments: np.ndarray,
        cluster_fits: Dict[int, ClusterGeometryFits],
        verbose: bool = False
    ) -> np.ndarray:
        """Refine cluster memberships using per-point distortion metrics.

        Args:
            decoder_dirs: Decoder directions (n_latents, d_model)
            soft_assignments: Initial soft assignments (n_latents, n_clusters)
            cluster_fits: Geometry fit results per cluster
            verbose: Print progress

        Returns:
            Refined soft assignments (n_latents, n_clusters)
        """
        n_latents, n_clusters = soft_assignments.shape
        refined = soft_assignments.copy()

        for cluster_id, fit_result in cluster_fits.items():
            member_mask = fit_result.initial_member_mask
            best_fit = fit_result.all_fits[fit_result.best_geometry]

            # Get per-point metrics for filtering
            filter_scores_raw = {}
            filter_scores_normalized = {}
            filter_scores_list = []

            for metric_name in self.config.filter_metrics:
                scores_raw = best_fit.get_metric(metric_name)
                filter_scores_raw[metric_name] = scores_raw

                # Normalize to [0, 1]
                scores_norm = (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min() + 1e-10)
                filter_scores_normalized[metric_name] = scores_norm
                filter_scores_list.append(scores_norm)

            # Average filter scores
            avg_score = np.mean(filter_scores_list, axis=0)

            # Apply threshold based on mode
            if self.config.threshold_mode == "normalized":
                # Apply threshold to normalized scores
                keep_mask = avg_score <= self.config.per_point_threshold
                threshold_applied = self.config.per_point_threshold
                threshold_type = "normalized"
            else:  # "raw"
                # Apply threshold directly to raw scores
                avg_raw_score = np.mean([filter_scores_raw[m] for m in self.config.filter_metrics], axis=0)
                keep_mask = avg_raw_score <= self.config.per_point_threshold
                threshold_applied = self.config.per_point_threshold
                threshold_type = "raw"

            # Update soft assignments: zero out high-distortion members
            all_member_indices = np.where(member_mask)[0]

            # Determine which member indices the metrics correspond to
            # The metrics could correspond to either filtered points or all points depending on whether refit happened
            if fit_result.preliminary_keep_mask is not None:
                # Check if metrics match filtered points or all points by comparing shapes
                n_filtered = fit_result.preliminary_keep_mask.sum()
                n_metrics = len(avg_score)

                if n_metrics == n_filtered and n_filtered < len(all_member_indices):
                    # Metrics correspond to filtered members (refit happened)
                    member_indices = all_member_indices[fit_result.preliminary_keep_mask]
                else:
                    # Metrics correspond to all members (no refit or refit failed)
                    member_indices = all_member_indices
            else:
                # No preliminary filtering - metrics correspond to all members
                member_indices = all_member_indices

            # Apply refinement filter
            kept_indices = member_indices[keep_mask]
            remove_indices = member_indices[~keep_mask]
            refined[remove_indices, cluster_id] = 0.0

            # Compute threshold values in both spaces for reporting
            raw_thresholds = {}
            normalized_thresholds = {}
            for metric_name in self.config.filter_metrics:
                scores_raw = filter_scores_raw[metric_name]
                scores_norm = filter_scores_normalized[metric_name]
                raw_min = scores_raw.min()
                raw_max = scores_raw.max()

                if self.config.threshold_mode == "normalized":
                    # Compute raw equivalent of normalized threshold
                    raw_threshold = self.config.per_point_threshold * (raw_max - raw_min) + raw_min
                    raw_thresholds[metric_name] = raw_threshold
                    normalized_thresholds[metric_name] = self.config.per_point_threshold
                else:  # "raw"
                    # Compute normalized equivalent of raw threshold
                    if raw_max - raw_min > 1e-10:
                        normalized_threshold = (self.config.per_point_threshold - raw_min) / (raw_max - raw_min)
                    else:
                        normalized_threshold = 0.5
                    raw_thresholds[metric_name] = self.config.per_point_threshold
                    normalized_thresholds[metric_name] = normalized_threshold

            # Store filtering details in fit_result
            fit_result.filtered_member_indices = kept_indices
            fit_result.removed_member_indices = remove_indices
            fit_result.filter_scores_raw = filter_scores_raw
            fit_result.filter_scores_normalized = filter_scores_normalized
            fit_result.filter_scores_averaged = avg_score
            fit_result.raw_thresholds = raw_thresholds
            fit_result.normalized_thresholds = normalized_thresholds
            fit_result.threshold_mode = self.config.threshold_mode
            fit_result.threshold_applied = threshold_applied

            if verbose and len(remove_indices) > 0:
                print(f"  Cluster {cluster_id}: removed {len(remove_indices)} high-distortion latents")

        return refined

    def _identify_noise_latents(
        self,
        refined_assignments: np.ndarray,
        cluster_fits: Dict[int, ClusterGeometryFits]
    ) -> np.ndarray:
        """Identify latents that don't fit well in any cluster.

        A latent is noise if:
        - It has zero assignment to all clusters after refinement, OR
        - All clusters it belongs to have poor geometry fit (high optimal distortion)

        Args:
            refined_assignments: Refined soft assignments (n_latents, n_clusters)
            cluster_fits: Geometry fit results

        Returns:
            Boolean mask of noise latents
        """
        n_latents = len(refined_assignments)

        # Latents with no cluster membership
        no_cluster_mask = refined_assignments.sum(axis=1) == 0

        # Latents only in poorly-fitting clusters
        poor_fit_mask = np.zeros(n_latents, dtype=bool)
        for i in range(n_latents):
            assigned_clusters = np.where(refined_assignments[i] > 0)[0]
            if len(assigned_clusters) == 0:
                continue

            # Check if all assigned clusters have poor fit (high optimal distance)
            all_poor = all(
                cluster_fits[cid].best_optimal_distance > self.config.optimal_distortion_threshold
                for cid in assigned_clusters
                if cid in cluster_fits
            )
            if all_poor:
                poor_fit_mask[i] = True

        return no_cluster_mask | poor_fit_mask
