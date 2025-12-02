"""Subspace clustering utilities for SAE latent analysis.

Implements K-Subspaces and Elastic-Net Subspace Clustering (ENSC) for finding
near-orthogonal groups of decoder directions.

Example Usage:
    >>> import numpy as np
    >>> from subspace_clustering_utils import (
    ...     normalize_and_deduplicate,
    ...     k_subspaces_clustering,
    ...     ensc_clustering,
    ... )
    >>>
    >>> # Load decoder directions (n_latents, d_features)
    >>> decoder_directions = ...  # Your SAE decoder matrix
    >>>
    >>> # Normalize and deduplicate
    >>> normalized, kept_indices = normalize_and_deduplicate(decoder_directions)
    >>>
    >>> # Option 1: Fully automatic clustering
    >>> result = k_subspaces_clustering(
    ...     normalized,
    ...     n_clusters=None,      # Auto-detect K
    ...     subspace_rank=None,   # Auto-detect rank per cluster
    ... )
    >>>
    >>> # Option 2: Specify K, auto-detect ranks
    >>> result = k_subspaces_clustering(
    ...     normalized,
    ...     n_clusters=5,         # Use 5 clusters
    ...     subspace_rank=None,   # Auto-detect rank per cluster
    ... )
    >>>
    >>> # Option 3: Specify both manually (original behavior)
    >>> result = k_subspaces_clustering(
    ...     normalized,
    ...     n_clusters=5,
    ...     subspace_rank=3,
    ... )
    >>>
    >>> # Access results
    >>> print(f"Found {result.n_clusters} clusters")
    >>> print(f"Cluster assignments: {result.cluster_labels}")
    >>> print(f"Actual ranks per cluster: {result.cluster_ranks}")
    >>>
    >>> # ENSC works the same way
    >>> result_ensc = ensc_clustering(
    ...     normalized,
    ...     n_clusters=None,      # Auto-detect from eigengap
    ...     subspace_rank=None,   # Auto-detect per cluster
    ... )
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Any, List

import numpy as np
import torch
from scipy.linalg import qr, svd, orth, subspace_angles
from sklearn.cluster import spectral_clustering
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

from mess3_gmg_analysis_utils import (
    fit_residual_to_belief_map,
    lasso_select_latents,
)


@dataclass
class RankEstimationDetails:
    """Details about how subspace rank was estimated."""
    variance_rank: int  # Rank suggested by variance threshold
    gap_rank: int  # Rank suggested by singular value gap
    final_rank: int  # Actual rank chosen (min of variance_rank and gap_rank)
    limiting_method: str  # "variance", "gap", or "both" (if equal)


@dataclass
class SubspaceClusterResult:
    """Result container for subspace clustering."""

    cluster_labels: np.ndarray  # (n_points,) cluster assignment for each point
    subspace_bases: Dict[int, np.ndarray]  # cluster_id -> orthonormal basis matrix
    reconstruction_errors: Dict[int, float]  # cluster_id -> mean reconstruction error
    total_reconstruction_error: float
    n_clusters: int
    subspace_rank: int | None  # None if auto-detected per cluster
    method: str  # "k_subspaces" or "ensc"

    # Per-cluster ranks (useful when subspace_rank=None and auto-detected)
    cluster_ranks: Dict[int, int] | None = None

    # Rank estimation details per cluster (when auto-detected)
    rank_estimation_details: Dict[int, RankEstimationDetails] | None = None

    # Diagnostics
    principal_angles: Dict[Tuple[int, int], np.ndarray] | None = None
    within_projection_energy: Dict[int, float] | None = None
    between_projection_energy: float | None = None


def normalize_and_deduplicate(
    vectors: np.ndarray,
    cosine_threshold: float = 0.995,
    protected_indices: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """L2-normalize vectors and remove near-duplicates.

    Args:
        vectors: (n_vectors, d_features) array
        cosine_threshold: Remove vectors with |cosine_similarity| > threshold

    Returns:
        normalized_vectors: (n_kept, d_features) L2-normalized unique vectors
        kept_indices: (n_kept,) indices of kept vectors in original array
    """
    # L2 normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = vectors / norms

    # Remove near-duplicates greedily
    kept_mask = np.ones(len(normalized), dtype=bool)
    protected_set = set(protected_indices) if protected_indices is not None else set()

    for i in range(len(normalized)):
        if not kept_mask[i]:
            continue
        # Check against all subsequent vectors
        for j in range(i + 1, len(normalized)):
            if not kept_mask[j]:
                continue
            cos_sim = np.abs(np.dot(normalized[i], normalized[j]))
            if cos_sim > cosine_threshold:
                if j in protected_set and i in protected_set:
                    continue
                if j in protected_set:
                    kept_mask[i] = False
                    break
                kept_mask[j] = False

    kept_indices = np.where(kept_mask)[0]
    return normalized[kept_indices], kept_indices


def _compute_belief_seed_sets(
    acts_flat: torch.Tensor,
    feature_acts_active: torch.Tensor,
    decoder_active: np.ndarray,
    component_beliefs: Dict[str, np.ndarray],
    *,
    ridge_alpha: float,
    lasso_alpha: float | None,
    lasso_cv: int,
    lasso_max_iter: int,
    coef_threshold: float,
    max_latents: int | None,
    random_state: int,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, Any]], np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    if not component_beliefs:
        raise ValueError("Component beliefs are required for belief-aligned seeding")

    component_names = list(component_beliefs.keys())
    belief_matrices = [component_beliefs[name] for name in component_names]
    if any(mat.shape[0] != belief_matrices[0].shape[0] for mat in belief_matrices):
        raise ValueError("All component belief matrices must share the same number of samples")

    beliefs_concat = np.concatenate(belief_matrices, axis=1)
    acts_np = acts_flat.detach().cpu().numpy()
    residual_map, residual_intercept = fit_residual_to_belief_map(
        acts_np,
        beliefs_concat,
        alpha=ridge_alpha,
    )

    if decoder_active.shape[1] != residual_map.shape[0]:
        raise ValueError(
            f"Decoder width {decoder_active.shape[1]} must equal residual_to_belief rows {residual_map.shape[0]}"
        )

    # latent_sens shape: (n_active_latents, total_belief_dim).
    # Each column records how one latent's decoder contribution moves a specific belief coordinate.
    latent_sens = decoder_active @ residual_map
    feature_np = feature_acts_active.detach().cpu().numpy()

    seed_map: Dict[str, List[int]] = {}
    component_metadata: Dict[str, Dict[str, Any]] = {}
    slice_map: Dict[str, Tuple[int, int]] = {}

    offset = 0
    for name, belief_matrix in zip(component_names, belief_matrices):
        comp_dim = belief_matrix.shape[1]
        start, end = offset, offset + comp_dim
        slice_map[name] = (start, end)
        offset = end

        comp_scores = np.zeros(decoder_active.shape[0], dtype=np.float64)
        per_dim_details: List[Dict[str, Any]] = []

        for local_idx, global_idx in enumerate(range(start, end)):
            sens_vec = latent_sens[:, global_idx]
            if np.allclose(sens_vec, 0.0):
                per_dim_details.append(
                    {
                        "dimension": int(local_idx),
                        "alpha": None,
                        "selected_latents": [],
                        "skipped": "zero_sensitivity",
                    }
                )
                continue

            # Broadcast sensitivities so column j becomes feature_np[:, j] * sens_vec[j].
            design = feature_np * sens_vec[np.newaxis, :]
            if np.allclose(design, 0.0):
                per_dim_details.append(
                    {
                        "dimension": int(local_idx),
                        "alpha": None,
                        "selected_latents": [],
                        "skipped": "zero_design",
                    }
                )
                continue

            try:
                coef, intercept, alpha_used = lasso_select_latents(
                    design,
                    belief_matrix[:, local_idx],
                    cv=lasso_cv,
                    max_iter=lasso_max_iter,
                    random_state=random_state,
                    alpha=lasso_alpha,
                )
            except ValueError as exc:
                per_dim_details.append(
                    {
                        "dimension": int(local_idx),
                        "alpha": None,
                        "selected_latents": [],
                        "error": str(exc),
                    }
                )
                continue

            abs_coef = np.abs(coef)
            comp_scores += abs_coef
            selected_latents = np.nonzero(abs_coef > 0.0)[0]
            per_dim_details.append(
                {
                    "dimension": int(local_idx),
                    "alpha": float(alpha_used),
                    "selected_latents": [int(idx) for idx in selected_latents],
                    "coef_sum": float(abs_coef.sum()),
                }
            )

        sorted_latents = np.argsort(comp_scores)[::-1]
        if coef_threshold is not None and coef_threshold > 0.0:
            selected = [int(idx) for idx in sorted_latents if comp_scores[idx] >= coef_threshold]
        else:
            selected = [int(idx) for idx in sorted_latents if comp_scores[idx] > 0.0]

        if max_latents is not None:
            selected = selected[:max_latents]

        seed_map[name] = selected
        component_metadata[name] = {
            "scores": {int(idx): float(comp_scores[idx]) for idx in np.nonzero(comp_scores > 0.0)[0]},
            "per_dimension": per_dim_details,
            "total_score": float(comp_scores.sum()),
        }

    return seed_map, component_metadata, residual_map, residual_intercept, slice_map



def pivoted_qr_seeds(
    vectors: np.ndarray,
    n_seeds: int,
) -> np.ndarray:
    """Select seed vectors using column-pivoted QR decomposition.

    Args:
        vectors: (n_vectors, d_features) array
        n_seeds: Number of seed vectors to select

    Returns:
        seed_indices: (n_seeds,) indices of selected seed vectors
    """
    # Transpose so we do QR on columns
    _, _, perm = qr(vectors.T, pivoting=True)
    # Take first n_seeds pivot columns
    return perm[:n_seeds]


def fit_subspace_qr(vectors: np.ndarray, rank: int) -> np.ndarray:
    """Fit a subspace to vectors using QR decomposition.

    Args:
        vectors: (n_vectors, d_features) array of vectors to fit
        rank: Target rank of subspace

    Returns:
        basis: (d_features, rank) orthonormal basis matrix
    """
    if len(vectors) == 0:
        raise ValueError("Cannot fit subspace to empty set of vectors")

    if len(vectors) < rank:
        rank = len(vectors)

    # Use SVD for better numerical stability and explicit rank control
    U, s, Vt = svd(vectors, full_matrices=False)
    # Keep top 'rank' components
    basis = Vt[:rank].T  # (d_features, rank)

    return basis


def estimate_subspace_rank_svd(
    vectors: np.ndarray,
    variance_threshold: float = 0.95,
    gap_threshold: float = 2.0,
) -> RankEstimationDetails:
    """Estimate subspace rank from singular value decay.

    Uses two heuristics:
    1. Cumulative variance threshold (e.g., capture 95% of variance)
    2. Singular value gap detection (large ratio between consecutive values)

    Args:
        vectors: (n_vectors, d_features) cluster vectors
        variance_threshold: Cumulative variance fraction to capture
        gap_threshold: Minimum ratio sigma_i/sigma_{i+1} to detect gap

    Returns:
        RankEstimationDetails with variance_rank, gap_rank, final_rank, and limiting_method
    """
    if len(vectors) == 0:
        return RankEstimationDetails(
            variance_rank=1,
            gap_rank=1,
            final_rank=1,
            limiting_method="both"
        )

    U, s, Vt = svd(vectors, full_matrices=False)

    if len(s) == 0:
        return RankEstimationDetails(
            variance_rank=1,
            gap_rank=1,
            final_rank=1,
            limiting_method="both"
        )

    # Method 1: Cumulative variance
    total_var = np.sum(s**2)
    if total_var == 0:
        return RankEstimationDetails(
            variance_rank=1,
            gap_rank=1,
            final_rank=1,
            limiting_method="both"
        )
    cumsum_var = np.cumsum(s**2)
    variance_rank = int(np.searchsorted(cumsum_var / total_var, variance_threshold) + 1)

    # Method 2: Singular value gaps
    if len(s) > 1:
        ratios = s[:-1] / (s[1:] + 1e-10)  # Avoid division by zero
        gap_indices = np.where(ratios >= gap_threshold)[0]
        gap_rank = int(gap_indices[0] + 1) if len(gap_indices) > 0 else len(s)
    else:
        gap_rank = 1

    # Determine final rank and which method limited it
    final_rank = min(variance_rank, gap_rank)

    if variance_rank == gap_rank:
        limiting_method = "both"
    elif final_rank == variance_rank:
        limiting_method = "variance"
    else:
        limiting_method = "gap"

    return RankEstimationDetails(
        variance_rank=variance_rank,
        gap_rank=gap_rank,
        final_rank=final_rank,
        limiting_method=limiting_method
    )


def estimate_n_clusters_eigengap(
    affinity: np.ndarray,
    max_clusters: int = 10,
) -> int:
    """Estimate number of clusters from eigengap in affinity matrix.

    Computes eigenvalues of normalized Laplacian and finds the largest
    gap, which indicates the natural number of clusters.

    Args:
        affinity: (n, n) symmetric affinity/similarity matrix
        max_clusters: Maximum K to consider

    Returns:
        Estimated number of clusters (between 2 and max_clusters)
    """
    from scipy.linalg import eigh

    n = affinity.shape[0]
    max_clusters = min(max_clusters, n - 1)

    # Compute normalized Laplacian
    D = np.sum(affinity, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D + 1e-10)
    L_norm = np.eye(n) - (D_inv_sqrt[:, None] * affinity * D_inv_sqrt[None, :])

    # Get smallest eigenvalues (number of connected components)
    eigvals = eigh(L_norm, eigvals_only=True, subset_by_index=[0, max_clusters])
    eigvals = np.sort(eigvals)

    # Find largest gap in eigenvalues
    if len(eigvals) > 1:
        gaps = eigvals[1:] - eigvals[:-1]
        best_k = int(np.argmax(gaps) + 1)
        # Ensure at least 2 clusters
        return max(2, min(best_k, max_clusters))
    else:
        return 2


def find_elbow_rank(
    vectors: np.ndarray,
    max_rank: int = 20,
) -> Tuple[int, np.ndarray]:
    """Find elbow in reconstruction error vs rank curve.

    Args:
        vectors: (n_vectors, d_features) cluster vectors
        max_rank: Maximum rank to consider

    Returns:
        best_rank: Rank at elbow point
        errors: Array of reconstruction errors for each rank
    """
    max_rank = min(max_rank, len(vectors))
    errors = []

    for r in range(1, max_rank + 1):
        basis = fit_subspace_qr(vectors, r)
        proj = vectors @ basis @ basis.T
        error = np.mean(np.sum((vectors - proj)**2, axis=1))
        errors.append(error)

    errors = np.array(errors)

    if len(errors) <= 2:
        return 1, errors

    # Find elbow using second derivative (acceleration)
    second_deriv = np.diff(errors, 2)
    # Most negative second derivative = sharpest bend
    elbow_idx = int(np.argmin(second_deriv))
    best_rank = min(elbow_idx + 1, len(errors))  # +1 for 1-indexing

    return max(1, best_rank), errors


def assign_to_subspaces(
    vectors: np.ndarray,
    subspace_bases: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Assign each vector to the nearest subspace.

    Args:
        vectors: (n_vectors, d_features) array
        subspace_bases: Dict mapping cluster_id -> (d_features, rank) basis

    Returns:
        labels: (n_vectors,) cluster assignment
        reconstruction_errors: Dict mapping cluster_id -> mean squared error
    """
    n_vectors = len(vectors)
    n_clusters = len(subspace_bases)

    # Compute reconstruction error for each vector under each subspace
    errors = np.zeros((n_vectors, n_clusters))

    for cluster_idx, (cluster_id, basis) in enumerate(subspace_bases.items()):
        # Project onto subspace: x_proj = U U^T x
        projections = vectors @ basis @ basis.T
        # Reconstruction error: ||x - x_proj||^2
        residuals = vectors - projections
        errors[:, cluster_idx] = np.sum(residuals ** 2, axis=1)

    # Assign to subspace with minimum error
    best_clusters = np.argmin(errors, axis=1)

    # Map back to cluster IDs
    cluster_ids = list(subspace_bases.keys())
    labels = np.array([cluster_ids[idx] for idx in best_clusters])

    # Compute mean reconstruction error per cluster
    reconstruction_errors = {}
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        if mask.sum() > 0:
            reconstruction_errors[cluster_id] = float(errors[mask, cluster_ids.index(cluster_id)].mean())
        else:
            reconstruction_errors[cluster_id] = 0.0

    return labels, reconstruction_errors


def k_subspaces_clustering(
    vectors: np.ndarray,
    n_clusters: int | None = None,
    subspace_rank: int | None = None,
    max_iters: int = 20,
    random_state: int | None = None,
    max_clusters: int = 10,
    variance_threshold: float = 0.95,
    gap_threshold: float = 2.0,
    initial_clusters: Dict[int, Sequence[int]] | None = None,
    lock_mode: str = "fixed",
) -> SubspaceClusterResult:
    """K-Subspaces clustering with pivoted QR seeding.

    Args:
        vectors: (n_vectors, d_features) normalized array
        n_clusters: Number of subspaces (K). If None, uses sqrt(n_vectors) heuristic
        subspace_rank: Rank of each subspace (r). If None, auto-estimated per cluster
        max_iters: Maximum alternating iterations
        random_state: Random seed for reproducibility
        max_clusters: Maximum K when auto-detecting (only used if n_clusters=None)
        variance_threshold: Variance threshold for auto rank selection (only used if subspace_rank=None)
        gap_threshold: Gap threshold for auto rank selection (only used if subspace_rank=None)

    Returns:
        SubspaceClusterResult with cluster assignments and subspace bases
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_vectors, d_features = vectors.shape

    # Auto-detect n_clusters if not provided
    seed_clusters: Dict[int, np.ndarray] = {}
    locked_assignments: Dict[int, int] = {}
    assigned_seed_indices: set[int] = set()

    if initial_clusters:
        for cluster_id, indices in initial_clusters.items():
            idx_arr = np.array(indices, dtype=int)
            if idx_arr.size == 0:
                continue
            if np.any(idx_arr < 0) or np.any(idx_arr >= n_vectors):
                raise ValueError("Initial cluster indices out of range for k-subspaces")
            unique_indices = np.unique(idx_arr)
            seed_clusters[int(cluster_id)] = unique_indices
            if lock_mode == "fixed":
                for idx in unique_indices:
                    locked_assignments[idx] = int(cluster_id)
            assigned_seed_indices.update(unique_indices.tolist())

    min_clusters = max(len(seed_clusters), 2)

    if n_clusters is None:
        n_clusters = min(int(np.sqrt(n_vectors)), max_clusters, n_vectors)
        n_clusters = max(min_clusters, n_clusters)

    if n_clusters > n_vectors:
        raise ValueError(f"n_clusters ({n_clusters}) cannot exceed n_vectors ({n_vectors})")

    if seed_clusters and max(seed_clusters.keys()) >= n_clusters:
        raise ValueError(
            f"Seed cluster id {max(seed_clusters.keys())} exceeds n_clusters={n_clusters - 1}"
        )

    # Seed subspaces
    subspace_bases: Dict[int, np.ndarray] = {}
    for cluster_id, seed_indices in seed_clusters.items():
        seed_vectors = vectors[seed_indices]
        rank = min(len(seed_vectors), subspace_rank) if subspace_rank is not None else len(seed_vectors)
        rank = max(1, rank)
        basis = fit_subspace_qr(seed_vectors, rank)
        subspace_bases[cluster_id] = basis

    remaining_cluster_ids = [cid for cid in range(n_clusters) if cid not in subspace_bases]
    if remaining_cluster_ids:
        available_indices = [idx for idx in range(n_vectors) if idx not in assigned_seed_indices]
        if len(available_indices) < len(remaining_cluster_ids):
            raise ValueError("Not enough unassigned vectors to initialize remaining clusters")
        remaining_vectors = vectors[available_indices]
        seed_positions = pivoted_qr_seeds(remaining_vectors, len(remaining_cluster_ids))
        for cluster_id, pos in zip(remaining_cluster_ids, seed_positions):
            seed_idx = available_indices[pos]
            seed_vector = vectors[seed_idx:seed_idx+1]
            basis = orth(seed_vector.T)
            subspace_bases[cluster_id] = basis

    # Alternating optimization
    labels = None
    prev_labels = None
    rank_details: Dict[int, RankEstimationDetails] = {}

    print_interval = max(1, max_iters // 30)
    for iter_idx in range(max_iters):
        if iter_idx % print_interval == 0 or iter_idx == max_iters - 1:
            print(f"K-Subspaces EM: {iter_idx}/{max_iters}")
        # Assignment step
        labels, reconstruction_errors = assign_to_subspaces(vectors, subspace_bases)

        if lock_mode == "fixed" and locked_assignments:
            for idx, cluster_id in locked_assignments.items():
                labels[idx] = cluster_id

        # Check convergence
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels.copy()

        # Refit step
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_vectors = vectors[cluster_mask]

            if len(cluster_vectors) == 0:
                # Empty cluster; reinitialize randomly
                random_idx = np.random.randint(n_vectors)
                basis = orth(vectors[random_idx:random_idx+1].T)
                subspace_bases[cluster_id] = basis
            else:
                # Determine rank for this cluster
                if subspace_rank is None:
                    # Auto-detect rank using SVD
                    rank_est = estimate_subspace_rank_svd(
                        cluster_vectors,
                        variance_threshold=variance_threshold,
                        gap_threshold=gap_threshold,
                    )
                    cluster_rank = rank_est.final_rank
                    rank_details[cluster_id] = rank_est
                else:
                    cluster_rank = min(subspace_rank, len(cluster_vectors))

                # Fit subspace
                basis = fit_subspace_qr(cluster_vectors, cluster_rank)
                subspace_bases[cluster_id] = basis

    # Final assignment and error computation
    labels, reconstruction_errors = assign_to_subspaces(vectors, subspace_bases)
    if lock_mode == "fixed" and locked_assignments:
        for idx, cluster_id in locked_assignments.items():
            labels[idx] = cluster_id
        # Recompute errors with locked assignments reflected
        labels, reconstruction_errors = assign_to_subspaces(vectors, subspace_bases)
    total_error = sum(reconstruction_errors.values())

    # Collect actual ranks used per cluster
    cluster_ranks = {cluster_id: basis.shape[1] for cluster_id, basis in subspace_bases.items()}

    result = SubspaceClusterResult(
        cluster_labels=labels,
        subspace_bases=subspace_bases,
        reconstruction_errors=reconstruction_errors,
        total_reconstruction_error=total_error,
        n_clusters=n_clusters,
        subspace_rank=subspace_rank,
        method="k_subspaces",
        cluster_ranks=cluster_ranks,
        rank_estimation_details=rank_details if rank_details else None,
    )
    result.initial_clusters = {cid: seed_clusters[cid].tolist() for cid in seed_clusters}
    return result


def elastic_net_subspace_clustering(
    vectors: np.ndarray,
    n_clusters: int,
    lambda_1: float = 0.01,
    lambda_2: float = 0.001,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Elastic-Net Subspace Clustering (ENSC).

    Args:
        vectors: (n_vectors, d_features) normalized array
        n_clusters: Number of clusters for final spectral clustering
        lambda_1: L1 regularization weight
        lambda_2: L2 regularization weight
        random_state: Random seed

    Returns:
        labels: (n_vectors,) cluster assignments
        affinity: (n_vectors, n_vectors) symmetric affinity matrix
    """
    n_vectors = len(vectors)
    C = np.zeros((n_vectors, n_vectors))

    # Solve elastic net for each point
    for i in tqdm(range(n_vectors), desc="ENSC Elastic Net"):
        # Build dictionary excluding current point
        X_minus_i = np.delete(vectors, i, axis=0)
        y_i = vectors[i]

        # Solve: min (1/2)||y - Xc||^2 + lambda_1||c||_1 + (lambda_2/2)||c||_2^2
        model = ElasticNet(
            alpha=lambda_1 + lambda_2,
            l1_ratio=lambda_1 / (lambda_1 + lambda_2) if (lambda_1 + lambda_2) > 0 else 0.5,
            fit_intercept=False,
            max_iter=5000,
        )
        model.fit(X_minus_i.T, y_i)

        # Insert coefficients (excluding self)
        c_i = model.coef_
        C[i, :i] = c_i[:i]
        C[i, i+1:] = c_i[i:]

    # Build symmetric affinity
    affinity = np.abs(C) + np.abs(C.T)

    # Spectral clustering
    labels = spectral_clustering(
        affinity,
        n_clusters=n_clusters,
        random_state=random_state,
        assign_labels='kmeans',
    )

    return labels, affinity


def ensc_clustering(
    vectors: np.ndarray,
    n_clusters: int | None = None,
    subspace_rank: int | None = None,
    lambda_1: float = 0.01,
    lambda_2: float = 0.001,
    random_state: int | None = None,
    max_clusters: int = 10,
    variance_threshold: float = 0.95,
    gap_threshold: float = 2.0,
) -> SubspaceClusterResult:
    """Full ENSC pipeline: solve elastic net + spectral cluster + fit subspaces.

    Args:
        vectors: (n_vectors, d_features) normalized array
        n_clusters: Number of subspaces. If None, auto-detected from eigengap
        subspace_rank: Rank of each subspace. If None, auto-estimated per cluster
        lambda_1: L1 regularization weight
        lambda_2: L2 regularization weight
        random_state: Random seed
        max_clusters: Maximum K when auto-detecting (only used if n_clusters=None)
        variance_threshold: Variance threshold for auto rank selection (only used if subspace_rank=None)
        gap_threshold: Gap threshold for auto rank selection (only used if subspace_rank=None)

    Returns:
        SubspaceClusterResult
    """
    n_vectors = len(vectors)

    # First pass: run ENSC with a temporary n_clusters to build affinity
    if n_clusters is None:
        # Use a reasonable default for building affinity matrix
        temp_n_clusters = min(int(np.sqrt(n_vectors)), max_clusters, n_vectors)
        temp_n_clusters = max(2, temp_n_clusters)
    else:
        temp_n_clusters = n_clusters

    # Run ENSC to get affinity and initial labels
    labels, affinity = elastic_net_subspace_clustering(
        vectors,
        temp_n_clusters,
        lambda_1,
        lambda_2,
        random_state,
    )

    # If n_clusters was None, refine using eigengap on affinity
    if n_clusters is None:
        n_clusters = estimate_n_clusters_eigengap(affinity, max_clusters)
        # Re-run spectral clustering with refined n_clusters
        labels = spectral_clustering(
            affinity,
            n_clusters=n_clusters,
            random_state=random_state,
            assign_labels='kmeans',
        )

    # Fit subspaces to each cluster
    subspace_bases: Dict[int, np.ndarray] = {}
    reconstruction_errors: Dict[int, float] = {}
    rank_details: Dict[int, RankEstimationDetails] = {}

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_vectors = vectors[cluster_mask]

        if len(cluster_vectors) == 0:
            continue

        # Determine rank for this cluster
        if subspace_rank is None:
            # Auto-detect rank using SVD
            rank_est = estimate_subspace_rank_svd(
                cluster_vectors,
                variance_threshold=variance_threshold,
                gap_threshold=gap_threshold,
            )
            cluster_rank = rank_est.final_rank
            rank_details[cluster_id] = rank_est
        else:
            cluster_rank = min(subspace_rank, len(cluster_vectors))

        # Fit subspace
        basis = fit_subspace_qr(cluster_vectors, cluster_rank)
        subspace_bases[cluster_id] = basis

        # Compute reconstruction error
        projections = cluster_vectors @ basis @ basis.T
        residuals = cluster_vectors - projections
        reconstruction_errors[cluster_id] = float(np.mean(np.sum(residuals ** 2, axis=1)))

    total_error = sum(reconstruction_errors.values())

    # Collect actual ranks used per cluster
    cluster_ranks = {cluster_id: basis.shape[1] for cluster_id, basis in subspace_bases.items()}

    return SubspaceClusterResult(
        cluster_labels=labels,
        subspace_bases=subspace_bases,
        reconstruction_errors=reconstruction_errors,
        total_reconstruction_error=total_error,
        n_clusters=n_clusters,
        subspace_rank=subspace_rank,
        method="ensc",
        cluster_ranks=cluster_ranks,
        rank_estimation_details=rank_details if rank_details else None,
    )


def compute_principal_angles(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
) -> np.ndarray:
    """Compute principal angles between two subspaces.

    Args:
        basis_a: (d_features, rank_a) orthonormal basis
        basis_b: (d_features, rank_b) orthonormal basis

    Returns:
        angles: Principal angles in radians, sorted ascending
    """
    return subspace_angles(basis_a, basis_b)


def compute_all_principal_angles(
    subspace_bases: Dict[int, np.ndarray],
) -> Dict[Tuple[int, int], np.ndarray]:
    """Compute principal angles between all pairs of subspaces.

    Args:
        subspace_bases: Dict mapping cluster_id -> basis matrix

    Returns:
        Dict mapping (cluster_i, cluster_j) -> principal angles array
    """
    cluster_ids = sorted(subspace_bases.keys())
    angles_dict: Dict[Tuple[int, int], np.ndarray] = {}

    for i, id_a in enumerate(cluster_ids):
        for id_b in cluster_ids[i+1:]:
            angles = compute_principal_angles(
                subspace_bases[id_a],
                subspace_bases[id_b],
            )
            angles_dict[(id_a, id_b)] = angles

    return angles_dict


def compute_projection_energies(
    vectors: np.ndarray,
    labels: np.ndarray,
    subspace_bases: Dict[int, np.ndarray],
) -> Tuple[Dict[int, float], float]:
    """Compute within-cluster and between-cluster projection energies.

    Args:
        vectors: (n_vectors, d_features) array
        labels: (n_vectors,) cluster assignments
        subspace_bases: Dict mapping cluster_id -> basis matrix

    Returns:
        within_energies: Dict mapping cluster_id -> mean squared projection norm
        between_energy: Mean squared cross-projection norm
    """
    within_energies: Dict[int, float] = {}
    between_contributions = []

    for cluster_id, basis in subspace_bases.items():
        cluster_mask = labels == cluster_id
        cluster_vectors = vectors[cluster_mask]

        if len(cluster_vectors) == 0:
            continue

        # Within: ||U U^T x||^2 for x in cluster
        projections = cluster_vectors @ basis @ basis.T
        within_energy = float(np.mean(np.sum(projections ** 2, axis=1)))
        within_energies[cluster_id] = within_energy

        # Between: ||U_j U_j^T x_i||^2 for x_i in cluster i, U_j != U_i
        for other_id, other_basis in subspace_bases.items():
            if other_id == cluster_id:
                continue
            cross_projections = cluster_vectors @ other_basis @ other_basis.T
            cross_energy = np.mean(np.sum(cross_projections ** 2, axis=1))
            between_contributions.append(cross_energy)

    between_energy = float(np.mean(between_contributions)) if between_contributions else 0.0

    return within_energies, between_energy


def grid_search_k_subspaces(
    vectors: np.ndarray,
    k_values: Sequence[int],
    r_values: Sequence[int],
    random_state: int | None = None,
    bic_penalty_weight: float = 0.1,
) -> Tuple[SubspaceClusterResult, Dict]:
    """Grid search over K and r for K-Subspaces clustering.

    Args:
        vectors: (n_vectors, d_features) normalized array
        k_values: Sequence of K values (number of clusters) to try
        r_values: Sequence of r values (subspace rank) to try
        random_state: Random seed
        bic_penalty_weight: Weight for BIC penalty term

    Returns:
        best_result: SubspaceClusterResult with best BIC score
        all_results: Dict mapping (K, r) -> (result, bic_score)
    """
    n_vectors, d_features = vectors.shape
    all_results = {}
    best_result = None
    best_bic = float('inf')

    for K in tqdm(k_values, desc="Grid Search K"):
        for r in r_values:
            if r > d_features or K > n_vectors:
                continue

            result = k_subspaces_clustering(
                vectors,
                n_clusters=K,
                subspace_rank=r,
                max_iters=20,
                random_state=random_state,
            )

            # Compute BIC-like score: reconstruction error + penalty
            # Penalty term: (K * r * d_features) parameters
            n_params = K * r * d_features
            bic_score = result.total_reconstruction_error + bic_penalty_weight * n_params

            all_results[(K, r)] = (result, bic_score)

            if bic_score < best_bic:
                best_bic = bic_score
                best_result = result

    if best_result is None:
        raise ValueError("No valid results found in grid search")

    return best_result, all_results


def add_diagnostics_to_result(
    result: SubspaceClusterResult,
    vectors: np.ndarray,
) -> SubspaceClusterResult:
    """Compute and add diagnostic information to clustering result.

    Args:
        result: SubspaceClusterResult to augment
        vectors: (n_vectors, d_features) original normalized vectors

    Returns:
        Updated result with diagnostics filled in
    """
    # Compute principal angles
    result.principal_angles = compute_all_principal_angles(result.subspace_bases)

    # Compute projection energies
    within, between = compute_projection_energies(
        vectors,
        result.cluster_labels,
        result.subspace_bases,
    )
    result.within_projection_energy = within
    result.between_projection_energy = between

    return result
