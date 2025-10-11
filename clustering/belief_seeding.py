"""Belief-aligned seeding for clustering."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from subspace_clustering_utils import _compute_belief_seed_sets

from .config import BeliefSeedingConfig


@dataclass
class BeliefSeedingResult:
    """Result from belief-aligned seeding."""
    seed_clusters_active: Dict[int, List[int]]  # Cluster ID -> list of active indices
    seed_clusters_global: Dict[int, List[int]]  # Cluster ID -> list of global indices
    metadata: Dict[str, Any]
    slice_map: Dict[str, Tuple[int, int]]
    succeeded: bool = True


class BeliefSeeder:
    """Handles belief-aligned seeding for clustering."""

    def __init__(self, config: BeliefSeedingConfig):
        self.config = config

    def compute_seed_clusters(
        self,
        acts_flat: torch.Tensor,
        feature_acts: torch.Tensor,
        decoder_active: np.ndarray,
        active_indices: np.ndarray,
        component_beliefs_flat: Dict[str, np.ndarray],
        component_order: List[str],
        site: str,
        site_idx: int,
    ) -> Optional[BeliefSeedingResult]:
        """Compute belief-aligned seed clusters.

        Args:
            acts_flat: Flattened activations (n_samples, d_model)
            feature_acts: SAE feature activations (n_samples, n_latents)
            decoder_active: Active decoder directions (n_active, d_model)
            active_indices: Indices of active latents in full decoder
            component_beliefs_flat: Dict mapping component name -> beliefs (n_samples, belief_dim)
            component_order: Ordered list of component names
            site: Site name for logging
            site_idx: Site index for random seed

        Returns:
            BeliefSeedingResult if successful, None otherwise
        """
        if not self.config.enabled or not component_beliefs_flat:
            return None

        print(f"{site}: doing belief-aligned seeding")

        # Get active features
        active_idx_tensor = torch.from_numpy(active_indices).to(feature_acts.device)
        feature_active = feature_acts[:, active_idx_tensor]

        try:
            seeds_active_map, seed_metadata, residual_map, residual_intercept, slice_map = _compute_belief_seed_sets(
                acts_flat,
                feature_active,
                decoder_active,
                component_beliefs_flat,
                ridge_alpha=self.config.ridge_alpha,
                lasso_alpha=self.config.lasso_alpha,
                lasso_cv=self.config.lasso_cv,
                lasso_max_iter=self.config.lasso_max_iter,
                coef_threshold=self.config.coef_threshold,
                max_latents=self.config.max_latents,
                random_state=site_idx,
            )
        except Exception as exc:
            print(f"{site}: belief-aligned seeding failed ({exc})")
            return None

        if not seeds_active_map:
            return None

        # Resolve conflicts: assign each latent to its best component
        latent_best_component: Dict[int, Tuple[str, float]] = {}
        for comp_name, meta in seed_metadata.items():
            for latent_idx, score in meta.get("scores", {}).items():
                latent_idx = int(latent_idx)
                score_val = float(score)
                best = latent_best_component.get(latent_idx)
                if best is None or score_val > best[1]:
                    latent_best_component[latent_idx] = (comp_name, score_val)

        resolved_seed_map: Dict[str, List[int]] = {}
        for comp_name in component_order:
            candidates = seeds_active_map.get(comp_name, [])
            resolved: List[int] = []
            for idx in candidates:
                best = latent_best_component.get(idx)
                if best is None or best[0] != comp_name:
                    continue
                if idx not in resolved:
                    resolved.append(idx)
            if resolved:
                resolved_seed_map[comp_name] = resolved

        if not resolved_seed_map:
            print(f"{site}: belief-aligned seeding produced no confident latents")
            return None

        # Build cluster maps
        cluster_component_map: Dict[int, str] = {}
        seed_clusters_active: Dict[int, List[int]] = {}
        seed_clusters_global: Dict[int, List[int]] = {}

        cluster_counter = 0
        for comp_name in component_order:
            indices = resolved_seed_map.get(comp_name, [])
            if not indices:
                continue
            seed_clusters_active[cluster_counter] = indices
            seed_clusters_global[cluster_counter] = [int(active_indices[idx]) for idx in indices]
            cluster_component_map[cluster_counter] = comp_name
            cluster_counter += 1

        # Build metadata
        metadata = {
            "per_component": seed_metadata,
            "resolved": {
                comp: [int(active_indices[idx]) for idx in indices]
                for comp, indices in resolved_seed_map.items()
            },
            "ridge_alpha": float(self.config.ridge_alpha),
            "lasso_alpha_override": None if self.config.lasso_alpha is None else float(self.config.lasso_alpha),
            "cluster_components": {
                int(cid): name for cid, name in cluster_component_map.items()
            },
        }

        return BeliefSeedingResult(
            seed_clusters_active=seed_clusters_active,
            seed_clusters_global=seed_clusters_global,
            metadata=metadata,
            slice_map=slice_map,
            succeeded=True,
        )

    def update_after_deduplication(
        self,
        seed_result: BeliefSeedingResult,
        kept_indices: np.ndarray,
        active_indices: np.ndarray,
        site: str,
    ) -> BeliefSeedingResult:
        """Update seed clusters after deduplication.

        Args:
            seed_result: Original seeding result
            kept_indices: Indices in active space that were kept after dedup
            active_indices: Global active indices
            site: Site name for logging

        Returns:
            Updated BeliefSeedingResult
        """
        # Map from active index to normalized index
        index_map = {int(active_idx): int(norm_idx) for norm_idx, active_idx in enumerate(kept_indices)}

        updated_active: Dict[int, List[int]] = {}
        updated_global: Dict[int, List[int]] = {}

        for cluster_id, active_list in seed_result.seed_clusters_active.items():
            normalized_list = [index_map[int(idx)] for idx in active_list if int(idx) in index_map]
            if not normalized_list:
                continue
            updated_active[cluster_id] = normalized_list
            updated_global[cluster_id] = [int(active_indices[idx]) for idx in active_list if int(idx) in index_map]

        if not updated_active:
            print(f"{site}: belief-aligned seeds were removed during deduplication")
            return BeliefSeedingResult(
                seed_clusters_active={},
                seed_clusters_global={},
                metadata=seed_result.metadata,
                slice_map=seed_result.slice_map,
                succeeded=False,
            )

        # Update metadata
        updated_metadata = seed_result.metadata.copy()
        updated_metadata["post_dedup"] = {
            int(cid): latents for cid, latents in updated_global.items()
        }

        return BeliefSeedingResult(
            seed_clusters_active=updated_active,
            seed_clusters_global=updated_global,
            metadata=updated_metadata,
            slice_map=seed_result.slice_map,
            succeeded=True,
        )

    def report_seed_retention(
        self,
        seed_clusters_global: Dict[int, List[int]],
        cluster_labels_full: np.ndarray,
        site: str,
    ) -> None:
        """Report how well seeds were retained during clustering.

        Args:
            seed_clusters_global: Seed clusters in global index space
            cluster_labels_full: Full cluster labels array
            site: Site name for logging
        """
        from collections import Counter

        for cid in sorted(seed_clusters_global.keys()):
            seeded_latents = seed_clusters_global[cid]
            label_counts = Counter()
            for latent_idx in seeded_latents:
                if 0 <= latent_idx < len(cluster_labels_full):
                    label_counts[cluster_labels_full[latent_idx]] += 1

            assigned = label_counts.get(cid, 0)
            total_seeded = len(seeded_latents)
            reassigned = {
                int(k): int(v)
                for k, v in label_counts.items()
                if k != cid and v > 0
            }

            if reassigned:
                print(
                    f"{site}: belief-seeded cluster {cid} retained {assigned}/{total_seeded} "
                    f"preselected latents (reassigned: {reassigned})"
                )
            else:
                print(
                    f"{site}: belief-seeded cluster {cid} retained {assigned}/{total_seeded} "
                    f"preselected latents"
                )
