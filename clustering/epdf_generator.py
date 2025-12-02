"""EPDF generation and visualization for clusters."""

import os
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# from epdf_utils import (
#     build_epdfs_from_sae_and_beliefs,
#     plot_epdfs_to_directory,
# )

from .config import EPDFConfig


class EPDFGenerator:
    """Handles EPDF generation and cosine similarity analysis."""

    def __init__(self, config: EPDFConfig):
        self.config = config

    def should_generate_for_site(self, site: str) -> bool:
        """Check if EPDFs should be generated for this site."""
        if not self.config.enabled:
            return False
        if self.config.sites is None:
            return True
        return site in self.config.sites

    def generate_cluster_epdfs(
        self,
        site: str,
        site_selected_k: int,
        site_dir: str,
        sae,
        acts_flat: torch.Tensor,
        cluster_labels: np.ndarray,
        n_clusters: int,
        component_beliefs_flat: Dict[str, np.ndarray],
        component_metadata: Dict[str, Dict[str, Any]],
        component_order: List[str],
        clustering_method: str,
        sae_type: str = "top_k",
        sae_param: float = 0.0,
        decoder_normalized: Optional[np.ndarray] = None,
        normalized_to_full_idx: Optional[np.ndarray] = None,
        decoder_dirs: Optional[np.ndarray] = None,
    ) -> Dict[int, str]:
        """Generate EPDFs for all clusters at a site.

        Args:
            site: Site name
            site_selected_k: Selected k for SAE (DEPRECATED, use sae_param)
            site_dir: Output directory
            sae: SAE model
            acts_flat: Flattened activations
            cluster_labels: Cluster labels for all latents
            n_clusters: Number of clusters
            component_beliefs_flat: Component beliefs
            component_metadata: Component metadata
            component_order: Ordered component names
            clustering_method: Clustering method name
            sae_type: SAE type ("top_k" or "vanilla")
            sae_param: SAE parameter (k for top_k, lambda for vanilla)
            decoder_normalized: Normalized decoder (for subspace methods)
            normalized_to_full_idx: Mapping from normalized to full indices
            decoder_dirs: Full decoder directions (for spectral)

        Returns:
            Dict mapping cluster_id -> output directory path
        """
        if not self.should_generate_for_site(site):
            return {}

        if component_metadata is None:
            print(f"{site}: skipping cluster EPDFs (not a multipartite sampler)")
            return {}

        print(f"{site}: generating cluster EPDFs")

        # Convert bandwidth
        try:
            epdf_bw = float(self.config.bandwidth)
        except ValueError:
            epdf_bw = self.config.bandwidth

        epdf_paths = {}

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_latent_indices = np.where(cluster_mask)[0].tolist()

            if not cluster_latent_indices:
                continue

            print(f"{site}: cluster {cluster_id} - building EPDFs for {len(cluster_latent_indices)} latents")

            try:
                # Construct SAE ID based on type
                if sae_type == "top_k":
                    sae_id = ("top_k", f"k{int(sae_param)}")
                else:  # vanilla
                    sae_id = ("vanilla", f"lambda_{sae_param}")

                try:
                    from epdf_utils import build_epdfs_from_sae_and_beliefs, plot_epdfs_to_directory
                except ImportError:
                    print(f"{site}: skipping EPDFs (epdf_utils/simplexity not available)")
                    return {}

                epdfs = build_epdfs_from_sae_and_beliefs(
                    site_name=site,
                    sae_id=sae_id,
                    sae=sae,
                    activations=acts_flat,
                    component_beliefs=component_beliefs_flat,
                    component_metadata=component_metadata,
                    latent_indices=cluster_latent_indices,
                    activation_threshold=1e-6,  # Could be configurable
                    min_active_samples=self.config.min_samples,
                    bw_method=epdf_bw,
                    progress=True,
                    progress_desc=f"{site} cluster {cluster_id}",
                )

                # Print activation statistics
                print(f"{site} cluster {cluster_id} activation statistics:")
                for latent_idx, epdf in epdfs.items():
                    activation_frac = epdf.activation_fraction
                    n_samples_used = int(activation_frac * len(acts_flat))
                    print(f"  Latent {latent_idx}: {activation_frac:.1%} active ({n_samples_used} samples)")

                # Compute cosine similarities within cluster
                self._compute_cluster_similarities(
                    cluster_id,
                    cluster_latent_indices,
                    site,
                    decoder_normalized,
                    normalized_to_full_idx,
                    decoder_dirs,
                )

                # Plot EPDFs
                epdf_cluster_dir = os.path.join(
                    site_dir, "epdfs", clustering_method, f"cluster_{cluster_id}"
                )
                plot_epdfs_to_directory(
                    epdfs,
                    epdf_cluster_dir,
                    component_order,
                    plot_mode=self.config.plot_mode,
                    grid_size=self.config.grid_size,
                    title_prefix=f"{site} cluster {cluster_id}: ",
                )
                print(f"{site}: cluster {cluster_id} EPDFs saved to {epdf_cluster_dir}")
                epdf_paths[cluster_id] = epdf_cluster_dir

            except Exception as exc:
                print(f"{site}: failed to generate EPDFs for cluster {cluster_id}: {exc}")

        return epdf_paths

    def _compute_cluster_similarities(
        self,
        cluster_id: int,
        cluster_latent_indices: List[int],
        site: str,
        decoder_normalized: Optional[np.ndarray],
        normalized_to_full_idx: Optional[np.ndarray],
        decoder_dirs: Optional[np.ndarray],
    ) -> None:
        """Compute and report cosine similarities within cluster."""
        if len(cluster_latent_indices) <= 1:
            return

        if decoder_normalized is not None and normalized_to_full_idx is not None:
            # Subspace clustering: use only non-deduplicated vectors
            kept_mask = np.isin(cluster_latent_indices, normalized_to_full_idx)
            kept_in_cluster = np.array(cluster_latent_indices)[kept_mask]

            if len(kept_in_cluster) > 1:
                # Map full indices to normalized indices
                full_to_normalized = {full_idx: norm_idx for norm_idx, full_idx in enumerate(normalized_to_full_idx)}
                normalized_indices = [full_to_normalized[idx] for idx in kept_in_cluster]

                # Compute similarities
                cluster_decoders = decoder_normalized[normalized_indices]
                cos_sim = cosine_similarity(cluster_decoders)

                self._report_similarities(cluster_id, kept_in_cluster, cos_sim, is_deduplicated=True)
            else:
                print(f"  Cluster {cluster_id} has only {len(kept_in_cluster)} non-deduplicated vector(s)")

        elif decoder_dirs is not None:
            # Spectral clustering: use decoder_dirs directly
            cluster_decoders = decoder_dirs[cluster_latent_indices]
            cos_sim = cosine_similarity(cluster_decoders)

            self._report_similarities(cluster_id, cluster_latent_indices, cos_sim, is_deduplicated=False)

    def _report_similarities(
        self,
        cluster_id: int,
        indices: List[int],
        cos_sim: np.ndarray,
        is_deduplicated: bool,
    ) -> None:
        """Report top similarities and warnings."""
        # Build list of all pairs
        all_pairs = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]
                all_pairs.append((idx_i, idx_j, cos_sim[i, j]))

        # Sort by similarity descending and show top 5
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        suffix = " (non-deduplicated vectors only)" if is_deduplicated else ""
        print(f"  Top 5 cosine similarities within cluster {cluster_id}{suffix}:")
        for idx_i, idx_j, sim in all_pairs[:5]:
            print(f"    Latent {idx_i} & {idx_j}: {sim:.4f}")

        # Warn about suspicious pairs > 0.99
        suspicious_pairs = [(i, j, s) for i, j, s in all_pairs if s > 0.99]
        if suspicious_pairs:
            print(f"  WARNING: Found {len(suspicious_pairs)} pairs with similarity > 0.99:")
            for idx_i, idx_j, sim in suspicious_pairs:
                print(f"    Latent {idx_i} & {idx_j}: {sim:.4f}")
