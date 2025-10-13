"""Result dataclasses for clustering pipeline."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .config import ClusteringConfig


@dataclass
class ClusteringResult:
    """Self-contained clustering result.

    This dataclass stores all outputs from a clustering run and provides
    methods for saving/loading. It's designed to be:
    - Comparable (via metrics)
    - Serializable (can save/load from disk)
    - Self-documenting (includes config that produced it)
    """
    config: ClusteringConfig
    site: str
    selected_k: int

    # Clustering outputs
    cluster_labels: np.ndarray  # Full array including inactive latents
    n_clusters: int
    active_indices: np.ndarray
    inactive_indices: np.ndarray

    # Activity statistics
    activity_rates: np.ndarray
    mean_abs_activation: np.ndarray
    total_activity_samples: int

    # Cluster statistics
    cluster_stats: Dict[int, Dict[str, Any]]

    # Metrics (populated by evaluators)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Optional components
    belief_seed_metadata: Optional[Dict[str, Any]] = None
    belief_r2_summary: Optional[Dict[int, Dict[str, Any]]] = None
    belief_r2_summary_soft: Optional[Dict[int, Dict[str, Any]]] = None
    belief_r2_summary_refined: Optional[Dict[int, Dict[str, Any]]] = None
    component_assignment: Optional[Dict[str, Any]] = None
    component_assignment_soft: Optional[Dict[str, Any]] = None
    component_assignment_refined: Optional[Dict[str, Any]] = None
    coherence_metrics: Optional[Dict[str, Any]] = None
    coherence_metrics_soft: Optional[Dict[str, Any]] = None
    subspace_diagnostics: Optional[Dict[str, Any]] = None
    pca_results: Optional[Dict[int, Any]] = None

    # EPDF results
    epdf_paths: Dict[int, str] = field(default_factory=dict)

    # Geometry refinement results (optional)
    geometry_refinement: Optional[Any] = None  # RefinedClusteringResult

    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add evaluation metrics to this result."""
        self.metrics.update(metrics)

    def add_geometry_refinement(self, refinement_result: Any) -> None:
        """Add geometry refinement results.

        Args:
            refinement_result: RefinedClusteringResult from geometry fitting
        """
        self.geometry_refinement = refinement_result

    def to_metadata_dict(self, process_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to metadata dictionary for saving."""
        metadata = {
            "site": self.site,
            "spectral_clusters": int(self.n_clusters),
            "selected_k": int(self.selected_k),
            "clustering_method": self.config.method,
            "n_activations_used": len(self.active_indices),
            "total_activity_samples": int(self.total_activity_samples),

            # SAE info
            "sae_type": self.config.sae_type,
            "sae_param": float(self.config.sae_param),

            # Activity stats
            "latent_activity_rates": {int(i): float(v) for i, v in enumerate(self.activity_rates)},
            "mean_abs_activation": {int(i): float(v) for i, v in enumerate(self.mean_abs_activation)},
            "active_latent_indices": [int(i) for i in self.active_indices.tolist()],
            "inactive_latent_indices": [int(i) for i in self.inactive_indices.tolist()],

            # Config params
            "cluster_activation_threshold": float(self.config.sampling_config.cluster_activation_threshold),
            "min_cluster_samples": int(self.config.sampling_config.min_cluster_samples),
            "latent_activity_threshold": float(self.config.sampling_config.latent_activity_threshold),
            "latent_activation_eps": float(self.config.sampling_config.latent_activation_eps),

            # Process info
            "process_kind": process_info.get("process_kind"),
            "components": process_info.get("components"),
            "process_config": process_info.get("process_config"),

            # Cluster stats
            "clusters": self.cluster_stats,
            "cluster_labels": [int(x) for x in self.cluster_labels.tolist()],

            # Metrics
            "metrics": self.metrics,
        }

        # Add optional components
        if self.config.method == "spectral":
            metadata["decoder_rows_centered"] = bool(self.config.spectral_params.center_decoder_rows)

        if self.belief_seed_metadata is not None:
            metadata["belief_seed_metadata"] = self.belief_seed_metadata
            metadata["belief_seeding_used"] = True
        else:
            metadata["belief_seeding_used"] = False

        if self.belief_r2_summary is not None:
            metadata["belief_cluster_r2_hard"] = self.belief_r2_summary

        if self.belief_r2_summary_soft is not None:
            metadata["belief_cluster_r2_soft"] = self.belief_r2_summary_soft

        if self.belief_r2_summary_refined is not None:
            metadata["belief_cluster_r2_refined"] = self.belief_r2_summary_refined

        if self.component_assignment is not None:
            metadata["component_assignment_hard"] = self.component_assignment

        if self.component_assignment_soft is not None:
            metadata["component_assignment_soft"] = self.component_assignment_soft

        if self.component_assignment_refined is not None:
            metadata["component_assignment_refined"] = self.component_assignment_refined

        if self.coherence_metrics is not None:
            metadata["coherence_metrics_hard"] = self.coherence_metrics

        if self.coherence_metrics_soft is not None:
            metadata["coherence_metrics_soft"] = self.coherence_metrics_soft

        if self.subspace_diagnostics is not None:
            metadata.update(self.subspace_diagnostics)

        if self.pca_results is not None:
            clusters_with_pca = list(self.pca_results.keys())
            metadata["clusters_with_pca"] = clusters_with_pca

        if self.geometry_refinement is not None:
            # Add comprehensive geometry refinement data
            metadata["geometry_refinement"] = {
                "enabled": True,
                "n_noise_latents": int(self.geometry_refinement.noise_mask.sum()),
                "config": {
                    "soft_assignment_method": self.geometry_refinement.config.soft_assignment_method,
                    "threshold_mode": self.geometry_refinement.config.threshold_mode,
                    "per_point_threshold": float(self.geometry_refinement.config.per_point_threshold),
                    "optimal_distortion_threshold": float(self.geometry_refinement.config.optimal_distortion_threshold),
                    "filter_metrics": self.geometry_refinement.config.filter_metrics,
                },
                "clusters": {
                    int(cid): fit.to_summary_dict()
                    for cid, fit in self.geometry_refinement.cluster_fits.items()
                }
            }
        else:
            metadata["geometry_refinement"] = {"enabled": False}

        return metadata

    def save_to_directory(
        self,
        site_dir: str,
        process_info: Dict[str, Any],
        l2_summary: Dict[str, Any]
    ) -> None:
        """Save clustering result to directory.

        Args:
            site_dir: Directory to save results
            process_info: Process configuration info
            l2_summary: L2 loss summary
        """
        os.makedirs(site_dir, exist_ok=True)

        # Generate PCA plots if data is available
        if self.pca_results and hasattr(self, '_pca_plot_data') and self._pca_plot_data:
            from .analysis import ClusterAnalyzer
            analyzer = ClusterAnalyzer(self.config.analysis_config)

            analyzer.generate_pca_plots(
                pca_results=self.pca_results,
                site=self.site,
                site_selected_k=self.selected_k,
                site_dir=site_dir,
                seed=self._pca_plot_data.get('seed', 0),
            )

        # Generate EPDFs if data is available
        if hasattr(self, '_epdf_data') and self._epdf_data:
            from .epdf_generator import EPDFGenerator
            epdf_generator = EPDFGenerator(self.config.epdf_config)

            self.epdf_paths = epdf_generator.generate_cluster_epdfs(
                site=self.site,
                site_selected_k=self.selected_k,
                site_dir=site_dir,
                sae=self._epdf_data['sae'],
                acts_flat=self._epdf_data['acts_flat'],
                cluster_labels=self.cluster_labels,
                n_clusters=self.n_clusters,
                component_beliefs_flat=self._epdf_data['component_beliefs_flat'],
                component_metadata=self._epdf_data['component_metadata'],
                component_order=self._epdf_data['component_order'],
                clustering_method=self.config.method,
                sae_type=self.config.sae_type,
                sae_param=self.config.sae_param,
                decoder_normalized=self._epdf_data['decoder_normalized'],
                normalized_to_full_idx=self._epdf_data['normalized_to_full_idx'],
                decoder_dirs=self._epdf_data['decoder_dirs'],
            )

        # Build metadata
        metadata = self.to_metadata_dict(process_info)

        # Write cluster summary JSON
        from mess3_gmg_analysis_utils import write_cluster_metadata
        metadata_path = os.path.join(site_dir, "cluster_summary.json")
        write_cluster_metadata(
            metadata_path,
            self.cluster_stats,
            self.selected_k,
            l2_summary,
            cluster_labels=[int(x) for x in self.cluster_labels.tolist()],
            extra_fields=metadata,
        )

    @classmethod
    def load_from_directory(cls, site_dir: str) -> 'ClusteringResult':
        """Load clustering result from directory.

        Note: This is a partial reconstruction - some fields like config
        may need to be inferred from the saved metadata.
        """
        metadata_path = os.path.join(site_dir, "cluster_summary.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # This is a simplified loader - full implementation would need
        # to reconstruct the config from metadata
        raise NotImplementedError("Loading from disk not yet implemented")
