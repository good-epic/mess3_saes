"""Modular clustering package for SAE analysis.

This package provides a modular architecture for clustering SAE decoder directions
and analyzing the resulting clusters. It supports:
- Multiple clustering methods (spectral, k-subspaces, ENSC)
- Belief-aligned seeding
- Geometry-guided refinement via Gromov-Monge Gap
- Pluggable metric evaluation (GMG, distortion, R², etc.)
- Parameter search-friendly design
"""

from .config import ClusteringConfig, SpectralParams, SubspaceParams, ENSCParams, GeometryFittingConfig, CooccurrenceConfig
from .results import ClusteringResult
from .strategies import ClusteringStrategy, SpectralClusteringStrategy, KSubspacesClusteringStrategy, ENSCClusteringStrategy
from .affinity_metrics import (
    COOCCURRENCE_METRICS,
    GEOMETRY_METRICS,
    CooccurrenceStats,
    collect_cooccurrence_stats,
    cooccurrence_affinity_from_stats,
    build_affinity_matrix,
)
try:
    from .pipeline import SiteClusteringPipeline
except ImportError:
    pass
from .metrics import MetricEvaluator, GMGMetricEvaluator, BeliefR2Evaluator, evaluate_clustering_metrics
from .geometries import BeliefGeometry, SimplexGeometry, CircleGeometry, HypersphereGeometry, create_geometry_catalog
from .geometry_fitting import GeometryFitter, GeometryRefinementPipeline, GeometryFitResult, RefinedClusteringResult

__all__ = [
    'ClusteringConfig',
    'SpectralParams',
    'SubspaceParams',
    'ENSCParams',
    'GeometryFittingConfig',
    'CooccurrenceConfig',
    'ClusteringResult',
    # Affinity metrics
    'COOCCURRENCE_METRICS',
    'GEOMETRY_METRICS',
    'CooccurrenceStats',
    'collect_cooccurrence_stats',
    'cooccurrence_affinity_from_stats',
    'build_affinity_matrix',
    'ClusteringStrategy',
    'SpectralClusteringStrategy',
    'KSubspacesClusteringStrategy',
    'ENSCClusteringStrategy',
    'SiteClusteringPipeline',
    'MetricEvaluator',
    'GMGMetricEvaluator',
    'BeliefR2Evaluator',
    'evaluate_clustering_metrics',
    'BeliefGeometry',
    'SimplexGeometry',
    'CircleGeometry',
    'HypersphereGeometry',
    'create_geometry_catalog',
    'GeometryFitter',
    'GeometryRefinementPipeline',
    'GeometryFitResult',
    'RefinedClusteringResult',
]
