"""Configuration dataclasses for clustering pipeline."""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple


@dataclass
class SpectralParams:
    """Parameters for spectral clustering."""
    sim_metric: Literal["cosine", "euclidean", "phi"] = "cosine"
    max_clusters: int = 12
    min_clusters: int = 2
    plot_eigengap: bool = False
    center_decoder_rows: bool = False


@dataclass
class SubspaceParams:
    """Parameters for k-subspaces clustering."""
    subspace_rank: Optional[int] = None  # None = auto
    n_clusters: Optional[int] = None  # None = auto
    use_grid_search: bool = False
    k_values: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8])
    r_values: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    cosine_dedup_threshold: float = 0.995
    seed_lock_mode: Literal["fixed", "release_after_init"] = "fixed"
    variance_threshold: float = 0.95  # Cumulative variance for auto rank detection
    gap_threshold: float = 2.0  # Singular value gap ratio for auto rank detection


@dataclass
class ENSCParams:
    """Parameters for ENSC clustering."""
    subspace_rank: Optional[int] = None  # None = auto
    n_clusters: Optional[int] = None  # None = auto (eigengap)
    lambda1: float = 0.01
    lambda2: float = 0.001
    cosine_dedup_threshold: float = 0.995
    variance_threshold: float = 0.95  # Cumulative variance for auto rank detection
    gap_threshold: float = 2.0  # Singular value gap ratio for auto rank detection


@dataclass
class BeliefSeedingConfig:
    """Configuration for belief-aligned seeding."""
    enabled: bool = False
    ridge_alpha: float = 1e-3
    lasso_alpha: Optional[float] = None  # None = cross-validated
    lasso_cv: int = 5
    lasso_max_iter: int = 5000
    coef_threshold: float = 1e-4
    max_latents: Optional[int] = None
    seed_lock_mode: Literal["fixed", "release_after_init"] = "fixed"
    protect_seed_duplicates: bool = False


@dataclass
class EPDFConfig:
    """Configuration for EPDF generation."""
    enabled: bool = False
    sites: Optional[List[str]] = None  # None = all sites
    plot_mode: Literal["both", "all_only", "per_latent_only"] = "both"
    bandwidth: str = "scott"  # or float
    grid_size: int = 100
    min_samples: int = 20


@dataclass
class SamplingConfig:
    """Configuration for sampling and activation collection."""
    sample_sequences: int = 1024
    sample_seq_len: Optional[int] = None  # None = model n_ctx
    max_activations: int = 50000
    cluster_activation_threshold: float = 1e-6
    min_cluster_samples: int = 512
    latent_activity_threshold: float = 0.01
    latent_activation_eps: float = 1e-6
    activation_batches: int = 8


@dataclass
class AnalysisConfig:
    """Configuration for cluster analysis (PCA, plots)."""
    pca_components: int = 6
    skip_pca_plots: bool = False
    plot_max_points: int = 4000


@dataclass
class GeometryFittingConfig:
    """Configuration for geometry-guided clustering refinement."""
    enabled: bool = False

    # Soft assignment extraction
    soft_assignment_method: Literal["top_m", "threshold", "both"] = "top_m"
    top_m: int = 3
    soft_threshold: float = 0.1

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
    n_target_samples: int = 1000

    # Cost function
    cost_fn: Literal["cosine", "euclidean"] = "cosine"
    normalize_vectors: bool = True

    # Filtering/refinement
    threshold_mode: Literal["normalized", "raw"] = "normalized"
    per_point_threshold: float = 0.5
    optimal_distortion_threshold: float = 1.0
    filter_metrics: List[str] = field(default_factory=lambda: ["gw_full"])


@dataclass
class ClusteringConfig:
    """Main configuration for clustering pipeline.

    This dataclass encapsulates all parameters needed to run a clustering
    experiment. It's designed to be:
    - Immutable (frozen after creation)
    - Hashable (for caching and parameter search)
    - Serializable (for saving/loading)
    """
    method: Literal["spectral", "k_subspaces", "ensc"]
    site: str
    selected_k: int  # DEPRECATED: use sae_param instead
    seed: int = 0

    # SAE configuration
    sae_type: Literal["top_k", "vanilla", "banded"] = "top_k"
    sae_param: float | Tuple[float, float] = 0.0  # k for top_k, lambda for vanilla, (ls, la) for banded

    # Method-specific params
    spectral_params: SpectralParams = field(default_factory=SpectralParams)
    subspace_params: SubspaceParams = field(default_factory=SubspaceParams)
    ensc_params: ENSCParams = field(default_factory=ENSCParams)

    # Analysis configs
    belief_seeding: BeliefSeedingConfig = field(default_factory=BeliefSeedingConfig)
    epdf_config: EPDFConfig = field(default_factory=EPDFConfig)
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    geometry_fitting_config: GeometryFittingConfig = field(default_factory=GeometryFittingConfig)

    @classmethod
    def from_args(cls, args, site: str, selected_k: int | float | Tuple[float, float]) -> 'ClusteringConfig':
        """Create config from command-line arguments."""
        spectral_params = SpectralParams(
            sim_metric=args.sim_metric,
            max_clusters=args.max_clusters,
            min_clusters=args.min_clusters,
            plot_eigengap=args.plot_eigengap,
            center_decoder_rows=args.center_decoder_rows,
        )

        subspace_params = SubspaceParams(
            subspace_rank=args.subspace_rank,
            n_clusters=args.subspace_n_clusters,
            use_grid_search=args.use_grid_search,
            k_values=args.subspace_k_values,
            r_values=args.subspace_r_values,
            cosine_dedup_threshold=args.cosine_dedup_threshold,
            seed_lock_mode=args.seed_lock_mode,
            variance_threshold=args.subspace_variance_threshold,
            gap_threshold=args.subspace_gap_threshold,
        )

        ensc_params = ENSCParams(
            subspace_rank=args.subspace_rank,
            n_clusters=args.subspace_n_clusters,
            lambda1=args.ensc_lambda1,
            lambda2=args.ensc_lambda2,
            cosine_dedup_threshold=args.cosine_dedup_threshold,
            variance_threshold=args.subspace_variance_threshold,
            gap_threshold=args.subspace_gap_threshold,
        )

        belief_seeding = BeliefSeedingConfig(
            enabled=args.seed_with_beliefs,
            ridge_alpha=args.belief_ridge_alpha,
            lasso_alpha=args.seed_lasso_alpha,
            lasso_cv=args.seed_lasso_cv,
            lasso_max_iter=args.seed_lasso_max_iter,
            coef_threshold=args.seed_coef_threshold,
            max_latents=args.seed_max_latents,
            seed_lock_mode=args.seed_lock_mode,
            protect_seed_duplicates=args.protect_seed_duplicates,
        )

        epdf_config = EPDFConfig(
            enabled=args.build_cluster_epdfs,
            sites=args.epdf_sites,
            plot_mode=args.epdf_plot_mode,
            bandwidth=args.epdf_bandwidth,
            grid_size=args.epdf_grid_size,
            min_samples=args.epdf_min_samples,
        )

        sampling_config = SamplingConfig(
            sample_sequences=args.sample_sequences,
            sample_seq_len=args.sample_seq_len,
            max_activations=args.max_activations,
            cluster_activation_threshold=args.cluster_activation_threshold,
            min_cluster_samples=args.min_cluster_samples,
            latent_activity_threshold=args.latent_activity_threshold,
            latent_activation_eps=args.latent_activation_eps,
            activation_batches=args.activation_batches,
        )

        analysis_config = AnalysisConfig(
            pca_components=args.pca_components,
            skip_pca_plots=args.skip_pca_plots,
            plot_max_points=args.plot_max_points,
        )

        # Geometry fitting config (if enabled)
        geometry_fitting_config = GeometryFittingConfig(
            enabled=getattr(args, 'refine_with_geometries', False),
            soft_assignment_method=getattr(args, 'geo_soft_method', 'top_m'),
            top_m=getattr(args, 'geo_top_m', 3),
            soft_threshold=getattr(args, 'geo_soft_threshold', 0.1),
            simplex_k_range=(getattr(args, 'geo_simplex_k_min', 1), getattr(args, 'geo_simplex_k_max', 8)),
            include_circle=getattr(args, 'geo_include_circle', True),
            include_hypersphere=getattr(args, 'geo_include_hypersphere', False),
            cost_fn=getattr(args, 'geo_cost_fn', 'cosine'),
            gw_epsilon=getattr(args, 'geo_gw_epsilon', 0.1),
            sinkhorn_epsilon=getattr(args, 'geo_sinkhorn_epsilon', 0.1),
            sinkhorn_max_iter=getattr(args, 'geo_sinkhorn_max_iter', 1000),
            threshold_mode=getattr(args, 'geo_threshold_mode', 'normalized'),
            per_point_threshold=getattr(args, 'geo_per_point_threshold', 0.5),
            optimal_distortion_threshold=getattr(args, 'geo_optimal_distortion_threshold', 1.0),
            filter_metrics=getattr(args, 'geo_filter_metrics', ['gw_full']),
        )

        # Extract SAE type and parameter
        sae_type = getattr(args, 'sae_type', 'top_k')
        if sae_type == 'top_k':
            sae_param = float(selected_k)
        elif sae_type == 'banded':
            sae_param = selected_k  # Tuple
        else:  # vanilla
            sae_param = float(selected_k)  # Will be lambda value passed as selected_k

        return cls(
            method=args.clustering_method,
            site=site,
            selected_k=selected_k,  # Keep for backward compatibility
            seed=args.seed,
            sae_type=sae_type,
            sae_param=sae_param,
            spectral_params=spectral_params,
            subspace_params=subspace_params,
            ensc_params=ensc_params,
            belief_seeding=belief_seeding,
            epdf_config=epdf_config,
            sampling_config=sampling_config,
            analysis_config=analysis_config,
            geometry_fitting_config=geometry_fitting_config,
        )
