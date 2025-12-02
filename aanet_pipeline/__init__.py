"""Utilities for preparing and training AAnet models on SAE reconstructions."""

from .cluster_summary import AAnetDescriptor, load_aanet_summary, parse_aanet_descriptors, build_latent_assignment_list
try:
    from .data_builder import AAnetDatasetResult, build_aanet_datasets
except ImportError:
    # Allow import to proceed if dependencies (simplexity) are missing
    # This enables using other parts of the package (training, extrema)
    pass
from .extrema import ExtremaConfig, compute_diffusion_extrema
from .overlap_filter import drop_overlapping_latents
from .training import TrainingConfig, TrainingResult, train_aanet_model

__all__ = [
    "AAnetDescriptor",
    "AAnetDatasetResult",
    "TrainingConfig",
    "TrainingResult",
    "build_aanet_datasets",
    "build_latent_assignment_list",
    "compute_diffusion_extrema",
    "ExtremaConfig",
    "load_aanet_summary",
    "parse_aanet_descriptors",
    "drop_overlapping_latents",
    "train_aanet_model",
]
