"""Belief geometry catalog for GMG-based clustering refinement.

This module provides geometric structures (K-simplices, circles, hyperspheres)
that SAE decoder directions may map to. Each geometry can:
- Sample uniformly from its surface
- Report its dimension
- Provide a descriptive name
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BeliefGeometry(ABC):
    """Abstract base class for belief geometries.

    A belief geometry represents a target structure (e.g., K-simplex, circle)
    that we hypothesize SAE latent directions might align with.
    """

    @abstractmethod
    def sample_uniform(self, n_samples: int, seed: int = 42) -> np.ndarray:
        """Sample uniformly from the geometry.

        Args:
            n_samples: Number of samples to draw
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_samples, dimension) with uniform samples
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the ambient dimension of the geometry.

        Returns:
            Dimension of the embedding space
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get descriptive name for this geometry.

        Returns:
            String like 'simplex_3', 'circle', 'sphere_4'
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}', dim={self.get_dimension()})"


class SimplexGeometry(BeliefGeometry):
    """K-simplex geometry.

    A K-simplex has K+1 vertices and lies in K+1 dimensional space.
    Points on the simplex are represented as barycentric coordinates
    that sum to 1 (e.g., probability distributions over K+1 outcomes).

    Examples:
        - K=0: point (1-simplex has 1 vertex)
        - K=1: line segment (2 vertices)
        - K=2: triangle (3 vertices, probability simplex)
        - K=3: tetrahedron (4 vertices)
    """

    def __init__(self, K: int):
        """Initialize K-simplex.

        Args:
            K: Simplex dimension (K+1 vertices)
        """
        if K < 0:
            raise ValueError(f"K must be non-negative, got {K}")
        self.K = K

    def sample_uniform(self, n_samples: int, seed: int = 42) -> np.ndarray:
        """Sample uniformly from K-simplex using Dirichlet distribution.

        The Dirichlet(1, 1, ..., 1) distribution generates uniform samples
        on the probability simplex.

        Args:
            n_samples: Number of samples
            seed: Random seed

        Returns:
            Array of shape (n_samples, K+1) with barycentric coordinates
        """
        rng = np.random.RandomState(seed)
        # Dirichlet with all alphas = 1 gives uniform distribution on simplex
        return rng.dirichlet(np.ones(self.K + 1), size=n_samples)

    def get_dimension(self) -> int:
        """K-simplex lives in (K+1)-dimensional space."""
        return self.K + 1

    def get_name(self) -> str:
        """Return name like 'simplex_2' for a 2-simplex (triangle)."""
        return f"simplex_{self.K}"


class CircleGeometry(BeliefGeometry):
    """Circle (1-sphere) geometry.

    Represents the unit circle in 2D, often used for:
    - Bloch sphere slices (quantum state representations)
    - Cyclic/periodic features
    - Phase-like representations

    Points are represented as (cos(θ), sin(θ)) for θ ∈ [0, 2π).
    """

    def __init__(self, radius: float = 1.0):
        """Initialize circle.

        Args:
            radius: Circle radius (default: 1.0 for unit circle)
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        self.radius = radius

    def sample_uniform(self, n_samples: int, seed: int = 42) -> np.ndarray:
        """Sample uniformly from circle.

        Args:
            n_samples: Number of samples
            seed: Random seed

        Returns:
            Array of shape (n_samples, 2) with points on circle
        """
        rng = np.random.RandomState(seed)
        angles = rng.uniform(0, 2 * np.pi, n_samples)
        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)
        return np.column_stack([x, y])

    def get_dimension(self) -> int:
        """Circle lives in 2D."""
        return 2

    def get_name(self) -> str:
        """Return 'circle' or 'circle_r{radius}' if radius != 1."""
        if self.radius == 1.0:
            return "circle"
        return f"circle_r{self.radius:.2f}"


class HypersphereGeometry(BeliefGeometry):
    """N-sphere geometry (surface of (N+1)-ball).

    Represents the N-dimensional sphere embedded in (N+1)-dimensional space.

    Examples:
        - N=1: Circle (already covered by CircleGeometry)
        - N=2: 2-sphere (surface of 3D ball, like Earth's surface)
        - N=3: 3-sphere (used in Bloch sphere and quantum mechanics)

    Points are sampled uniformly on the sphere surface and normalized to unit length.
    """

    def __init__(self, N: int, radius: float = 1.0):
        """Initialize N-sphere.

        Args:
            N: Sphere dimension (N-sphere in (N+1)-dimensional space)
            radius: Sphere radius (default: 1.0)
        """
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        self.N = N
        self.radius = radius

    def sample_uniform(self, n_samples: int, seed: int = 42) -> np.ndarray:
        """Sample uniformly from N-sphere using Gaussian method.

        Sample from N+1 dimensional Gaussian, then normalize to unit sphere.
        This produces uniform distribution on the sphere surface.

        Args:
            n_samples: Number of samples
            seed: Random seed

        Returns:
            Array of shape (n_samples, N+1) with points on sphere
        """
        rng = np.random.RandomState(seed)
        # Sample from N+1 dimensional Gaussian
        samples = rng.randn(n_samples, self.N + 1)
        # Normalize to unit sphere
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = samples / (norms + 1e-10)  # Avoid division by zero
        # Scale by radius
        return self.radius * samples

    def get_dimension(self) -> int:
        """N-sphere lives in (N+1)-dimensional space."""
        return self.N + 1

    def get_name(self) -> str:
        """Return name like 'sphere_2' for 2-sphere."""
        if self.radius == 1.0:
            return f"sphere_{self.N}"
        return f"sphere_{self.N}_r{self.radius:.2f}"


def create_geometry_catalog(
    simplex_k_range: Optional[tuple] = None,
    include_circle: bool = True,
    include_hypersphere: bool = False,
    hypersphere_dims: Optional[list] = None
) -> list[BeliefGeometry]:
    """Create a catalog of belief geometries to test.

    Args:
        simplex_k_range: Tuple of (K_min, K_max) for simplices. Default: (1, 8)
        include_circle: Whether to include circle geometry
        include_hypersphere: Whether to include hypersphere geometries
        hypersphere_dims: List of N values for N-spheres. Default: [2, 3, 4]

    Returns:
        List of BeliefGeometry instances
    """
    geometries = []

    # Add simplices
    if simplex_k_range is None:
        simplex_k_range = (1, 8)
    K_min, K_max = simplex_k_range
    for K in range(K_min, K_max + 1):
        geometries.append(SimplexGeometry(K))

    # Add circle
    if include_circle:
        geometries.append(CircleGeometry())

    # Add hyperspheres
    if include_hypersphere:
        if hypersphere_dims is None:
            hypersphere_dims = [2, 3, 4]
        for N in hypersphere_dims:
            if N > 1:  # N=1 is just a circle
                geometries.append(HypersphereGeometry(N))

    return geometries
