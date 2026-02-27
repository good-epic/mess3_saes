import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import torch
import torch.nn.functional as F
import numpy as np
import jax
import jax.numpy as jnp

from typing import Dict, List, Tuple, Sequence, Mapping, Any, Callable
import itertools
from itertools import combinations
import warnings
import json
from tqdm.auto import tqdm
from collections import defaultdict
import glob
import re

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collections import Counter
import pathlib
import dill
import scipy.stats as st
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from simplexity.generative_processes.torch_generator import generate_data_batch
from BatchTopK.sae import VanillaSAE, TopKSAE
from multipartite_utils import MultipartiteSampler, build_components_from_config

from training_and_analysis_utils import (
    project_simplex3_to_2d,
    _to_numpy_array,
)

def tom_quantum_params_to_bloch(coords: Any) -> np.ndarray:
    """Convert Tom Quantum belief parameters to Bloch sphere coordinates."""

    arr = _to_numpy_array(coords).astype(np.float64)
    if arr.shape[-1] != 3:
        raise ValueError("Expected Tom Quantum params with last dimension 3")
    x = 2.0 * arr[..., 1]
    y = -2.0 * arr[..., 2]
    z = 2.0 * arr[..., 0] - 1.0
    return np.stack([x, y, z], axis=-1)


def sample_residual_stream_activations(model, mess3, seq_len, n_samples: int = 1000, device: str = "cuda"):
    """
    Get residual stream activations samples.
    """
    # Generate a batch for analysis
    key = jax.random.PRNGKey(43)
    key, subkey = jax.random.split(key)
    gen_states = jnp.repeat(mess3.initial_state[None, :], n_samples, axis=0)
    _, inputs, _ = generate_data_batch(gen_states, mess3, n_samples, seq_len, subkey)

    if isinstance(inputs, torch.Tensor):
        tokens = inputs.long().to(device)
    else:
        tokens = torch.from_numpy(np.array(inputs)).long().to(device)

    # Run with cache to get all activations
    logits, cache = model.run_with_cache(tokens)

    # Extract residual stream activations at different layers
    residual_streams = {
        'embeddings': cache['hook_embed'],  # Shape: [batch, seq, d_model]
        'layer_0': cache['blocks.0.hook_resid_post'],  # After first layer
        'layer_1': cache['blocks.1.hook_resid_post'],  # After second layer,
        'layer_2': cache['blocks.2.hook_resid_post'],  # After third layer
        'layer_3': cache['blocks.3.hook_resid_post'],  # After fourth layer
    }
    return residual_streams, tokens



class LatentEPDF:
    def __init__(
        self,
        site_name,
        sae_id,
        latent_idx,
        coords=None,
        weights=None,
        *,
        component_coords: Mapping[str, Any] | None = None,
        component_weights: Mapping[str, Any] | None = None,
        component_info: Mapping[str, Mapping[str, Any]] | None = None,
        activation_fraction: float | None = None,
    ):
        self.site_name = site_name
        self.sae_id = sae_id
        self.latent_idx = latent_idx

        self.coords = None if coords is None else np.asarray(coords)
        self.weights = None if weights is None else np.asarray(weights)
        self.kde = None

        # Optional per-component data for multipartite analysis
        self.component_coords: dict[str, np.ndarray] = {}
        if component_coords:
            self.component_coords = {
                str(name): np.asarray(arr)
                for name, arr in component_coords.items()
            }
        self.component_weights: dict[str, np.ndarray | None] = {}
        if component_weights:
            self.component_weights = {
                str(name): (None if arr is None else np.asarray(arr))
                for name, arr in component_weights.items()
            }
        self.component_info: dict[str, Mapping[str, Any]] = {
            str(name): dict(info)
            for name, info in (component_info or {}).items()
        }
        self.component_kdes: dict[str, Any] = {}
        self.activation_fraction = activation_fraction

        # Detect triangle on init when legacy coords provided
        self.triangle_vertices = self._detect_triangle_vertices()

    def __str__(self):
        return f"({self.site_name}, {self.sae_id[0]}, {self.sae_id[1]}, {self.latent_idx})"

    def __repr__(self):
        return self.__str__()

    def fit_kde(self, bw_method="scott"):
        if self.coords is None or self.coords.shape[0] < 5:
            return None
        self.kde = st.gaussian_kde(self.coords.T, weights=self.weights, bw_method=bw_method)
        return self.kde

    def evaluate_on_grid(self, grid_x, grid_y):
        if self.kde is None:
            self.fit_kde()
        xy = np.vstack([grid_x.ravel(), grid_y.ravel()])
        z = self.kde(xy).reshape(grid_x.shape)
        return z

    def fit_component_kdes(self, bw_method="scott"):
        """Fit a KDE for each component's belief coordinates.

        Returns:
            dict mapping component names to fitted KDE objects
        """
        if not hasattr(self, 'component_kdes'):
            self.component_kdes = {}

        for comp_name, coords in self.component_coords.items():
            if coords is None or coords.shape[0] < 5:
                self.component_kdes[comp_name] = None
                continue
            weights = self.component_weights.get(comp_name)
            try:
                kde = st.gaussian_kde(coords.T, weights=weights, bw_method=bw_method)
                self.component_kdes[comp_name] = kde
            except np.linalg.LinAlgError as exc:
                cov = np.cov(coords.T, aweights=weights, bias=False)
                rank = np.linalg.matrix_rank(cov)
                variances = np.diag(cov)
                print(
                    f"[EPDF] Singular covariance for latent {self.latent_idx} component '{comp_name}' "
                    f"(site={self.site_name}, samples={coords.shape[0]}, rank={rank}, variances={variances})"
                )
                self.component_kdes[comp_name] = None
            except Exception as exc:
                print(
                    f"[EPDF] Failed to fit KDE for latent {self.latent_idx} component '{comp_name}': {exc}"
                )
                self.component_kdes[comp_name] = None

        return self.component_kdes

    def _detect_triangle_vertices(self, tol=0.02, verbose=False):
        """Infer simplex triangle vertices from this latent's coords (robust to PCA noise)."""
        if self.coords is None or self.coords.shape[0] < 3:
            if verbose:
                print("Not enough coordinates to detect triangle vertices.")
            return None

        all_coords = self.coords
        xmin, xmax = np.min(all_coords[:,0]), np.max(all_coords[:,0])
        ymin, ymax = np.min(all_coords[:,1]), np.max(all_coords[:,1])

        if verbose:
            print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

        candidates = []

        # Near xmin
        mask = np.isclose(all_coords[:,0], xmin, atol=tol)
        if mask.any():
            ys = all_coords[mask,1]
            if verbose:
                print(f"Near xmin mask: {mask}, ys: {ys}")
            candidates.append([xmin, ys.min()])
            candidates.append([xmin, ys.max()])

        # Near xmax
        mask = np.isclose(all_coords[:,0], xmax, atol=tol)
        if mask.any():
            ys = all_coords[mask,1]
            if verbose:
                print(f"Near xmax mask: {mask}, ys: {ys}")
            candidates.append([xmax, ys.min()])
            candidates.append([xmax, ys.max()])

        # Near ymin
        mask = np.isclose(all_coords[:,1], ymin, atol=tol)
        if mask.any():
            xs = all_coords[mask,0]
            if verbose:
                print(f"Near ymin mask: {mask}, xs: {xs}")
            candidates.append([xs.min(), ymin])
            candidates.append([xs.max(), ymin])

        # Near ymax
        mask = np.isclose(all_coords[:,1], ymax, atol=tol)
        if mask.any():
            xs = all_coords[mask,0]
            if verbose:
                print(f"Near ymax mask: {mask}, xs: {xs}")
            candidates.append([xs.min(), ymax])
            candidates.append([xs.max(), ymax])

        candidates = np.unique(np.array(candidates), axis=0)
        if verbose:
            print(f"Unique candidate vertices: {candidates}")

        if candidates.shape[0] < 3:
            if verbose:
                print("Not enough unique candidates to form a triangle.")
            return None

        hull = ConvexHull(candidates)
        verts = candidates[hull.vertices]
        if verbose:
            print(f"Convex hull vertices: {verts}")

        # Force to exactly 3 vertices if noisy hull produced >3
        if verts.shape[0] > 3:
            dists = np.sum((verts[:, None, :] - verts[None, :, :])**2, axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            if verbose:
                print(f"Max distance between hull vertices: {i}, {j}")

            def area(p, q, r):
                return abs(0.5 * ((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])))

            best_k, best_area = None, -1
            for k in range(len(verts)):
                if k in (i, j):
                    continue
                A = area(verts[i], verts[j], verts[k])
                if verbose:
                    print(f"Area for triangle ({i}, {j}, {k}): {A}")
                if A > best_area:
                    best_area, best_k = A, k

            verts = np.array([verts[i], verts[j], verts[best_k]])
            if verbose:
                print(f"Selected triangle vertices: {verts}")

        return verts

def save_latent_epdfs(epdfs: Mapping[int, LatentEPDF], path: str | os.PathLike) -> None:
    """Persist a mapping of latent index -> ``LatentEPDF`` via dill."""

    with open(path, "wb") as f:
        dill.dump(epdfs, f)


def load_latent_epdfs(path: str | os.PathLike) -> dict[int, LatentEPDF]:
    """Load previously saved latent EPDFs."""

    with open(path, "rb") as f:
        data = dill.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"EPDF cache at {path} is not a dict; got {type(data)}")
    return data


def build_epdfs_from_sae_and_beliefs(
    site_name: str,
    sae_id: tuple[str, str],
    sae: Any,
    activations: Any,
    component_beliefs: Mapping[str, np.ndarray],
    component_metadata: Mapping[str, Mapping[str, Any]],
    *,
    latent_indices: Sequence[int] | None = None,
    activation_threshold: float = 1e-6,
    min_active_samples: int = 20,
    bw_method: str | float = "scott",
    progress: bool = True,
    progress_desc: str | None = None,
) -> dict[int, LatentEPDF]:
    """High-level wrapper to build EPDFs from SAE and component beliefs.

    Args:
        site_name: Name of the site (e.g., "layer_2")
        sae_id: Tuple of (sae_type, param_token) for identification
        sae: SAE model instance
        activations: Input activations to encode (n_samples, d_model)
        component_beliefs: Dict mapping component names to belief arrays
        component_metadata: Dict mapping component names to metadata dicts
        latent_indices: Indices of latents to process (None = all)
        activation_threshold: Minimum activation magnitude to include samples
        bw_method: KDE bandwidth method ("scott", "silverman", or float)
        progress: Show progress bar
        progress_desc: Description for progress bar

    Returns:
        Dictionary mapping latent indices to LatentEPDF objects
    """
    # Convert activations to tensor if needed
    if isinstance(activations, np.ndarray):
        activations = torch.from_numpy(activations)

    # Get SAE latent activations
    device = next(sae.parameters()).device
    activations = activations.to(device)

    with torch.no_grad():
        sae_out = sae(activations)
        latent_acts = sae_out["feature_acts"].cpu().numpy()

    # Build EPDFs using the core function
    return build_mp_latent_epdfs(
        site_name=site_name,
        sae_id=sae_id,
        latent_activations=latent_acts,
        component_beliefs=component_beliefs,
        component_metadata=component_metadata,
        active_latent_indices=latent_indices,
        activation_threshold=activation_threshold,
        min_active_samples=min_active_samples,
        bw_method=bw_method,
        progress=progress,
        progress_desc=progress_desc,
        progress_disable=False,
    )


def plot_epdfs_to_directory(
    epdfs: dict[int, LatentEPDF] | Sequence[LatentEPDF],
    output_dir: str | os.PathLike,
    component_order: Sequence[str],
    *,
    plot_mode: str = "both",
    grid_size: int = 100,
    title_prefix: str = "",
    mode: str = "transparency",
    marker_size: int = 5,
    base_opacity: float = 0.6,
    legend_outside: bool = False,
) -> None:
    """Plot EPDFs and save to directory.

    Args:
        epdfs: Dictionary or sequence of LatentEPDF objects
        output_dir: Directory to save plots
        component_order: Order of components for subplots
        plot_mode: "both" (all+per-latent), "all_only", or "per_latent_only"
        grid_size: Grid density for KDE evaluation
        title_prefix: Prefix for plot titles
        mode: Visualization mode ("transparency", "contours", or "scatter")
        marker_size: Marker size for scatter/transparency modes
        base_opacity: Base opacity for transparency mode
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to dict if it's a sequence
    if not isinstance(epdfs, dict):
        epdfs_dict = {epdf.latent_idx: epdf for epdf in epdfs}
    else:
        epdfs_dict = epdfs

    if not epdfs_dict:
        print(f"Warning: No EPDFs to plot in {output_dir}")
        return

    epdf_list = list(epdfs_dict.values())

    # Plot "all latents" figure
    if plot_mode in ["both", "all_only"]:
        fig_all = plot_mp_epdfs(
            epdf_list,
            component_order=component_order,
            title=f"{title_prefix}All latents" if title_prefix else "All latents",
            mode=mode,
            grid_size=grid_size,
            marker_size=marker_size,
            base_opacity=base_opacity,
            legend_outside=legend_outside,
        )
        all_path = os.path.join(output_dir, "all_latents.png")
        fig_all.savefig(all_path, dpi=150, bbox_inches='tight')
        plt.close(fig_all)

    # Plot per-latent figures
    if plot_mode in ["both", "per_latent_only"]:
        for latent_idx, epdf in epdfs_dict.items():
            fig_latent = plot_mp_epdfs(
                [epdf],
                component_order=component_order,
                title=f"{title_prefix}Latent {latent_idx}" if title_prefix else f"Latent {latent_idx}",
                mode=mode,
                grid_size=grid_size,
                marker_size=marker_size,
                base_opacity=base_opacity,
                legend_outside=legend_outside,
            )
            latent_path = os.path.join(output_dir, f"latent_{latent_idx}.png")
            fig_latent.savefig(latent_path, dpi=150, bbox_inches='tight')
            plt.close(fig_latent)


def _plot_hull_and_triangles(hull_pts, all_candidates, epdfs, sae_param, sae_type):
    plt.figure(figsize=(6, 6))
    plt.scatter(hull_pts[:, 0], hull_pts[:, 1], color='black', label='Hull Points')
    for i in range(hull_pts.shape[0]):
        plt.text(hull_pts[i, 0], hull_pts[i, 1], str(i), color='black', fontsize=10)

    # Draw each original triangle from all_candidates
    for tri in all_candidates:
        tri = np.asarray(tri)
        if tri.shape[0] == 3:
            closed_tri = np.vstack([tri, tri[0]])  # close the triangle
            plt.plot(closed_tri[:, 0], closed_tri[:, 1], alpha=0.7)

    plt.title(f"Original triangles from all_candidates (layer {epdfs[0].site_name[6:]}, param={sae_param}, type={sae_type})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()


def get_global_triangle_vertices(epdfs):
    """
    Build a global triangle that covers all EPDFs from the same SAE by taking the
    convex hull of all per-EPDF triangle vertices and selecting the maximum-area triangle
    from that hull.
    """
    all_candidates = []
    for e in epdfs:
        tv = getattr(e, "triangle_vertices", None)
        if tv is None:
            tv = e._detect_triangle_vertices(verbose=False)
        if tv is None:
            e_id = getattr(e, "sae_id", None)
            e_type = type(e).__name__
            e_site = getattr(e, "site_name", None)
            warnings.warn(
                f"Could not detect triangle vertices for EPDF: "
                f"sae_id={e_id}, type={e_type}, site_name={e_site}, latent_idx={e.latent_idx}"
            )
            continue
        else:
            all_candidates.append(np.asarray(tv))
    if not all_candidates:
        raise ValueError("Could not detect triangle vertices for plotting.")
    pts = np.vstack(all_candidates)
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except Exception:
        hull_pts = pts
    if hull_pts.shape[0] < 3:
        raise ValueError("Insufficient hull vertices to define a triangle.")
    if hull_pts.shape[0] == 3:
        triangle_vertices = hull_pts
    else:
        sae_type = "l1" if "lambda" in epdfs[0].sae_id[1] else "top_k"
        if sae_type == "l1":
            sae_param = epdfs[0].sae_id[1][7:]
        else:
            sae_param = epdfs[0].sae_id[1][1:]
        print(f"Got more than 3 hull vertices for layer {epdfs[0].site_name[6:]}, {sae_type} SAE with parameter {sae_param}")
        if False:
            _plot_hull_and_triangles(hull_pts, all_candidates, epdfs, sae_param, sae_type)
        
        def tri_area(a, b, c):
            return abs(0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])))
        best = None
        best_area = -1.0
        for i, j, k in combinations(range(hull_pts.shape[0]), 3):
            A = tri_area(hull_pts[i], hull_pts[j], hull_pts[k])
            if A > best_area:
                best_area = A
                best = (hull_pts[i], hull_pts[j], hull_pts[k])
        triangle_vertices = np.array(best)
    return triangle_vertices



def plot_epdfs(epdfs, mode="3d", grid_size=150, active_latents=None, triangle_vertices=None, color_start_ind=None):
    """
    epdfs: list[LatentEPDF] (must be from the same SAE if multiple)
    mode: "3d", "transparency", or "contours"
    """
    import matplotlib.tri as tri

    if not isinstance(epdfs, (list, tuple)):
        epdfs = [epdfs]

    # Consistency check
    sae_ids = {e.sae_id for e in epdfs}
    if len(sae_ids) > 1 and mode == "3d":
        raise ValueError("3D plotting requires all ePDFs from the same SAE.")


    # Use the new function in plot_epdfs
    if triangle_vertices is None:
        triangle_vertices = get_global_triangle_vertices(epdfs)

    # Generate a triangular grid inside detected simplex
    pts_x, pts_y = [], []
    for i in range(grid_size + 1):
        for j in range(grid_size + 1 - i):
            a = i / grid_size
            b = j / grid_size
            c = 1 - a - b
            x = a * triangle_vertices[0, 0] + b * triangle_vertices[1, 0] + c * triangle_vertices[2, 0]
            y = a * triangle_vertices[0, 1] + b * triangle_vertices[1, 1] + c * triangle_vertices[2, 1]
            pts_x.append(x)
            pts_y.append(y)
    pts_x, pts_y = np.array(pts_x), np.array(pts_y)

    fig = go.Figure()
    # 20 maximally distinctive colors (colorblind-friendly palette)
    colors = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#FFFF33",  # yellow
        "#A65628",  # brown
        "#F781BF",  # pink
        "#999999",  # gray
        "#66C2A5",  # teal
        "#E6AB02",  # mustard
        "#A6CEE3",  # light blue
        "#B2DF8A",  # light green
        "#FB9A99",  # salmon
        "#CAB2D6",  # lavender
        "#FFD92F",  # gold
        "#1B9E77",  # dark green
        "#D95F02",  # dark orange
        "#7570B3",  # indigo
        "#E7298A",  # magenta
    ]

    for idx, epdf in enumerate(epdfs):
        if epdf.kde is None:
            epdf.fit_kde()
        xy = np.vstack([pts_x, pts_y])
        z = epdf.kde(xy)

        if z is None or z.size == 0:
            continue
        color_ind = (idx + (color_start_ind if color_start_ind is not None else 0)) % len(colors)
        color = colors[color_ind]

        # Normalize inside triangle
        z_min, z_max = np.min(z), np.max(z)
        z_norm = (z - z_min) / (z_max - z_min + 1e-12)

        legend_name = f"Latent {epdf.latent_idx}"
        if active_latents is not None and str(epdf.latent_idx) in active_latents:
            legend_name += f", active rate: {active_latents[str(epdf.latent_idx)][0]:.1%}"
        if mode == "3d":
            fig.add_trace(go.Mesh3d(
                x=pts_x, y=pts_y, z=z_norm,
                color=color,
                opacity=0.6,
                name=legend_name,
                showlegend=False,
                alphahull=0
            ))
            # Opaque legend marker
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=6, color=color, opacity=1.0),
                name=legend_name,
                showlegend=True
            ))
        elif mode == "transparency":
            fig.add_trace(go.Scatter(
                x=pts_x, y=pts_y,
                mode="markers",
                marker=dict(size=5, color=color, opacity=z_norm),
                name=legend_name,
                showlegend=False
            ))
            # Opaque legend marker
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=8, color=color, opacity=1.0),
                name=legend_name,
                showlegend=True
            ))
        elif mode == "contours":
            triang = tri.Triangulation(pts_x, pts_y)
            fig.add_trace(go.Contour(
                x=pts_x,
                y=pts_y,
                z=z_norm,
                colorscale=[[0, color], [1, color]],
                contours_coloring="lines",
                line_width=2,
                name=legend_name,
                connectgaps=False
            ))

    # Add simplex boundary
    tri_x = np.append(triangle_vertices[:, 0], triangle_vertices[0, 0])
    tri_y = np.append(triangle_vertices[:, 1], triangle_vertices[0, 1])
    fig.add_trace(go.Scatter(
        x=tri_x, y=tri_y, mode="lines",
        line=dict(color="black"), showlegend=False
    ))

    fig.update_layout(
        title=f"Empirical PDFs for {epdfs[0].site_name} / SAE {epdfs[0].sae_id}",
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=700, width=800,
        legend=dict(font=dict(color="black"))
    )
    return fig



def build_epdfs_for_sae(
    model,
    sae,
    mess3,
    site_name: str,
    sae_type: str,
    param_value: str,
    active_latent_indices,
    pcas: dict,
    seq_len: int = 10,
    n_batches: int = 100,
    device: str = "cpu",
    hook_name: str | None = None,
) -> dict:
    """
    Build LatentEPDFs for all non-dead latents in one SAE.

    Returns:
      {site_name: {sae_type: {param_value: {latent_idx: LatentEPDF}}}}
    """

    # Infer hook name if not provided
    if hook_name is None:
        if site_name == "embeddings":
            hook_name = "hook_embed"
        else:
            layer_idx = int(site_name.split("_")[1])
            hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Generate activation samples
    residuals, tokens = sample_residual_stream_activations(model, mess3, seq_len=seq_len, n_samples=n_batches, device=device)
    acts = residuals[site_name].reshape(-1, residuals[site_name].shape[-1]).to(device)

    # Run through SAE
    with torch.no_grad():
        sae_out = sae(acts)
        latents = sae_out["feature_acts"].cpu().numpy()

    # 🔹 CHANGE: use layer-specific PCA
    if site_name not in pcas:
        raise KeyError(f"No PCA provided for site {site_name}")
    pca_proj = pcas[site_name]

    coords = pca_proj.transform(acts.cpu().numpy())

    # Normalize active latent indices: can be list/array or dict {idx: [count, sum]}
    if isinstance(active_latent_indices, dict):
        try:
            indices = [int(k) for k in active_latent_indices.keys()]
        except Exception:
            indices = list(active_latent_indices.keys())
    else:
        indices = [int(i) for i in active_latent_indices]

    # Build EPDFs
    latent_map = {}
    for li in indices:
        li_int = int(li)
        if li_int < 0 or li_int >= latents.shape[1]:
            continue
        mask = latents[:, li_int] > 1e-6
        if not np.any(mask):
            continue
        latent_map[li_int] = LatentEPDF(
            site_name=site_name,
            sae_id=(sae_type, param_value),
            latent_idx=li_int,
            coords=coords[mask],
            weights=latents[mask, li_int],
        )

    return {site_name: {sae_type: {param_value: latent_map}}}


def build_epdfs_for_all_saes(
    model,
    mess3,
    loaded_saes: dict,
    active_latents: dict,
    pcas: dict,
    seq_len: int = 10,
    n_batches: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Loop over all loaded_saes and build EPDFs.

    Returns:
      sequence_epdfs[site_name][sae_type][param_value][latent_idx] = LatentEPDF
    """

    sequence_epdfs = {}

    for site_name, sae_dict in loaded_saes.items():
        sequence_epdfs[site_name] = {}
        for (k, lmbda), sae_ckpt in sae_dict.items():
            if k is not None:
                sae_class = TopKSAE
                sae_type = "top_k"
                param_value = f"k{k}"
                active_key = (site_name, "sequence", "top_k", f"k{k}")
            else:
                sae_class = VanillaSAE
                sae_type = "l1"
                param_value = f"lambda_{lmbda}"
                active_key = (site_name, "sequence", "vanilla", f"lambda_{lmbda}")

            sae = sae_class(sae_ckpt["cfg"])
            sae.load_state_dict(sae_ckpt["state_dict"])
            sae.to(device).eval()

            active_inds = active_latents.get(active_key, [])
            if not active_inds:
                continue

            # 🔹 CHANGE: pass pcas dict instead of single PCA
            epdfs_dict = build_epdfs_for_sae(
                model,
                sae,
                mess3,
                site_name,
                sae_type,
                param_value,
                active_inds,
                pcas,
                seq_len=seq_len,
                n_batches=n_batches,
                device=device,
            )

            # Merge into top-level dict
            for site, type_map in epdfs_dict.items():
                for sae_type, param_map in type_map.items():
                    sequence_epdfs[site].setdefault(sae_type, {})
                    for param_value, latent_map in param_map.items():
                        sequence_epdfs[site][sae_type][param_value] = latent_map

    return sequence_epdfs


# ==== Multipartite EPDF Utilities ===== #
#########################################


def _project_component_beliefs(beliefs: np.ndarray, comp_meta: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Map raw belief vectors to 2D plotting coordinates based on component type."""

    comp_type = str(comp_meta.get("type", "unknown"))
    info: dict[str, Any] = {"name": comp_meta.get("name"), "type": comp_type}

    if comp_type == "mess3":
        x, y = project_simplex3_to_2d(beliefs)
        coords = np.stack([x, y], axis=-1)
        triangle = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, float(np.sqrt(3) / 2.0)],
            ]
        )
        info["geometry"] = "triangle"
        info["triangle_vertices"] = triangle.tolist()
        return coords, info

    if comp_type == "tom_quantum":
        bloch = tom_quantum_params_to_bloch(beliefs)
        coords = bloch[..., :2]
        info["geometry"] = "circle"
        info["radius"] = 1.0
        return coords, info

    coords = np.asarray(beliefs, dtype=float)
    if coords.ndim == 2 and coords.shape[1] >= 2:
        coords = coords[:, :2]
    else:
        coords = np.zeros((coords.shape[0], 2), dtype=float)
    info["geometry"] = "scatter"
    return coords, info


def build_mp_latent_epdfs(
    *,
    site_name: str,
    sae_id: tuple[str, str],
    latent_activations: np.ndarray,
    component_beliefs: Mapping[str, np.ndarray],
    component_metadata: Mapping[str, Mapping[str, Any]],
    active_latent_indices: Mapping[str, Any] | Sequence[int] | None,
    activation_threshold: float = 1e-6,
    min_active_samples: int = 20,
    bw_method: str | float = "scott",
    progress: bool = False,
    progress_desc: str | None = None,
    progress_disable: bool = False,
) -> dict[int, LatentEPDF]:
    """Create ``LatentEPDF`` instances for multipartite beliefs."""

    if latent_activations.ndim != 2:
        raise ValueError("latent_activations must have shape (samples, n_latents)")

    num_samples, num_latents = latent_activations.shape
    if not component_beliefs:
        raise ValueError("component_beliefs is empty")

    for name, arr in component_beliefs.items():
        if arr.shape[0] != num_samples:
            raise ValueError(
                f"Component '{name}' belief array has {arr.shape[0]} samples; expected {num_samples}."
            )

    if active_latent_indices is None:
        latent_indices = list(range(num_latents))
    elif isinstance(active_latent_indices, Mapping):
        try:
            latent_indices = [int(k) for k in active_latent_indices.keys()]
        except Exception:
            latent_indices = list(active_latent_indices.keys())
    else:
        latent_indices = [int(i) for i in active_latent_indices]

    component_coords_cache: dict[str, np.ndarray] = {}
    component_info_cache: dict[str, dict[str, Any]] = {}
    for comp_name, beliefs_arr in component_beliefs.items():
        meta = component_metadata.get(comp_name, {})
        coords, info = _project_component_beliefs(beliefs_arr, meta)
        component_coords_cache[comp_name] = coords
        component_info_cache[comp_name] = info

    epdfs: dict[int, LatentEPDF] = {}
    iterator = latent_indices
    if progress:
        iterator = tqdm(
            latent_indices,
            desc=progress_desc,
            leave=True,
            disable=False, #progress_disable,
        )

    for latent_idx in iterator:
        if latent_idx < 0 or latent_idx >= num_latents:
            continue
        #print(f"Building EPDF for latent {latent_idx}")
        acts = latent_activations[:, latent_idx]
        mask = np.asarray(acts) > activation_threshold
        active_count = int(mask.sum())
        if active_count < max(1, min_active_samples):
            # Too few active samples to fit a stable KDE; skip this latent.
            continue

        weights = np.asarray(acts[mask])
        component_coords = {
            name: coords[mask]
            for name, coords in component_coords_cache.items()
        }
        component_weights = {
            name: weights.copy()
            for name in component_coords_cache.keys()
        }

        activation_fraction = float(mask.mean())
        epdf = LatentEPDF(
            site_name=site_name,
            sae_id=sae_id,
            latent_idx=int(latent_idx),
            coords=None,
            weights=None,
            component_coords=component_coords,
            component_weights=component_weights,
            component_info=component_info_cache,
            activation_fraction=activation_fraction,
        )
        # Fit KDEs for each component's belief geometry
        epdf.fit_component_kdes(bw_method=bw_method)
        epdfs[int(latent_idx)] = epdf

        sys.stderr.flush()
        sys.stdout.flush()

    return epdfs


def _generate_triangle_grid(vertices: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate barycentric grid inside triangle for mess3 simplex.

    Args:
        vertices: (3, 2) array of triangle vertices
        grid_size: Number of grid divisions per side

    Returns:
        pts_x, pts_y: 1D arrays of grid point coordinates
    """
    pts_x, pts_y = [], []
    for i in range(grid_size + 1):
        for j in range(grid_size + 1 - i):
            a = i / grid_size
            b = j / grid_size
            c = 1 - a - b
            x = a * vertices[0, 0] + b * vertices[1, 0] + c * vertices[2, 0]
            y = a * vertices[0, 1] + b * vertices[1, 1] + c * vertices[2, 1]
            pts_x.append(x)
            pts_y.append(y)
    return np.array(pts_x), np.array(pts_y)


def _generate_circle_grid(radius: float, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate polar grid inside circle for tom_quantum Bloch sphere slice.

    Args:
        radius: Circle radius
        grid_size: Number of grid divisions

    Returns:
        pts_x, pts_y: 1D arrays of grid point coordinates (flattened)
    """
    n_theta = max(grid_size, 50)
    n_r = max(grid_size // 2, 25)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    r = np.linspace(0, radius, n_r)
    theta_grid, r_grid = np.meshgrid(theta, r)
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    return x.ravel(), y.ravel()


def plot_mp_epdfs(
    epdfs: Sequence[LatentEPDF],
    *,
    component_order: Sequence[str] | None = None,
    max_points: int = 4000,
    marker_size: int = 5,
    base_opacity: float = 0.6,
    title: str | None = None,
    mode: str = "transparency",
    grid_size: int = 100,
    legend_outside: bool = False,
):
    """Plot multipartite EPDFs as per-component KDE visualizations using matplotlib.

    Args:
        epdfs: Sequence of LatentEPDF objects with fitted component KDEs
        component_order: Order of components for subplots
        max_points: Maximum points for scatter fallback (unused in KDE modes)
        marker_size: Marker size for scatter/transparency modes
        base_opacity: Base opacity for transparency mode
        title: Figure title
        mode: Visualization mode - "transparency" (scatter with KDE opacity),
              "contours" (contour lines), or "scatter" (fallback, no KDE)
        grid_size: Grid density for KDE evaluation

    Returns:
        Matplotlib figure with KDE-based visualizations
    """

    epdfs = list(epdfs)
    if not epdfs:
        raise ValueError("No EPDFs provided for plotting")

    first = epdfs[0]
    if not first.component_coords:
        raise ValueError(
            "Provided EPDFs lack component_coords; use plot_epdfs for single-component cases"
        )

    if component_order is None:
        component_order = sorted(first.component_coords.keys())
    else:
        component_order = [str(name) for name in component_order if name in first.component_coords]

    if not component_order:
        raise ValueError("component_order could not be determined from EPDFs")

    n_components = len(component_order)
    cols = min(3, n_components)
    rows = int(np.ceil(n_components / cols))

    # Create matplotlib figure with subplots
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.5 * cols, 3.5 * rows),
        squeeze=False,
    )

    # Color palette
    colors = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#FFFF33", "#A65628", "#F781BF", "#999999", "#66C2A5",
        "#E6AB02", "#A6CEE3", "#B2DF8A", "#FB9A99", "#CAB2D6",
        "#FFD92F", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    ]

    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple (0-1 range)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    rng = np.random.default_rng(0)

    # Plot each latent's data
    for latent_offset, epdf in enumerate(epdfs):
        color_hex = colors[latent_offset % len(colors)]
        color_rgb = hex_to_rgb(color_hex)
        legend_name = f"Latent {epdf.latent_idx}"
        if epdf.activation_fraction is not None:
            legend_name += f" (active {epdf.activation_fraction:.1%})"

        for comp_idx, comp_name in enumerate(component_order):
            row_idx = comp_idx // cols
            col_idx = comp_idx % cols
            ax = axes[row_idx, col_idx]

            # Get fitted KDE for this component
            kde = getattr(epdf, 'component_kdes', {}).get(comp_name)
            info = epdf.component_info.get(comp_name, {})
            geometry = info.get("geometry")

            showlegend = comp_idx == 0

            # If KDE is available and mode is not scatter, use KDE visualization
            if kde is not None and mode != "scatter":
                # Generate grid based on geometry
                if geometry == "triangle":
                    verts = np.array(info.get("triangle_vertices"))
                    if verts.size == 0:
                        continue
                    pts_x, pts_y = _generate_triangle_grid(verts, grid_size)
                elif geometry == "circle":
                    radius = float(info.get("radius", 1.0))
                    pts_x, pts_y = _generate_circle_grid(radius, grid_size)
                else:
                    continue  # Unknown geometry, skip

                # Evaluate KDE on grid
                xy = np.vstack([pts_x, pts_y])
                z = kde(xy)
                z_min, z_max = np.min(z), np.max(z)
                z_norm = (z - z_min) / (z_max - z_min + 1e-12)

                # Plot based on mode
                if mode == "transparency":
                    opacity_values = z_norm * base_opacity
                    opacity_values = np.clip(opacity_values, 0.05, 1.0)
                    # Create RGBA colors for each point
                    colors_rgba = np.zeros((len(pts_x), 4))
                    colors_rgba[:, :3] = color_rgb
                    colors_rgba[:, 3] = opacity_values
                    ax.scatter(
                        pts_x, pts_y,
                        s=marker_size,
                        c=colors_rgba,
                        label=legend_name if showlegend else None,
                        edgecolors='none',
                    )
                elif mode == "contours":
                    # Use tricontour for scattered data
                    levels = np.linspace(0.1, 1.0, 8)
                    ax.tricontour(
                        pts_x, pts_y, z_norm,
                        levels=levels,
                        colors=[color_hex],
                        linewidths=2,
                    )
                    if showlegend:
                        # Add a dummy line for legend
                        ax.plot([], [], color=color_hex, linewidth=2, label=legend_name)
            else:
                # Fallback to scatter plot (old behavior)
                coords = epdf.component_coords.get(comp_name)
                if coords is None or coords.size == 0:
                    continue

                weights = epdf.component_weights.get(comp_name)
                total_points = coords.shape[0]
                if total_points == 0:
                    continue

                if total_points > max_points:
                    sel = rng.choice(total_points, size=max_points, replace=False)
                    coords_plot = coords[sel]
                    weights_plot = None if weights is None else weights[sel]
                else:
                    coords_plot = coords
                    weights_plot = weights

                opacity = base_opacity
                if weights_plot is not None and weights_plot.size > 0:
                    norm = weights_plot / (np.max(np.abs(weights_plot)) + 1e-12)
                    opacity = float(np.clip(norm.mean() ** 0.5, 0.15, 1.0))
                    opacity = base_opacity * opacity + 0.1

                ax.scatter(
                    coords_plot[:, 0], coords_plot[:, 1],
                    s=marker_size,
                    c=[color_hex],
                    alpha=opacity,
                    label=legend_name if showlegend else None,
                    edgecolors='none',
                )

    # Add geometry borders and configure axes
    for comp_idx, comp_name in enumerate(component_order):
        row_idx = comp_idx // cols
        col_idx = comp_idx % cols
        ax = axes[row_idx, col_idx]

        info = first.component_info.get(comp_name, {})
        geometry = info.get("geometry")

        if geometry == "triangle":
            verts = np.array(info.get("triangle_vertices"))
            if verts.size:
                line_x = np.append(verts[:, 0], verts[0, 0])
                line_y = np.append(verts[:, 1], verts[0, 1])
                ax.plot(line_x, line_y, 'k-', linewidth=1, zorder=1000)
        elif geometry == "circle":
            radius = float(info.get("radius", 1.0))
            theta = np.linspace(0.0, 2 * np.pi, 200)
            circle_x = radius * np.cos(theta)
            circle_y = radius * np.sin(theta)
            ax.plot(circle_x, circle_y, 'k-', linewidth=1, zorder=1000)

        # Configure axes
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(comp_name, fontsize=10)

        # Add legend only to first subplot (skipped when legend_outside=True)
        if not legend_outside and comp_idx == 0 and len(epdfs) > 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.7)

    # Hide unused subplots
    for idx in range(n_components, rows * cols):
        row_idx = idx // cols
        col_idx = idx % cols
        axes[row_idx, col_idx].axis('off')

    # Place legend outside axes when requested (suited for single-panel with many latents)
    if legend_outside and len(epdfs) > 0:
        all_handles, all_labels = [], []
        seen_labels: set[str] = set()
        for ax_row in axes:
            for ax_ in ax_row:
                for h, l in zip(*ax_.get_legend_handles_labels()):
                    if l not in seen_labels:
                        all_handles.append(h)
                        all_labels.append(l)
                        seen_labels.add(l)
        if all_handles:
            ncol = 2 if len(all_handles) > 12 else 1
            fig.legend(
                all_handles, all_labels,
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                ncol=ncol,
                fontsize=8,
                framealpha=0.7,
                borderaxespad=0,
            )

    # Set overall title
    fig.suptitle(title or "Multipartite Latent EPDFs", fontsize=12)
    fig.tight_layout()

    return fig
