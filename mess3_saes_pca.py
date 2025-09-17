import os
from typing import Dict, List, Tuple
import itertools
from itertools import combinations
import warnings

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from simplexity.generative_processes.torch_generator import generate_data_batch
import torch
import jax
import jax.numpy as jnp
import scipy.stats as st
import plotly.graph_objects as go
from BatchTopK.sae import TopKSAE, VanillaSAE



def project_decoder_directions(
    sae_entries: list,
    pcas: dict,
    latent_indices: dict,
    dims: int = 2,
    normalize: bool = True,
) -> dict:
    """
    Project SAE decoder directions for specified latents into a PCA subspace per site.

    Args:
      sae_entries: list of SAE descriptors; each item is a dict with keys:
        - 'site': site_name, e.g. 'layer_0', 'embeddings'
        - 'sae': instantiated SAE module with attribute W_dec (torch.nn.Parameter)
        - 'sae_type': e.g. 'top_k' or 'vanilla'
        - 'param': string identifier, e.g. 'k2' or 'lambda_0.1'
      pcas: dict mapping site_name -> fitted PCA object (sklearn.decomposition.PCA)
      latent_indices: dict mapping (site, sae_type, param) -> iterable of latent indices to project
      dims: 2 or 3, number of principal components to project onto
      normalize: if True, L2-normalize decoder rows before projection (direction only)

    Returns:
      projections: dict[site][sae_type][param][latent_idx] = np.ndarray of shape (dims,)
    """
    assert dims in (2, 3), "dims must be 2 or 3"

    projections: dict = {}

    for entry in sae_entries:
        site = entry.get('site')
        sae = entry.get('sae')
        sae_type = entry.get('sae_type')
        param = entry.get('param')

        if site not in pcas:
            raise KeyError(f"No PCA provided for site '{site}'")
        pca = pcas[site]
        if getattr(pca, 'components_', None) is None:
            raise ValueError(f"PCA for site '{site}' is not fitted (missing components_)\n")
        if pca.components_.shape[0] < dims:
            raise ValueError(f"PCA for site '{site}' has only {pca.components_.shape[0]} components, but dims={dims}")

        # Get latent indices to project
        idx_key = (site, sae_type, param)
        idxs = latent_indices.get(idx_key, None)
        if idxs is None:
            # If not provided, skip this SAE
            continue

        # Access decoder weights
        if not hasattr(sae, 'W_dec'):
            raise AttributeError(f"SAE for {idx_key} has no attribute 'W_dec'")
        W_dec = sae.W_dec.detach().cpu().numpy()  # shape: [dict_size, d_model]

        # Prepare PCA components (dims x d_model)
        comps = pca.components_[:dims, :]  # (dims, d_model)

        # Initialize nested dicts
        projections.setdefault(site, {}).setdefault(sae_type, {}).setdefault(param, {})

        for li in idxs:
            li_int = int(li)
            if li_int < 0 or li_int >= W_dec.shape[0]:
                continue
            v = W_dec[li_int]  # (d_model,)
            if normalize:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            # Project: coords = components * v
            coords = comps @ v  # (dims,)
            projections[site][sae_type][param][li_int] = coords

    return projections


def plot_decoder_projections(
    projections: dict,
    *,
    dims: int = 2,
    selection: list | None = None,
    title: str | None = None,
    use_cones_3d: bool = True,
    scale: float = 1.0,
    activity_rates: dict | None = None,
) -> go.Figure:
    """
    Plot decoder direction projections as arrows from the origin.

    projections: dict returned by project_decoder_directions:
      projections[site][sae_type][param][latent_idx] = np.ndarray (dims,)

    dims: 2 or 3 for 2D/3D plot
    selection: optional list of (site, sae_type, param, latent_idx) to plot;
               if None, plot all in projections
    title: optional plot title
    use_cones_3d: when dims==3, draw Cone arrows (anchored at tail) if True, otherwise line segments
    scale: multiply each vector by this factor for visualization scaling
    activity_rates: optional dict mapping (site, sae_type, param, latent_idx) -> activity fraction in [0,1]

    Returns: plotly.graph_objects.Figure
    """
    assert dims in (2, 3), "dims must be 2 or 3"

    # Collect vectors
    items = []  # list of (site, sae_type, param, latent_idx, vec)
    if selection is not None:
        for site, sae_type, param, li in selection:
            try:
                v = projections[site][sae_type][param][int(li)]
                if v.shape[0] >= dims:
                    items.append((site, sae_type, param, int(li), v[:dims] * scale))
            except Exception:
                continue
    else:
        for site, d1 in projections.items():
            for sae_type, d2 in d1.items():
                for param, d3 in d2.items():
                    for li, v in d3.items():
                        if v.shape[0] >= dims:
                            items.append((site, sae_type, param, int(li), v[:dims] * scale))

    fig = go.Figure()

    # color palette
    colors = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#FFFF33", "#A65628", "#F781BF", "#999999", "#66C2A5",
        "#E6AB02", "#A6CEE3", "#B2DF8A", "#FB9A99", "#CAB2D6",
        "#FFD92F", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    ]

    # Group by (site, sae_type, param) for legend clarity
    for idx, (site, sae_type, param, li, vec) in enumerate(items):
        color = colors[idx % len(colors)]
        # Build compact legend name: just latent number, optionally with activity rate
        rate_suffix = ""
        if activity_rates is not None:
            key = (site, sae_type, param, li)
            rate = activity_rates.get(key)
            if rate is None:
                # try string latent key variant if upstream stored strings
                rate = activity_rates.get((site, sae_type, param, int(li)))
            if rate is not None:
                try:
                    rate_suffix = f" ({float(rate):.1%})"
                except Exception:
                    pass
        name = f"latent {li}{rate_suffix}"

        if dims == 2:
            x0, y0 = 0.0, 0.0
            x1, y1 = float(vec[0]), float(vec[1])
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color, opacity=1.0),
                name=name,
            ))
        else:
            x0, y0, z0 = 0.0, 0.0, 0.0
            x1, y1, z1 = float(vec[0]), float(vec[1]), float(vec[2] if vec.shape[0] >= 3 else 0.0)
            if use_cones_3d:
                fig.add_trace(go.Cone(
                    x=[x0], y=[y0], z=[z0],
                    u=[x1], v=[y1], w=[z1],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.12,
                    anchor="tail",
                    name=name,
                ))
                # opaque legend marker
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers",
                    marker=dict(size=6, color=color, opacity=1.0),
                    name=name,
                    showlegend=True,
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode="lines+markers",
                    line=dict(color=color, width=4),
                    marker=dict(size=4, color=color, opacity=1.0),
                    name=name,
                ))

    if dims == 2:
        fig.update_layout(
            title=title or "Decoder directions (PCA space, 2D)",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(),
            height=700, width=800,
            legend=dict(font=dict(color="black")),
        )
    else:
        fig.update_layout(
            title=title or "Decoder directions (PCA space, 3D)",
            scene=dict(
                xaxis=dict(title="PC1"),
                yaxis=dict(title="PC2"),
                zaxis=dict(title="PC3"),
                aspectmode="cube",
            ),
            height=700, width=900,
            legend=dict(font=dict(color="black")),
        )

    return fig
