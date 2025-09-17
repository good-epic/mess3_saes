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



from scipy.spatial import ConvexHull

class LatentEPDF:
    def __init__(self, site_name, sae_id, latent_idx, coords, weights):
        self.site_name = site_name
        self.sae_id = sae_id
        self.latent_idx = latent_idx
        self.coords = np.asarray(coords)
        self.weights = np.asarray(weights)
        self.kde = None

        # Detect triangle on init
        self.triangle_vertices = self._detect_triangle_vertices()

    def __str__(self):
        return f"({self.site_name}, {self.sae_id[0]}, {self.sae_id[1]}, {self.latent_idx})"

    def __repr__(self):
        return self.__str__()

    def fit_kde(self, bw_method="scott"):
        if self.coords.shape[0] < 5:
            return None
        self.kde = st.gaussian_kde(self.coords.T, weights=self.weights, bw_method=bw_method)
        return self.kde

    def evaluate_on_grid(self, grid_x, grid_y):
        if self.kde is None:
            self.fit_kde()
        xy = np.vstack([grid_x.ravel(), grid_y.ravel()])
        z = self.kde(xy).reshape(grid_x.shape)
        return z

    def _detect_triangle_vertices(self, tol=0.02, verbose=False):
        """Infer simplex triangle vertices from this latent's coords (robust to PCA noise)."""
        if self.coords.shape[0] < 3:
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
