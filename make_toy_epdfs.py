#!/usr/bin/env python3
"""
Generate per-cluster EPDF figures for the multipartite toy model.

Reads cluster assignments from top_r2_run_layer_1_cluster_summary.json,
samples sequences from the 3xMess3+2xTomQuantum process, runs the model+SAE
to get latent activations and component beliefs, then renders EPDFs using
improved imshow+pixel-grid KDE (matching the style of remake_latent_spatial.py).

Outputs per cluster:
  cluster_{id}_all.png          — all latents overlaid, 5 component subplots
  cluster_{id}/latent_{idx}.png — per-latent figure, same 5 component subplots

Colors are assigned once per cluster (sorted latent index → stable color) and
are identical in the all-latents and per-latent figures.

Usage
-----
    python make_toy_epdfs.py \\
        --cluster_summary outputs/reports/multipartite_003e/top_r2_run_layer_1_cluster_summary.json \\
        --model_ckpt outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt \\
        --sae_path  outputs/saes/multipartite_003e/layer_1_top_k_k12.pt \\
        --output_dir outputs/reports/multipartite_003e/cluster_epdfs \\
        [--n_sequences 10000] [--batch_size 512] [--grid_size 150] [--dpi 150]
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import colorsys
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

import jax
from transformer_lens import HookedTransformer, HookedTransformerConfig

from BatchTopK.sae import TopKSAE
from multipartite_utils import MultipartiteSampler, build_components_from_config
from epdf_utils import build_mp_latent_epdfs


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_GOLDEN = 0.6180339887498949   # 1/φ — golden angle step in hue space


def build_palette(n: int) -> list:
    """
    Return n maximally-differentiated RGB triples.

    Uses the golden-angle hue step (≈137.5°) so that any two adjacent
    indices in the sequence are as far apart in hue as possible, regardless
    of n.  Alternating saturation/value provides extra separation.
    """
    if n == 0:
        return []
    colours = []
    for i in range(n):
        hue = (i * _GOLDEN) % 1.0          # maximally separated hues
        if i % 3 == 0:
            sat, val = 0.90, 0.85
        elif i % 3 == 1:
            sat, val = 0.65, 0.92
        else:
            sat, val = 0.85, 0.60
        colours.append(colorsys.hsv_to_rgb(hue, sat, val))
    return colours


# ---------------------------------------------------------------------------
# KDE helpers — triangle geometry (mess3)
# ---------------------------------------------------------------------------

_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])


def triangle_mask(img_size: int):
    sqrt3_2 = math.sqrt(3) / 2
    xs = np.linspace(0.0, 1.0, img_size)
    ys = np.linspace(0.0, sqrt3_2, img_size)
    xx, yy = np.meshgrid(xs, ys)
    s3 = math.sqrt(3)
    inside = (yy >= 0) & (yy <= xx * s3) & (yy <= s3 * (1.0 - xx))
    return xx, yy, inside


def kde_to_rgba_triangle(kde, color_rgb, img_size: int, base_opacity: float = 0.6):
    """Render KDE as RGBA array over the canonical equilateral triangle."""
    xx, yy, inside = triangle_mask(img_size)
    z = np.zeros(xx.shape)
    pts = np.vstack([xx[inside], yy[inside]])
    if pts.shape[1] > 0:
        z[inside] = kde(pts)
    z_in = z[inside]
    z_norm = np.zeros_like(z)
    if z_in.max() > z_in.min() + 1e-12:
        z_norm[inside] = (z_in - z_in.min()) / (z_in.max() - z_in.min())
    rgba = np.zeros((*xx.shape, 4), dtype=float)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = np.clip(z_norm * base_opacity, 0.0, 1.0)
    rgba[~inside] = 0.0
    return rgba


def imshow_kde_triangle(ax, kde, color_rgb, img_size: int, base_opacity: float = 0.6):
    sqrt3_2 = math.sqrt(3) / 2
    rgba = kde_to_rgba_triangle(kde, color_rgb, img_size, base_opacity)
    ax.imshow(rgba, extent=[0.0, 1.0, 0.0, sqrt3_2],
              origin="lower", aspect="equal", interpolation="bilinear", zorder=5)


def draw_triangle(ax, pad=0.06):
    loop = np.vstack([_V, _V[:1]])
    ax.plot(loop[:, 0], loop[:, 1], "k-", lw=1.5, zorder=20)
    offsets = [(-pad, -pad * 0.8), (1 + pad * 0.5, -pad * 0.8),
               (0.5, _V[2, 1] + pad * 0.6)]
    for label, (ox, oy) in zip(("V0", "V1", "V2"), offsets):
        ax.text(ox, oy, label, fontsize=8, ha="center", va="center",
                color="#333333", zorder=21)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.14, _V[2, 1] + 0.14)


# ---------------------------------------------------------------------------
# KDE helpers — circle geometry (tom_quantum)
# ---------------------------------------------------------------------------

def circle_mask(img_size: int, radius: float = 1.0):
    xs = np.linspace(-radius, radius, img_size)
    ys = np.linspace(-radius, radius, img_size)
    xx, yy = np.meshgrid(xs, ys)
    inside = (xx ** 2 + yy ** 2) <= radius ** 2
    return xx, yy, inside


def kde_to_rgba_circle(kde, color_rgb, img_size: int, radius: float = 1.0,
                       base_opacity: float = 0.6):
    """Render KDE as RGBA array over a disk of given radius."""
    xx, yy, inside = circle_mask(img_size, radius)
    z = np.zeros(xx.shape)
    pts = np.vstack([xx[inside], yy[inside]])
    if pts.shape[1] > 0:
        z[inside] = kde(pts)
    z_in = z[inside]
    z_norm = np.zeros_like(z)
    if z_in.max() > z_in.min() + 1e-12:
        z_norm[inside] = (z_in - z_in.min()) / (z_in.max() - z_in.min())
    rgba = np.zeros((*xx.shape, 4), dtype=float)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = np.clip(z_norm * base_opacity, 0.0, 1.0)
    rgba[~inside] = 0.0
    return rgba


def imshow_kde_circle(ax, kde, color_rgb, img_size: int, radius: float = 1.0,
                      base_opacity: float = 0.6):
    rgba = kde_to_rgba_circle(kde, color_rgb, img_size, radius, base_opacity)
    ax.imshow(rgba, extent=[-radius, radius, -radius, radius],
              origin="lower", aspect="equal", interpolation="bilinear", zorder=5)


def draw_circle(ax, radius: float = 1.0, pad: float = 0.1):
    theta = np.linspace(0.0, 2 * math.pi, 300)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k-", lw=1.5, zorder=20)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    r = radius + pad
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)


# ---------------------------------------------------------------------------
# Generic dispatchers
# ---------------------------------------------------------------------------

def imshow_kde_geom(ax, kde, color_rgb, geometry: str, img_size: int,
                    radius: float = 1.0, base_opacity: float = 0.6):
    if geometry == "triangle":
        imshow_kde_triangle(ax, kde, color_rgb, img_size, base_opacity)
    elif geometry == "circle":
        imshow_kde_circle(ax, kde, color_rgb, img_size, radius, base_opacity)


def draw_geom(ax, geometry: str, radius: float = 1.0):
    if geometry == "triangle":
        draw_triangle(ax)
    elif geometry == "circle":
        draw_circle(ax, radius)


def configure_ax(ax, comp_name: str, assigned_comp: str | None,
                 component_info: dict, n_epdfs: int):
    """Set background, border, title, ticks, axis limits for one subplot."""
    ax.set_facecolor("white")
    info = component_info.get(comp_name, {})
    geometry = info.get("geometry", "unknown")
    radius = float(info.get("radius", 1.0))
    draw_geom(ax, geometry, radius)
    title = comp_name
    if comp_name == assigned_comp:
        title += "  ★"
    ax.set_title(title, fontsize=10, pad=4)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def sample_beliefs_and_acts(sampler, model, sae, n_sequences: int,
                             seq_len: int, batch_size: int, layer: int,
                             token_inds: list, device: str):
    """Sample sequences, run model+SAE, return (beliefs_flat, latent_acts).

    beliefs_flat : (N, total_belief_dim)   N = n_sequences * len(token_inds)
    latent_acts  : (N, n_latents)
    """
    all_beliefs = []
    all_latent_acts = []

    key = jax.random.PRNGKey(42)
    n_done = 0
    while n_done < n_sequences:
        bs = min(batch_size, n_sequences - n_done)
        key, beliefs_jax, tokens_jax, _ = sampler.sample(key, bs, seq_len)

        # beliefs_jax: (bs, seq_len-1, total_belief_dim)
        # tokens_jax:  (bs, seq_len-1)
        beliefs_np = np.array(beliefs_jax)
        tokens_np = np.array(tokens_jax)

        # Select token positions; both beliefs and tokens have seq_len-1 positions
        beliefs_sel = beliefs_np[:, token_inds, :]           # (bs, n_pos, bd)
        beliefs_flat = beliefs_sel.reshape(-1, beliefs_np.shape[-1])  # (bs*n_pos, bd)

        tokens_t = torch.from_numpy(tokens_np).long().to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_t)
            resid = cache[f"blocks.{layer}.hook_resid_post"]  # (bs, seq, d_model)
            resid_sel = resid[:, token_inds, :]               # (bs, n_pos, d_model)
            resid_flat = resid_sel.reshape(-1, resid.shape[-1])

            sae_out = sae(resid_flat)
            lat_acts = sae_out["feature_acts"].cpu().numpy()  # (bs*n_pos, n_latents)

        all_beliefs.append(beliefs_flat)
        all_latent_acts.append(lat_acts)
        n_done += bs
        print(f"  sampled {n_done}/{n_sequences} sequences", flush=True)

    return np.concatenate(all_beliefs, axis=0), np.concatenate(all_latent_acts, axis=0)


from scipy.stats import gaussian_kde as _gaussian_kde


def _fast_kde(kde, max_samples: int = 2000, seed: int = 0):
    """Return a scipy gaussian_kde re-fitted on at most max_samples points.

    Evaluation cost is O(n_eval × n_train), so capping training data at 2000
    makes even a 45k-point grid evaluation take < 0.1 s.  The bandwidth
    is preserved from the original fit.
    """
    data = kde.dataset   # shape (d, n)
    n = data.shape[1]
    if n <= max_samples:
        return kde
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_samples, replace=False)
    return _gaussian_kde(data[:, idx], bw_method=kde.factor)


def _render_kde_rgba(kde, color_rgb, xx, yy, inside, base_opacity: float):
    """Evaluate a (possibly subsampled) KDE on a pre-computed pixel grid."""
    pts = np.vstack([xx[inside], yy[inside]])
    z = np.zeros(xx.shape)
    if pts.shape[1] > 0:
        z[inside] = kde(pts)
    z_in = z[inside]
    z_norm = np.zeros_like(z)
    if z_in.max() > z_in.min() + 1e-12:
        z_norm[inside] = (z_in - z_in.min()) / (z_in.max() - z_in.min())
    rgba = np.zeros((*xx.shape, 4), dtype=float)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = np.clip(z_norm * base_opacity, 0.0, 1.0)
    rgba[~inside] = 0.0
    return rgba


# ---------------------------------------------------------------------------
# Prerendering — evaluate each KDE on the pixel grid once, reuse everywhere
# ---------------------------------------------------------------------------

def prerender_cluster(epdf_dict: dict, component_order: list,
                      comp_info: dict, colour_map: dict,
                      img_size: int, base_opacity: float,
                      max_kde_samples: int = 2000,
                      geom_grids: dict | None = None) -> tuple[dict, dict]:
    """
    Evaluate every (latent, component) KDE on its pixel grid exactly once.

    Returns
    -------
    rendered : {global_latent_idx: {comp_name: rgba_array (H, W, 4)}}
    extents  : {comp_name: [x0, x1, y0, y1]}
    """
    rendered: dict[int, dict[str, np.ndarray]] = {}
    extents: dict[str, list] = {}

    # Build geom_grids if not supplied
    if geom_grids is None:
        geom_grids = {}
        for comp_name in component_order:
            info = comp_info.get(comp_name, {})
            geometry = info.get("geometry", "unknown")
            radius = float(info.get("radius", 1.0))
            if geometry == "triangle":
                sqrt3_2 = math.sqrt(3) / 2
                xx, yy, inside = triangle_mask(img_size)
                pts = np.vstack([xx[inside], yy[inside]])
                geom_grids[comp_name] = (geometry, xx, yy, inside, pts, radius)
                extents[comp_name] = [0.0, 1.0, 0.0, sqrt3_2]
            elif geometry == "circle":
                xx, yy, inside = circle_mask(img_size, radius)
                pts = np.vstack([xx[inside], yy[inside]])
                geom_grids[comp_name] = (geometry, xx, yy, inside, pts, radius)
                extents[comp_name] = [-radius, radius, -radius, radius]
    else:
        # extents from supplied geom_grids
        for comp_name, tup in geom_grids.items():
            geometry, xx, yy, inside, pts, radius = tup
            if geometry == "triangle":
                extents[comp_name] = [0.0, 1.0, 0.0, math.sqrt(3) / 2]
            elif geometry == "circle":
                extents[comp_name] = [-radius, radius, -radius, radius]

    n_total = len(epdf_dict) * len(component_order)
    done = 0
    for latent_idx, epdf in epdf_dict.items():
        col_rgb = colour_map[latent_idx]
        rendered[latent_idx] = {}
        for comp_name in component_order:
            kde = getattr(epdf, "component_kdes", {}).get(comp_name)
            if kde is None or comp_name not in geom_grids:
                done += 1
                continue
            geometry, xx, yy, inside, pts, radius = geom_grids[comp_name]
            fast = _fast_kde(kde, max_kde_samples)
            rendered[latent_idx][comp_name] = _render_kde_rgba(
                fast, col_rgb, xx, yy, inside, base_opacity
            )
            done += 1
            if done % max(1, n_total // 20) == 0:
                print(f"  rendered {done}/{n_total} KDEs", flush=True)

    return rendered, extents


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _make_axes_grid(n_comp: int) -> tuple[plt.Figure, np.ndarray]:
    cols = min(3, n_comp)
    rows = math.ceil(n_comp / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows),
                             squeeze=False)
    return fig, axes


def _add_legend_to_slot(axes, n_comp: int, cols: int, legend_handles: list):
    """Place legend in the first empty grid slot; fall back to outside-right."""
    n_total = math.ceil(n_comp / cols) * cols
    empty_slots = [(i // cols, i % cols) for i in range(n_comp, n_total)]
    ncol = max(1, math.ceil(len(legend_handles) / 20))
    if empty_slots:
        leg_row, leg_col = empty_slots[0]
        leg_ax = axes[leg_row][leg_col]
        leg_ax.axis("off")
        leg_ax.legend(handles=legend_handles, loc="upper left",
                      bbox_to_anchor=(0.0, 1.0), ncol=ncol,
                      fontsize=6, framealpha=0.8, borderaxespad=0,
                      title="Latent (act%)", title_fontsize=7)
        for r, c in empty_slots[1:]:
            axes[r][c].axis("off")
    else:
        for r, c in empty_slots:
            axes[r][c].axis("off")
        axes[0][0].get_figure().legend(
            handles=legend_handles, loc="upper left",
            bbox_to_anchor=(1.01, 1.0), ncol=ncol,
            fontsize=6, framealpha=0.8, borderaxespad=0,
            title="Latent (act%)", title_fontsize=7,
        )


def plot_all_latents(cluster_id: str, epdf_dict: dict,
                     component_order: list,
                     assigned_comp: str | None,
                     colour_map: dict,
                     rendered: dict,   # prerendered: {latent_idx: {comp_name: rgba}}
                     extents: dict,    # {comp_name: [x0,x1,y0,y1]}
                     title_prefix: str) -> plt.Figure:
    """One figure with n_components subplots; all latents overlaid."""
    n_comp = len(component_order)
    fig, axes = _make_axes_grid(n_comp)
    cols = min(3, n_comp)

    first_epdf = next(iter(epdf_dict.values()))
    comp_info = dict(first_epdf.component_info)

    legend_handles = []
    for latent_idx, epdf in epdf_dict.items():
        col_rgb = colour_map[latent_idx]
        act_frac = epdf.activation_fraction or 0.0
        legend_handles.append(
            mpatches.Patch(color=col_rgb, label=f"Latent {latent_idx} ({act_frac:.0%})")
        )
        for ci, comp_name in enumerate(component_order):
            rgba = rendered.get(latent_idx, {}).get(comp_name)
            if rgba is None:
                continue
            row_i, col_i = divmod(ci, cols)
            ax = axes[row_i][col_i]
            ext = extents.get(comp_name, [0, 1, 0, 1])
            ax.imshow(rgba, extent=ext, origin="lower", aspect="equal",
                      interpolation="bilinear", zorder=5)

    for ci, comp_name in enumerate(component_order):
        row_i, col_i = divmod(ci, cols)
        configure_ax(axes[row_i][col_i], comp_name, assigned_comp, comp_info,
                     len(epdf_dict))

    _add_legend_to_slot(axes, n_comp, cols, legend_handles[:40])

    assigned_str = f" — assigned to {assigned_comp}" if assigned_comp else " — noise"
    fig.suptitle(f"{title_prefix}: Cluster {cluster_id}{assigned_str}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_solo_latent(cluster_id: str, latent_idx: int, epdf,
                     component_order: list,
                     assigned_comp: str | None,
                     colour_map: dict,        # global_latent_idx → RGB
                     geom_grids: dict,        # comp_name → (geometry, xx, yy, inside, pts, radius)
                     extents: dict,           # comp_name → [x0,x1,y0,y1]
                     solo_opacity: float,
                     max_kde_samples: int,
                     title_prefix: str) -> plt.Figure:
    """Five-subplot figure for a single latent.

    KDEs are rendered fresh here (not from the prerender cache) at full
    solo_opacity so each figure has vivid, clearly-coloured blobs.
    Only 5 KDE evaluations → fast.
    """
    n_comp = len(component_order)
    fig, axes = _make_axes_grid(n_comp)
    cols = min(3, n_comp)

    comp_info = dict(epdf.component_info)
    col_rgb = colour_map[latent_idx]          # this latent's unique colour
    act_frac = epdf.activation_fraction or 0.0

    for ci, comp_name in enumerate(component_order):
        row_i, col_i = divmod(ci, cols)
        ax = axes[row_i][col_i]
        kde = getattr(epdf, "component_kdes", {}).get(comp_name)
        if kde is not None and comp_name in geom_grids:
            _, xx, yy, inside, pts, radius = geom_grids[comp_name]
            fast = _fast_kde(kde, max_kde_samples)
            rgba = _render_kde_rgba(fast, col_rgb, xx, yy, inside, solo_opacity)
            ext = extents.get(comp_name, [0, 1, 0, 1])
            ax.imshow(rgba, extent=ext, origin="lower", aspect="equal",
                      interpolation="bilinear", zorder=5)
        configure_ax(ax, comp_name, assigned_comp, comp_info, 1)

    for idx in range(n_comp, math.ceil(n_comp / cols) * cols):
        axes[idx // cols][idx % cols].axis("off")

    assigned_str = f" — assigned to {assigned_comp}" if assigned_comp else " — noise"
    fig.suptitle(
        f"{title_prefix}: Cluster {cluster_id}{assigned_str}\n"
        f"Latent {latent_idx}  (active {act_frac:.1%})",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_summary",
        default="outputs/reports/multipartite_003e/top_r2_run_layer_1_cluster_summary.json")
    ap.add_argument("--model_ckpt",
        default="outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt")
    ap.add_argument("--sae_path",
        default="outputs/saes/multipartite_003e/layer_1_top_k_k12.pt")
    ap.add_argument("--output_dir",
        default="outputs/reports/multipartite_003e/cluster_epdfs")
    ap.add_argument("--n_sequences", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--grid_size", type=int, default=80,
        help="Pixel resolution for imshow KDE grid (img_size = grid_size * 3). "
             "Higher = sharper but slower; 80→240px is good for print.")
    ap.add_argument("--base_opacity", type=float, default=0.55)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--ext", default="png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--token_inds", default="4,9,14")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--process_config_name", default="3xmess3_2xtquant_003")
    ap.add_argument("--no_solo", action="store_true",
        help="Skip per-latent solo figures (faster)")
    ap.add_argument("--max_kde_samples", type=int, default=2000,
        help="Subsample KDE training data to this many points before evaluating "
             "on the pixel grid. Keeps rendering fast without visible quality loss.")
    ap.add_argument("--solo_opacity", type=float, default=0.90,
        help="Opacity for solo latent figures (higher = more vivid than all-latents overlay)")
    # Model arch (must match checkpoint)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_ctx", type=int, default=16)
    ap.add_argument("--d_head", type=int, default=32)
    ap.add_argument("--act_fn", default="relu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token_inds = [int(x) for x in args.token_inds.split(",")]
    device = args.device
    img_size = args.grid_size * 3

    # ------------------------------------------------------------------
    # Load cluster summary
    # ------------------------------------------------------------------
    print("Loading cluster summary …")
    with open(args.cluster_summary) as fh:
        summary = json.load(fh)

    clusters = summary["clusters"]  # {str(id): {"latent_indices": [...], ...}}
    assign_hard = summary.get("component_assignment_hard", {}).get("assignments", {})
    # assign_hard: {comp_name: cluster_id_int}
    cluster_to_comp = {str(v): k for k, v in assign_hard.items()}

    # ------------------------------------------------------------------
    # Build MultipartiteSampler
    # ------------------------------------------------------------------
    print("Building MultipartiteSampler …")
    cfg_path = Path(__file__).parent / "process_configs.json"
    with open(cfg_path) as fh:
        all_cfgs = json.load(fh)
    components = build_components_from_config(all_cfgs[args.process_config_name])
    sampler = MultipartiteSampler(components)
    print(f"  components: {[c.name for c in sampler.components]}")
    print(f"  vocab_size: {sampler.vocab_size}")

    comp_names = [c.name for c in sampler.components]
    # process_type attribute drives geometry selection in _project_component_beliefs
    comp_types = {c.name: getattr(c, "process_type", c.name.split("_")[0])
                  for c in sampler.components}
    component_state_dims = list(sampler.component_state_dims)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading model …")
    cfg = HookedTransformerConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        n_ctx=args.n_ctx, d_vocab=sampler.vocab_size,
        act_fn=args.act_fn, device=device, d_head=args.d_head,
    )
    model = HookedTransformer(cfg).to(device)
    ckpt = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded from {args.model_ckpt}")

    # ------------------------------------------------------------------
    # Load SAE
    # ------------------------------------------------------------------
    print("Loading SAE …")
    sae_ckpt = torch.load(args.sae_path, map_location=device, weights_only=False)
    sae_cfg = dict(sae_ckpt["cfg"])
    sae_cfg["device"] = device  # ensure buffer initialised on target device
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(sae_ckpt["state_dict"])
    sae.to(device).eval()
    # num_batches_not_active is not a registered buffer; move it manually
    if hasattr(sae, "num_batches_not_active"):
        sae.num_batches_not_active = sae.num_batches_not_active.to(device)
    print(f"  SAE dict_size={sae_ckpt['cfg']['dict_size']}")

    # ------------------------------------------------------------------
    # Generate data
    # ------------------------------------------------------------------
    print(f"Generating {args.n_sequences} sequences …")
    # sampler.sample(key, bs, seq_len) returns seq_len-1 tokens and
    # seq_len-1 belief states.  With seq_len=n_ctx=16 → 15 positions (0–14).
    # token_inds default [4,9,14] are all valid.
    seq_len = args.n_ctx
    beliefs_flat, latent_acts = sample_beliefs_and_acts(
        sampler, model, sae,
        n_sequences=args.n_sequences,
        seq_len=seq_len,
        batch_size=args.batch_size,
        layer=args.layer,
        token_inds=token_inds,
        device=device,
    )
    # beliefs_flat: (N, total_belief_dim)  where N = n_sequences * len(token_inds)
    # latent_acts:  (N, n_latents)
    print(f"  beliefs_flat: {beliefs_flat.shape}, latent_acts: {latent_acts.shape}")

    # Split beliefs by component
    cursor = 0
    component_beliefs: dict[str, np.ndarray] = {}
    for comp, dim in zip(sampler.components, component_state_dims):
        component_beliefs[comp.name] = beliefs_flat[:, cursor: cursor + dim]
        cursor += dim

    # Component metadata for build_mp_latent_epdfs
    component_metadata: dict[str, dict] = {
        c.name: {"name": c.name, "type": comp_types[c.name]}
        for c in sampler.components
    }

    component_order = comp_names  # [tom_quantum, tom_quantum_1, mess3, mess3_1, mess3_2]

    # ------------------------------------------------------------------
    # Per-cluster loop
    # ------------------------------------------------------------------
    for cluster_id, cluster_info in clusters.items():
        latent_indices = cluster_info["latent_indices"]
        assigned_comp = cluster_to_comp.get(cluster_id)
        print(f"\nCluster {cluster_id}: {len(latent_indices)} latents "
              f"→ assigned to {assigned_comp or 'noise'}")

        # Subset activations to this cluster's latents (local indices 0…m-1)
        lat_idx_arr = np.array(latent_indices, dtype=int)
        lat_acts_cluster = latent_acts[:, lat_idx_arr]  # (N, m)

        # Build EPDFs — keys are local indices 0…m-1
        active_latent_indices = {str(i): None for i in range(len(latent_indices))}
        epdf_dict_local = build_mp_latent_epdfs(
            site_name=f"cluster_{cluster_id}",
            sae_id=("top_k", "12"),
            latent_activations=lat_acts_cluster,
            component_beliefs=component_beliefs,
            component_metadata=component_metadata,
            active_latent_indices=active_latent_indices,
            activation_threshold=1e-6,
            min_active_samples=20,
            progress=True,
            progress_desc=f"Cluster {cluster_id} EPDFs",
        )
        print(f"  Built {len(epdf_dict_local)} EPDFs")
        if not epdf_dict_local:
            continue

        # Remap local → global SAE latent indices; update epdf.latent_idx
        epdf_dict: dict[int, object] = {}
        for local_idx, epdf in epdf_dict_local.items():
            global_idx = latent_indices[int(local_idx)]
            epdf.latent_idx = global_idx
            epdf_dict[global_idx] = epdf

        # Build colour map: sort global indices → assign palette in that order
        # so colours are identical across all-latents and solo figures
        sorted_global = sorted(epdf_dict.keys())
        palette = build_palette(len(sorted_global))
        colour_map: dict[int, tuple] = {
            gidx: palette[rank] for rank, gidx in enumerate(sorted_global)
        }

        title_prefix = "Layer 1 TopK k=12"

        # Get component_info from first epdf for geometry lookups
        first_epdf = next(iter(epdf_dict.values()))
        comp_info = dict(first_epdf.component_info)

        # ---- build geom_grids once (shared by prerender + solo) ----
        geom_grids: dict = {}
        extents: dict = {}
        for comp_name in component_order:
            info = comp_info.get(comp_name, {})
            geometry = info.get("geometry", "unknown")
            radius = float(info.get("radius", 1.0))
            if geometry == "triangle":
                sqrt3_2 = math.sqrt(3) / 2
                xx, yy, inside = triangle_mask(img_size)
                pts = np.vstack([xx[inside], yy[inside]])
                geom_grids[comp_name] = (geometry, xx, yy, inside, pts, radius)
                extents[comp_name] = [0.0, 1.0, 0.0, sqrt3_2]
            elif geometry == "circle":
                xx, yy, inside = circle_mask(img_size, radius)
                pts = np.vstack([xx[inside], yy[inside]])
                geom_grids[comp_name] = (geometry, xx, yy, inside, pts, radius)
                extents[comp_name] = [-radius, radius, -radius, radius]

        # ---- prerender all KDEs once for all-latents overlay ----
        print(f"  Prerendering {len(epdf_dict)} × {len(component_order)} KDEs "
              f"(subsampled to ≤{args.max_kde_samples} pts each) …")
        rendered, _ = prerender_cluster(
            epdf_dict=epdf_dict,
            component_order=component_order,
            comp_info=comp_info,
            colour_map=colour_map,
            img_size=img_size,
            base_opacity=args.base_opacity,
            max_kde_samples=args.max_kde_samples,
            geom_grids=geom_grids,
        )

        # ---- all-latents figure ----
        fig_all = plot_all_latents(
            cluster_id=cluster_id,
            epdf_dict=epdf_dict,
            component_order=component_order,
            assigned_comp=assigned_comp,
            colour_map=colour_map,
            rendered=rendered,
            extents=extents,
            title_prefix=title_prefix,
        )
        all_path = out_dir / f"cluster_{cluster_id}_all.{args.ext}"
        fig_all.savefig(all_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_all)
        print(f"  Saved all-latents → {all_path}")

        # ---- per-latent solo figures (rendered fresh at solo_opacity) ----
        if not args.no_solo:
            solo_dir = out_dir / f"cluster_{cluster_id}"
            solo_dir.mkdir(exist_ok=True)
            for global_idx, epdf in epdf_dict.items():
                fig_solo = plot_solo_latent(
                    cluster_id=cluster_id,
                    latent_idx=global_idx,
                    epdf=epdf,
                    component_order=component_order,
                    assigned_comp=assigned_comp,
                    colour_map=colour_map,
                    geom_grids=geom_grids,
                    extents=extents,
                    solo_opacity=args.solo_opacity,
                    max_kde_samples=args.max_kde_samples,
                    title_prefix=title_prefix,
                )
                solo_path = solo_dir / f"latent_{global_idx}.{args.ext}"
                fig_solo.savefig(solo_path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig_solo)
            print(f"  Saved {len(epdf_dict)} solo figures → {solo_dir}/")

    print(f"\nDone. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
