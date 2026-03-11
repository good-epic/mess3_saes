#!/usr/bin/env python3
"""
Remake all latent-spatial visualisations from saved data — no GPU required.

Reproduces the original visual style (KDE-on-grid, smooth blobs) but with:
  - A consistent, distinguishable colour assigned to each latent across ALL figures
    in a cluster (all_latents, centroid_scatter, vertex_latent_pies, per-latent)

Produces per cluster:
  all_latents.{ext}                    — all latents overlaid, KDE density coloured
  cluster_{key}_centroid_scatter.{ext} — each latent dot at activation-weighted centroid
  cluster_{key}_vertex_latent_pies.{ext}
  latent_{idx}.{ext}                   — per-latent KDE density

Usage
-----
python validation/remake_latent_spatial.py \\
    --clusters 512_17,512_22,... \\
    --simplex_dir  outputs/simplex_samples \\
    --vertex_dir   outputs/selected_clusters_broad_2 \\
    --output_dir   outputs/validation/latent_spatial_v2 \\
    [--grid_size 100] [--dpi 150] [--ext jpg]
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Triangle geometry
# ---------------------------------------------------------------------------

# Vertices: V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)  — matches epdf_utils
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])


def bary_to_xy(bary: np.ndarray) -> np.ndarray:
    bary = np.asarray(bary, dtype=float)
    if bary.ndim == 1:
        bary = bary[None]
    return bary @ _V


def triangle_mask(img_size: int):
    """Return (xx, yy, inside) for a regular pixel grid covering the triangle bbox."""
    sqrt3_2 = math.sqrt(3) / 2
    xs = np.linspace(0.0, 1.0, img_size)
    ys = np.linspace(0.0, sqrt3_2, img_size)
    xx, yy = np.meshgrid(xs, ys)
    s3 = math.sqrt(3)
    inside = (yy >= 0) & (yy <= xx * s3) & (yy <= s3 * (1.0 - xx))
    return xx, yy, inside


def draw_triangle(ax, lw=1.5, pad=0.06):
    loop = np.vstack([_V, _V[:1]])
    ax.plot(loop[:, 0], loop[:, 1], "k-", lw=lw, zorder=20)
    offsets = [(-pad, -pad * 0.8), (1 + pad * 0.5, -pad * 0.8),
               (0.5, _V[2, 1] + pad * 0.6)]
    for label, (ox, oy) in zip(("V0", "V1", "V2"), offsets):
        ax.text(ox, oy, label, fontsize=9, ha="center", va="center", color="#333333",
                zorder=21)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.14, _V[2, 1] + 0.14)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def build_palette(n: int) -> list:
    """n RGBA colours that are visually distinguishable (works up to ~30)."""
    import colorsys
    colours = []
    for i in range(n):
        hue = (i * 2 / n) % 1.0        # step by 2 so adjacent indices differ in hue
        sat, val = (0.85, 0.85) if i % 2 == 0 else (0.65, 0.60)
        colours.append(colorsys.hsv_to_rgb(hue, sat, val))  # (r, g, b)
    return colours


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse(x):
    return json.loads(x) if isinstance(x, str) else x


def load_simplex_samples(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["barycentric_coords"] = _parse(rec["barycentric_coords"])
                rec["latent_acts"] = _parse(rec["latent_acts"])
                yield rec


def load_arrays(sample_path: Path, n_latents: int, k: int):
    """Return bary (N, k) and acts (N, n_latents) as float32."""
    barys, actss = [], []
    for rec in load_simplex_samples(sample_path):
        b = rec["barycentric_coords"]
        a = rec["latent_acts"]
        if len(b) == k and len(a) == n_latents:
            barys.append(b)
            actss.append(a)
    return np.array(barys, np.float32), np.array(actss, np.float32)


def load_vertex_acts(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["vertex_id"] = int(rec["vertex_id"])
                rec["barycentric_coords"] = _parse(rec["barycentric_coords"])
                rec["latent_acts"] = _parse(rec["latent_acts"])
                yield rec


# ---------------------------------------------------------------------------
# KDE rendering helpers
# ---------------------------------------------------------------------------

def fit_kde(xy: np.ndarray, weights: np.ndarray):
    """Fit a 2-D weighted Gaussian KDE. Returns None if too few points."""
    if len(xy) < 3:
        return None
    try:
        return gaussian_kde(xy.T, weights=weights)
    except Exception:
        return None


def kde_to_rgba(kde, color_rgb, img_size: int, base_opacity: float = 0.6):
    """Render a KDE as an RGBA image array suitable for imshow.

    Outside the triangle is fully transparent (alpha=0).
    Inside, alpha ∝ normalised KDE density * base_opacity.
    Resolution is img_size × img_size pixels — independent of DPI.
    """
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
    rgba[~inside] = 0.0   # transparent outside
    return rgba


def imshow_kde(ax, kde, color_rgb, img_size: int, base_opacity: float = 0.6):
    """Render one latent's KDE as a smooth imshow layer on ax."""
    sqrt3_2 = math.sqrt(3) / 2
    rgba = kde_to_rgba(kde, color_rgb, img_size, base_opacity)
    ax.imshow(rgba, extent=[0.0, 1.0, 0.0, sqrt3_2],
              origin="lower", aspect="equal", interpolation="bilinear",
              zorder=5)


# ---------------------------------------------------------------------------
# Figure 1: all_latents
# ---------------------------------------------------------------------------

def plot_all_latents(bary, acts, latent_indices, colours, cluster_key,
                     out_dir: Path, ext: str, dpi: int, grid_size: int):
    xy = bary_to_xy(bary)
    act_fracs = (acts > 0).mean(axis=0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.set_facecolor("white")

    legend_handles = []
    for li, (idx, col) in enumerate(zip(latent_indices, colours)):
        lat_acts = acts[:, li].astype(np.float64)
        active = lat_acts > 0
        if not active.any():
            continue

        kde = fit_kde(xy[active], lat_acts[active])
        if kde is None:
            continue

        imshow_kde(ax, kde, col, img_size=grid_size * 3)
        handle = mpatches.Patch(color=col,
                                label=f"Latent {idx} (active {act_fracs[li]:.1%})")
        legend_handles.append(handle)

    draw_triangle(ax)
    ax.set_title(f"Cluster {cluster_key}: All latents", fontsize=10)

    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper left",
                   bbox_to_anchor=(1.02, 1.0), ncol=2 if len(legend_handles) > 10 else 1,
                   fontsize=7, framealpha=0.8, borderaxespad=0,
                   title="Latent", title_fontsize=8)

    fig.tight_layout()
    out = out_dir / f"all_latents.{ext}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: per-latent KDE
# ---------------------------------------------------------------------------

def plot_per_latent(bary, acts, latent_indices, colours, cluster_key,
                    out_dir: Path, ext: str, dpi: int, grid_size: int):
    xy = bary_to_xy(bary)

    for li, (idx, col) in enumerate(zip(latent_indices, colours)):
        lat_acts = acts[:, li].astype(np.float64)
        active = lat_acts > 0
        n_active = int(active.sum())
        act_frac = n_active / len(bary)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.set_facecolor("white")

        kde = fit_kde(xy[active], lat_acts[active]) if n_active >= 3 else None
        if kde is not None:
            imshow_kde(ax, kde, col, img_size=grid_size * 3)

        draw_triangle(ax)
        ax.set_title(f"Cluster {cluster_key}: Latent {idx}\n"
                     f"(active {act_frac:.1%} of {len(bary)} samples)", fontsize=9)
        fig.tight_layout()
        out = out_dir / f"latent_{idx}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"    {len(latent_indices)} per-latent KDE figures written")


# ---------------------------------------------------------------------------
# Figure 3: centroid scatter
# ---------------------------------------------------------------------------

def plot_centroid_scatter(bary, acts, latent_indices, colours, cluster_key,
                          out_dir: Path, ext: str, dpi: int):
    n_latents = len(latent_indices)
    k = bary.shape[1]
    weight_sums = np.zeros(n_latents, np.float64)
    wcoord_sums = np.zeros((n_latents, k), np.float64)

    for i in range(len(bary)):
        a = acts[i].astype(np.float64)
        pos = a > 0
        if pos.any():
            weight_sums[pos] += a[pos]
            wcoord_sums[pos] += a[pos, None] * bary[i].astype(np.float64)

    active = weight_sums > 0
    if not active.any():
        return

    centroids = np.zeros((n_latents, k))
    centroids[active] = wcoord_sums[active] / weight_sums[active, None]
    cent_xy = bary_to_xy(centroids[active])

    w = weight_sums[active]
    sizes = (w / w.max()) * 400 + 30
    idx_active = [latent_indices[i] for i, a in enumerate(active) if a]
    col_active = [colours[i] for i, a in enumerate(active) if a]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.set_facecolor("white")
    draw_triangle(ax)

    ax.scatter(cent_xy[:, 0], cent_xy[:, 1], s=sizes, c=col_active,
               zorder=10, edgecolors="black", linewidths=0.5, alpha=0.9)
    for xi, yi, li in zip(cent_xy[:, 0], cent_xy[:, 1], idx_active):
        ax.annotate(str(li), (xi, yi), fontsize=5.5, ha="center", va="center",
                    zorder=15, color="white", fontweight="bold")

    handles = [mpatches.Patch(color=colours[i], label=f"L{latent_indices[i]}")
               for i, a in enumerate(active) if a]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="upper right",
              framealpha=0.75, title="Latent", title_fontsize=7)

    ax.set_title(f"Cluster {cluster_key}: latent centroids\n"
                 f"(size ∝ activation weight; colour = latent identity)", fontsize=9)
    fig.tight_layout()
    out = out_dir / f"cluster_{cluster_key}_centroid_scatter.{ext}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    {out.name}")


# ---------------------------------------------------------------------------
# Figure 4: vertex latent pies
# ---------------------------------------------------------------------------

def _compute_mean_acts(vertex_acts_path: Path, k: int, n_latents: int):
    sum_acts = np.zeros((k, n_latents), np.float64)
    counts   = np.zeros(k, np.int64)
    for rec in load_vertex_acts(vertex_acts_path):
        v = rec["vertex_id"]
        if v >= k:
            continue
        a = np.mean(np.array(rec["latent_acts"], np.float64), axis=0)
        if a.shape[0] != n_latents:
            continue
        sum_acts[v] += a
        counts[v]   += 1
    mean_acts = np.where(counts[:, None] > 0,
                         sum_acts / np.maximum(counts[:, None], 1), 0.0)
    return mean_acts, counts


def _draw_one_pie(ax, acts, n_samples, v_label, colours):
    total = acts.sum()
    seen = set()
    if total <= 0 or n_samples == 0:
        ax.text(0.5, 0.5, f"{v_label}\n(n=0)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#666666")
        ax.axis("off")
        return seen
    thresh = total * 0.01
    active_pos = np.where(acts >= thresh)[0]
    if len(active_pos) > 15:
        active_pos = active_pos[np.argsort(-acts[active_pos])[:15]]
    active_pos = sorted(active_pos.tolist())
    seen.update(active_pos)
    pie_vals   = acts[active_pos]
    other_val  = total - pie_vals.sum()
    pie_colors = [colours[p] for p in active_pos]
    if other_val > 1e-12:
        pie_vals   = np.append(pie_vals, other_val)
        pie_colors.append("#dddddd")
    ax.pie(pie_vals, colors=pie_colors, startangle=90,
           wedgeprops={"linewidth": 0.4, "edgecolor": "white"})
    ax.set_title(f"{v_label}  (n={n_samples})", fontsize=8, pad=3)
    return seen


def plot_vertex_latent_pies(vertex_acts_path: Path, latent_indices, colours,
                             cluster_key: str, k: int, out_dir: Path,
                             ext: str, dpi: int):
    n_latents = len(latent_indices)
    mean_acts, counts = _compute_mean_acts(vertex_acts_path, k, n_latents)
    seen = set()

    if k == 3:
        sqrt3_2 = math.sqrt(3) / 2
        xlim, ylim = (-0.5, 1.5), (-0.45, sqrt3_2 + 0.45)
        ax_w, ax_h = xlim[1] - xlim[0], ylim[1] - ylim[0]

        def to_ax(dx, dy):
            return (dx - xlim[0]) / ax_w, (dy - ylim[0]) / ax_h

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.axis("off")
        loop = np.vstack([_V, _V[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "k-", lw=1.5, zorder=5)

        pie_diam = 0.30
        for v in range(3):
            vax, vay = to_ax(*_V[v])
            inset = ax.inset_axes([vax - pie_diam/2, vay - pie_diam/2,
                                   pie_diam, pie_diam])
            seen |= _draw_one_pie(inset, mean_acts[v], int(counts[v]),
                                  f"V{v}", colours)
        ax.set_title(f"Cluster {cluster_key}: mean latent activation per vertex\n"
                     f"(slice ∝ mean magnitude; <1% grouped as 'other')", fontsize=10)
    else:
        ncols = min(k, 3)
        nrows = math.ceil(k / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = np.array(axes).reshape(-1)
        for v in range(k):
            seen |= _draw_one_pie(axes[v], mean_acts[v], int(counts[v]),
                                  f"V{v}", colours)
        for v in range(k, len(axes)):
            axes[v].axis("off")
        fig.suptitle(f"Cluster {cluster_key}: mean latent activation per vertex",
                     fontsize=10)

    handles = [mpatches.Patch(color=colours[p], label=f"L{latent_indices[p]}")
               for p in sorted(seen)]
    handles.append(mpatches.Patch(color="#dddddd", label="other (<1%)"))
    fig.legend(handles=handles, loc="lower center", fontsize=7,
               ncol=min(len(handles), 7), bbox_to_anchor=(0.5, 0.0),
               title="Latent", title_fontsize=8, framealpha=0.85)
    fig.tight_layout()
    out = out_dir / f"cluster_{cluster_key}_vertex_latent_pies.{ext}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    {out.name}")


# ---------------------------------------------------------------------------
# Latent-index discovery
# ---------------------------------------------------------------------------

PRIORITY_CLUSTERS = [
    "512_17", "512_22", "512_67", "512_181", "512_229",
    "512_261", "512_471", "512_504",
    "768_140", "768_210", "768_306", "768_581", "768_596",
]


def find_latent_indices(simplex_dir: Path, vertex_dir: Path,
                        n: int, cid: int, k: int, n_latents: int):
    """Try simplex_stats.json then vertex_stats.json; fall back to 0-based."""
    # simplex_stats
    for p in sorted((simplex_dir / f"n{n}").glob(
            f"cluster_{cid}_k{k}_simplex_stats.json")):
        d = json.loads(p.read_text())
        li = d.get("latent_indices")
        if li and len(li) == n_latents:
            return li
    # vertex_stats
    for p in sorted((vertex_dir / f"n{n}").glob(
            f"cluster_{cid}_k{k}_*_vertex_stats.json")):
        d = json.loads(p.read_text())
        li = d.get("latent_indices") or d.get("cluster_latent_indices")
        if li and len(li) == n_latents:
            return li
    return list(range(n_latents))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters",   default=",".join(PRIORITY_CLUSTERS))
    ap.add_argument("--simplex_dir", default="outputs/simplex_samples")
    ap.add_argument("--vertex_dir",  default="outputs/selected_clusters_broad_2")
    ap.add_argument("--output_dir",  default="outputs/validation/latent_spatial_v2")
    ap.add_argument("--grid_size",  type=int, default=100)
    ap.add_argument("--dpi",        type=int, default=150)
    ap.add_argument("--ext",        default="jpg")
    args = ap.parse_args()

    simplex_dir = Path(args.simplex_dir)
    vertex_dir  = Path(args.vertex_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_key in [c.strip() for c in args.clusters.split(",") if c.strip()]:
        n_str, cid_str = cluster_key.split("_")
        n, cid = int(n_str), int(cid_str)

        print(f"\n{'='*60}\nCluster {cluster_key}")

        sample_files = sorted((simplex_dir / f"n{n}").glob(
            f"cluster_{cid}_k*_simplex_samples.jsonl"))
        if not sample_files:
            print("  No simplex_samples file — skipping"); continue

        # Infer k and n_latents from first record
        with open(sample_files[0]) as fh:
            first = json.loads(fh.readline())
        k = len(_parse(first["barycentric_coords"]))
        n_latents = len(_parse(first["latent_acts"]))

        latent_indices = find_latent_indices(
            simplex_dir, vertex_dir, n, cid, k, n_latents)
        colours = build_palette(n_latents)

        print(f"  k={k}, n_latents={n_latents}, indices={latent_indices}")
        print(f"  Loading {sample_files[0].name}...")
        bary, acts = load_arrays(sample_files[0], n_latents, k)
        print(f"  Loaded {len(bary)} samples")

        cluster_dir = output_dir / f"cluster_{cluster_key}"
        cluster_dir.mkdir(exist_ok=True)

        if k == 3:
            plot_all_latents(bary, acts, latent_indices, colours, cluster_key,
                             cluster_dir, args.ext, args.dpi, args.grid_size)
            plot_centroid_scatter(bary, acts, latent_indices, colours, cluster_key,
                                  cluster_dir, args.ext, args.dpi)
            plot_per_latent(bary, acts, latent_indices, colours, cluster_key,
                            cluster_dir, args.ext, args.dpi, args.grid_size)
        else:
            print(f"  k={k}: skipping 2-D triangle figures")

        va_files = sorted((vertex_dir / f"n{n}").glob(
            f"cluster_{cid}_k{k}_*_vertex_samples_with_acts.jsonl"))
        if va_files:
            plot_vertex_latent_pies(va_files[0], latent_indices, colours,
                                    cluster_key, k, cluster_dir, args.ext, args.dpi)
        else:
            print("  No vertex_samples_with_acts file — skipping pies")

    print(f"\nDone.  Output in {output_dir}")


if __name__ == "__main__":
    main()
