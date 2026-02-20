#!/usr/bin/env python3
"""
Generate elbow plots for the top-N clusters by elbow strength.

Saves one PNG per cluster showing reconstruction loss and archetypal loss
curves vs k, with the elbow point marked. Designed for manual browsing to
find clusters with genuine 3+ vertex structure.

Output structure:
    output_dir/
        clusters_512/
            cluster_0042_recon0.312_arch0.287.png
            ...
        clusters_768/
            ...

Usage:
    python scripts/plot_elbow_browser.py
    python scripts/plot_elbow_browser.py --top_n 50
    python scripts/plot_elbow_browser.py --n_clusters 512 --top_n 200
    python scripts/plot_elbow_browser.py --csv_dir outputs/real_data_analysis_canonical --top_n 100
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# =============================================================================
# Plotting
# =============================================================================

def plot_cluster_elbows(cluster_id, cluster_df, output_path):
    """Generate and save a 2-panel elbow plot for one cluster."""
    cluster_df = cluster_df.sort_values("aanet_k")
    k_vals = cluster_df["aanet_k"].values

    recon_losses = cluster_df["aanet_recon_loss"].values
    arch_losses  = cluster_df["aanet_archetypal_loss"].values

    # Elbow metadata (same for all rows of this cluster)
    row0 = cluster_df.iloc[0]
    recon_elbow_k  = row0.get("aanet_recon_loss_elbow_k",  None)
    arch_elbow_k   = row0.get("aanet_archetypal_loss_elbow_k", None)
    recon_strength = row0.get("aanet_recon_loss_elbow_strength",  float("nan"))
    arch_strength  = row0.get("aanet_archetypal_loss_elbow_strength", float("nan"))
    n_latents      = int(row0.get("n_latents", 0))
    is_monotonic_r = row0.get("recon_is_monotonic", None)
    is_monotonic_a = row0.get("arch_is_monotonic",  None)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"Cluster {cluster_id}  |  n_latents={n_latents}  |  "
        f"recon_str={recon_strength:.3f}  arch_str={arch_strength:.3f}",
        fontsize=11,
    )

    for ax, losses, elbow_k, strength, label, color in [
        (axes[0], recon_losses, recon_elbow_k, recon_strength, "Reconstruction loss", "steelblue"),
        (axes[1], arch_losses,  arch_elbow_k,  arch_strength,  "Archetypal loss",     "darkorange"),
    ]:
        ax.plot(k_vals, losses, "o-", color=color, linewidth=2, markersize=6)

        if elbow_k is not None and not np.isnan(float(elbow_k)):
            ek = int(elbow_k)
            ax.axvline(x=ek, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
            # Mark the elbow point itself
            mask = k_vals == ek
            if mask.any():
                ax.plot(k_vals[mask], losses[mask], "o", color="red", markersize=10,
                        zorder=5, label=f"elbow k={ek}  str={strength:.3f}")
            ax.legend(fontsize=8, loc="upper right")

        ax.set_xlabel("k (vertices)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(k_vals)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Selection
# =============================================================================

def select_top_clusters(df, top_n):
    """Return set of cluster_ids in top-N by recon strength OR top-N by arch strength.

    Uses per-cluster elbow strength (same for all k rows; take first occurrence).
    """
    per_cluster = (
        df.groupby("cluster_id")
        .first()
        .reset_index()[
            ["cluster_id", "aanet_recon_loss_elbow_strength", "aanet_archetypal_loss_elbow_strength"]
        ]
    )

    top_recon = set(
        per_cluster.nlargest(top_n, "aanet_recon_loss_elbow_strength")["cluster_id"]
    )
    top_arch = set(
        per_cluster.nlargest(top_n, "aanet_archetypal_loss_elbow_strength")["cluster_id"]
    )

    selected = top_recon | top_arch
    print(f"  Top {top_n} by recon: {len(top_recon)} clusters")
    print(f"  Top {top_n} by arch:  {len(top_arch)} clusters")
    print(f"  Union:               {len(selected)} unique clusters to plot")
    return selected, per_cluster


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate elbow plot browser PNGs")
    parser.add_argument("--csv_dir", type=str,
                        default="outputs/real_data_analysis_canonical",
                        help="Directory containing clusters_{N}/consolidated_metrics_n{N}.csv")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/elbow_plots",
                        help="Root output directory")
    parser.add_argument("--n_clusters", type=str, default="512,768",
                        help="Comma-separated list of n_clusters values to process")
    parser.add_argument("--top_n", type=int, default=100,
                        help="Top-N clusters by elbow strength (applied separately to recon and arch)")
    return parser.parse_args()


def main():
    args = parse_args()

    csv_dir    = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    n_clusters_list = [int(n.strip()) for n in args.n_clusters.split(",")]

    for n_clusters in n_clusters_list:
        csv_path = csv_dir / f"clusters_{n_clusters}" / f"consolidated_metrics_n{n_clusters}.csv"
        if not csv_path.exists():
            print(f"\nSkipping n_clusters={n_clusters}: CSV not found at {csv_path}")
            continue

        print(f"\n{'='*60}")
        print(f"n_clusters={n_clusters}  ({csv_path})")
        print(f"{'='*60}")

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows, {df['cluster_id'].nunique()} unique clusters")

        # Select clusters
        selected_ids, per_cluster = select_top_clusters(df, args.top_n)

        # Sort for deterministic output: by recon strength descending
        strength_order = (
            per_cluster[per_cluster["cluster_id"].isin(selected_ids)]
            .sort_values("aanet_recon_loss_elbow_strength", ascending=False)["cluster_id"]
            .tolist()
        )

        # Output directory for this n_clusters
        out_subdir = output_dir / f"clusters_{n_clusters}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        print(f"  Saving plots to: {out_subdir}")

        for i, cluster_id in enumerate(strength_order):
            cluster_df = df[df["cluster_id"] == cluster_id]

            row0 = cluster_df.iloc[0]
            rs = row0.get("aanet_recon_loss_elbow_strength", 0)
            as_ = row0.get("aanet_archetypal_loss_elbow_strength", 0)

            fname = f"cluster_{cluster_id:04d}_recon{rs:.3f}_arch{as_:.3f}.png"
            out_path = out_subdir / fname

            plot_cluster_elbows(cluster_id, cluster_df, out_path)

            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{len(strength_order)} done")

        print(f"  Saved {len(strength_order)} plots to {out_subdir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
