import os
from typing import Dict, List, Tuple
import itertools

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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _site_keys(metrics_summary: Dict) -> List[str]:
    keys = []
    if "embeddings" in metrics_summary:
        keys.append("embeddings")
    layer_keys = [k for k in metrics_summary.keys() if k.startswith("layer_")]
    # sort by layer index
    def _layer_idx(k: str) -> int:
        try:
            return int(k.split("_")[1])
        except Exception:
            return 1_000_000
    layer_keys.sort(key=_layer_idx)
    keys.extend(layer_keys)
    return keys


def _parse_k_entries(seq_group: Dict) -> List[Tuple[int, Dict]]:
    entries = []
    for kname, rec in seq_group.items():
        if not kname.startswith("k"):
            continue
        try:
            kval = int(kname[1:])
        except Exception:
            continue
        entries.append((kval, rec))
    entries.sort(key=lambda x: x[0])
    return entries


def _parse_lambda_entries(beliefs_group: Dict) -> List[Tuple[float, Dict]]:
    entries = []
    for lname, rec in beliefs_group.items():
        if not lname.startswith("lambda_"):
            continue
        try:
            lval = float(lname.split("lambda_")[1])
        except Exception:
            # try scientific forms like 1e-05 embedded as strings
            try:
                lval = float(lname.replace("lambda_", ""))
            except Exception:
                continue
        entries.append((lval, rec))
    entries.sort(key=lambda x: x[0])
    return entries


def _fit_line(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    # guard for constant arrays to avoid polyfit warnings
    if len(y) < 2 or np.allclose(y, y[0]):
        return np.full_like(y, fill_value=y[-1], dtype=float)
    coeffs = np.polyfit(x, y, 1)
    y_fit = coeffs[0] * x + coeffs[1]
    return y_fit


def generate_last50_loss_plots(metrics_summary: Dict, out_dir: str) -> None:
    """
    Generate line plots of the last 50 loss values with a straight line (values)
    and a dotted linear fit line, grouped by k (for top_k and batch_top_k, sequence/L2)
    and by lambda (for vanilla SAEs under beliefs).

    Saves PNGs to out_dir.
    """
    _ensure_dir(out_dir)
    sites = _site_keys(metrics_summary)

    # Sequence: top_k and batch_top_k (L2 only) — last50_loss arrays
    for site in sites:
        site_data = metrics_summary.get(site, {})
        sequence = site_data.get("sequence", {})

        for sae_type in ["top_k"]:
            if sae_type not in sequence:
                continue
            group = sequence[sae_type]
            k_entries = _parse_k_entries(group)
            if not k_entries:
                continue

            plt.figure(figsize=(8, 5))
            for kval, rec in k_entries:
                series = np.asarray(rec.get("last50_loss", []), dtype=float)
                if series.size == 0:
                    continue
                x = np.arange(series.size)
                plt.plot(x, series, label=f"k={kval}")
                y_fit = _fit_line(series)
                plt.plot(x, y_fit, linestyle=":", color=plt.gca().lines[-1].get_color())

            plt.title(f"Last 50 L2 loss — {site} — {sae_type} (sequence)")
            plt.xlabel("Step (last 50)")
            plt.ylabel("Loss (L2)")
            plt.legend()
            plt.tight_layout()
            fname = f"last50_{site}_{sae_type}_l2_sequence.png"
            plt.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.show()
            plt.close()

    # Vanilla SAEs (beliefs): by lambda; we save both L2 and L1 variants (if only one series is available,
    # it will be the same time series for both saved figures, since metrics only expose one last50 series).
    for site in sites:
        site_data = metrics_summary.get(site, {})
        for data_key, label_suffix in [("beliefs", "beliefs"), ("sequence", "sequence")]:
            lam_group = site_data.get(data_key, {}) if data_key == "beliefs" else site_data.get("sequence", {}).get("vanilla", {})
            # For beliefs: keys are lambda_* mapping to rec; for sequence vanilla: same
            lambda_entries = _parse_lambda_entries(lam_group)
            if not lambda_entries:
                continue

            def _plot_and_save(loss_label: str, suffix: str) -> None:
                plt.figure(figsize=(8, 5))
                for lmbda, rec in lambda_entries:
                    series = np.asarray(rec.get("last50_loss", []), dtype=float)
                    if series.size == 0:
                        continue
                    x = np.arange(series.size)
                    plt.plot(x, series, label=f"λ={lmbda}")
                    y_fit = _fit_line(series)
                    plt.plot(x, y_fit, linestyle=":", color=plt.gca().lines[-1].get_color())

                plt.title(f"Last 50 {loss_label} loss — {site} — vanilla ({suffix})")
                plt.xlabel("Step (last 50)")
                plt.ylabel(f"Loss ({loss_label})")
                plt.legend(ncol=2)
                plt.tight_layout()
                fname = f"last50_{site}_vanilla_{loss_label.lower()}_{suffix}.png"
                plt.savefig(os.path.join(out_dir, fname), dpi=200)
                plt.show()
                plt.close()

            _plot_and_save("L2", label_suffix)
            _plot_and_save("L1", label_suffix)


def generate_l0_bar_plots(metrics_summary: Dict, out_dir: str) -> None:
    """
    Side-by-side bar plots for sequence SAEs (top_k, batch_top_k): avg_last_quarter L2
    across layers, grouped by k. Vanilla L0/L2 bar charts are intentionally omitted.
    """
    _ensure_dir(out_dir)
    sites = _site_keys(metrics_summary)

    # Build per-layer collections first so bars are aligned across layers per group
    layer_like = [k for k in sites if k == "embeddings" or k.startswith("layer_")]

    # top_k and batch_top_k sequence — L2 bars by k, multiple bars per k (one per layer)
    def _bar_by_k_l2(sae_type: str, title_suffix: str, filename_suffix: str) -> None:
        # Determine all k values available across layers to align groups
        all_k: List[int] = []
        per_layer_values: Dict[str, Dict[int, float]] = {}
        for site in layer_like:
            seq = metrics_summary.get(site, {}).get("sequence", {})
            group = seq.get(sae_type, {})
            k_entries = _parse_k_entries(group)
            per_layer_values[site] = {}
            for kval, rec in k_entries:
                l2 = rec.get("avg_last_quarter", {}).get("l2")
                if l2 is None:
                    continue
                per_layer_values[site][kval] = float(l2)
                if kval not in all_k:
                    all_k.append(kval)
        all_k.sort()
        if not all_k:
            return

        x = np.arange(len(all_k), dtype=float)
        width = 0.12 if len(layer_like) > 0 else 0.2
        offsets = (np.arange(len(layer_like)) - (len(layer_like) - 1) / 2.0) * width

        plt.figure(figsize=(max(8, 1.2 * len(all_k)), 5))
        for idx, site in enumerate(layer_like):
            values = [per_layer_values.get(site, {}).get(kv, np.nan) for kv in all_k]
            plt.bar(x + offsets[idx], values, width=width, label=site)

        plt.xticks(x, [f"k={kv}" for kv in all_k])
        plt.ylabel("L2 (avg last quarter)")
        plt.title(f"L2 (avg last quarter) — {title_suffix}")
        plt.legend(ncol=2)
        plt.tight_layout()
        fname = f"bars_{filename_suffix}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.show()
        plt.close()

    _bar_by_k_l2("top_k", "sequence — top_k", "sequence_top_k_avg_l2")


def generate_vanilla_l0_average_line_plots(metrics_summary: Dict, out_dir: str) -> None:
    """
    For vanilla SAEs: plot two lines, one for L0 and one for L2, where each is the
    average across layers of the avg_last_quarter metric, vs lambda value (numeric).
    Saves PNGs.
    """
    _ensure_dir(out_dir)
    sites = _site_keys(metrics_summary)
    layer_sites = [s for s in sites if s.startswith("layer_")]
    beliefs_l0_by_lambda: Dict[float, List[float]] = {}
    beliefs_l2_by_lambda: Dict[float, List[float]] = {}
    seq_l0_by_lambda: Dict[float, List[float]] = {}
    seq_l2_by_lambda: Dict[float, List[float]] = {}

    for site in layer_sites:
        beliefs = metrics_summary.get(site, {}).get("beliefs", {})
        for lmbda, rec in _parse_lambda_entries(beliefs):
            avg = rec.get("avg_last_quarter", {})
            l0 = avg.get("l0")
            l2 = avg.get("l2")
            if l0 is not None:
                beliefs_l0_by_lambda.setdefault(lmbda, []).append(float(l0))
            if l2 is not None:
                beliefs_l2_by_lambda.setdefault(lmbda, []).append(float(l2))

        seq_v = metrics_summary.get(site, {}).get("sequence", {}).get("vanilla", {})
        for lmbda, rec in _parse_lambda_entries(seq_v):
            avg = rec.get("avg_last_quarter", {})
            l0 = avg.get("l0")
            l2 = avg.get("l2")
            if l0 is not None:
                seq_l0_by_lambda.setdefault(lmbda, []).append(float(l0))
            if l2 is not None:
                seq_l2_by_lambda.setdefault(lmbda, []).append(float(l2))

    if beliefs_l0_by_lambda:
        lambdas = sorted(beliefs_l0_by_lambda.keys())
        avg_values = [float(np.nanmean(beliefs_l0_by_lambda[l])) for l in lambdas]
        plt.figure(figsize=(8, 5))
        # draw dashed gray gridlines at y-axis ticks, beneath data
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", color="#cccccc", linewidth=1)
        plt.plot(lambdas, avg_values, marker="o", zorder=1)
        plt.xlabel("λ")
        plt.ylabel("L0 (avg last quarter) — avg across layers")
        plt.title("Vanilla (beliefs): L0 avg across layers vs λ")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "lines_beliefs_vanilla_avg_l0_across_layers.png"), dpi=200)
        plt.show()
        plt.close()

    if beliefs_l2_by_lambda:
        lambdas = sorted(beliefs_l2_by_lambda.keys())
        avg_values = [float(np.nanmean(beliefs_l2_by_lambda[l])) for l in lambdas]
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", color="#cccccc", linewidth=1)
        plt.plot(lambdas, avg_values, marker="o", zorder=1)
        plt.xlabel("λ")
        plt.ylabel("L2 (avg last quarter) — avg across layers")
        plt.title("Vanilla (beliefs): L2 avg across layers vs λ")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "lines_beliefs_vanilla_avg_l2_across_layers.png"), dpi=200)
        plt.show()
        plt.close()

    if seq_l0_by_lambda:
        lambdas = sorted(seq_l0_by_lambda.keys())
        avg_values = [float(np.nanmean(seq_l0_by_lambda[l])) for l in lambdas]
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", color="#cccccc", linewidth=1)
        plt.plot(lambdas, avg_values, marker="o", zorder=1)
        plt.xlabel("λ")
        plt.ylabel("L0 (avg last quarter) — avg across layers")
        plt.title("Vanilla (sequence): L0 avg across layers vs λ")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "lines_sequence_vanilla_avg_l0_across_layers.png"), dpi=200)
        plt.show()
        plt.close()

        # ylim (0,5) variant
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", color="#cccccc", linewidth=1)
        plt.plot(lambdas, avg_values, marker="o", zorder=1)
        plt.xlabel("λ")
        plt.ylabel("L0 (avg last quarter) — avg across layers")
        plt.title("Vanilla (sequence): L0 avg across layers vs λ (ylim 0–8)")
        plt.ylim(0, 8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "lines_sequence_vanilla_avg_l0_across_layers_ylim_0_8.png"), dpi=200)
        plt.show()
        plt.close()

    if seq_l2_by_lambda:
        lambdas = sorted(seq_l2_by_lambda.keys())
        avg_values = [float(np.nanmean(seq_l2_by_lambda[l])) for l in lambdas]
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", color="#cccccc", linewidth=1)
        plt.plot(lambdas, avg_values, marker="o", zorder=1)
        plt.xlabel("λ")
        plt.ylabel("L2 (avg last quarter) — avg across layers")
        plt.title("Vanilla (sequence): L2 avg across layers vs λ")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "lines_sequence_vanilla_avg_l2_across_layers.png"), dpi=200)
        plt.show()
        plt.close()


def analyze_latent_activation_by_sequence(
    model,
    sae,
    mess3,
    seq_len: int = 6,
    device: str = "cpu",
    active_latents: list[int] | None = None,
    hook_name: str = "blocks.3.hook_resid_post",
    threshold: float = 1e-6,
) -> dict:
    """
    Enumerate all sequences of length `seq_len` in Mess3,
    run through model+SAE, and record which latents activate.
    
    Returns:
      results: dict {
        latent_index: [
          {"seq": tuple(ints), "value": float}
        ]
      }
    """

    vocab_size = mess3.vocab_size
    # active_latents can be:
    # - None (analyze all)
    # - list/tuple/array of indices
    # - dict mapping latent_index (as str or int) -> [count, sum]
    if active_latents is None:
        indices = list(range(sae.cfg["dict_size"]))
    elif isinstance(active_latents, dict):
        try:
            indices = [int(k) for k in active_latents.keys()]
        except Exception:
            indices = list(active_latents.keys())
    else:
        indices = [int(i) for i in active_latents]
    results = {int(i): [] for i in indices}

    for seq in itertools.product(range(vocab_size), repeat=seq_len):
        tokens = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None)
            acts = cache[hook_name]  # [1, seq_len, d_model]
            acts = acts.reshape(-1, acts.shape[-1])  # flatten across seq
            sae_out = sae(acts)
            lat_acts = sae_out["feature_acts"]  # [seq_len, dict_size]

        # collapse over time (take max magnitude per latent across positions)
        lat_values = lat_acts.max(dim=0).values.cpu().numpy()

        for li in indices:
            li_int = int(li)
            if li_int < 0 or li_int >= lat_values.shape[0]:
                continue
            if lat_values[li_int] > threshold:
                results[li_int].append({"seq": seq, "value": float(lat_values[li_int])})

    return results

def summarize_latent_sequence_results(results: dict, top_n: int = 10):
    """
    Given results from analyze_latent_activation_by_sequence, print useful summaries.
    """
    summaries = {}
    for li, entries in results.items():
        if not entries:
            continue
        values = np.array([e["value"] for e in entries])
        avg_val = values.mean()
        max_val = values.max()
        top_entries = sorted(entries, key=lambda e: e["value"], reverse=True)[:top_n]
        summaries[li] = {
            "count": len(entries),
            "avg_value": float(avg_val),
            "max_value": float(max_val),
            "top_sequences": [(e["seq"], e["value"]) for e in top_entries],
        }
    return summaries


def plot_latent_histograms(results: dict, out_dir: str, bins: int = 30):
    """
    Save histogram plots of activation magnitudes for each latent.
    """
    _ensure_dir(out_dir)
    for li, entries in results.items():
        if not entries:
            continue
        values = np.array([e["value"] for e in entries])
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=bins, color="steelblue", alpha=0.7)
        plt.xlabel("Activation magnitude (max over seq)")
        plt.ylabel("Frequency")
        plt.title(f"Latent {li}: {len(entries)} sequences")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"latent_{li}_hist.png"), dpi=200)
        plt.show()
        plt.close()



def compute_latent_token_fingerprints(results: dict, vocab_size: int = 3, seq_len: int = 6):
    """
    Build token-level fingerprints for each latent:
      - overall token counts across all active sequences
      - position-specific counts
    """
    fingerprints = {}
    for li, entries in results.items():
        if not entries:
            continue
        token_counts = Counter()
        pos_counts = [Counter() for _ in range(seq_len)]
        for e in entries:
            seq = e["seq"]
            for pos, tok in enumerate(seq):
                token_counts[tok] += 1
                pos_counts[pos][tok] += 1

        # Normalize to probabilities
        total = sum(token_counts.values())
        token_probs = {tok: token_counts[tok] / total for tok in range(vocab_size)}

        pos_probs = []
        for pos in range(seq_len):
            row_total = sum(pos_counts[pos].values())
            if row_total == 0:
                pos_probs.append({tok: 0.0 for tok in range(vocab_size)})
            else:
                pos_probs.append({tok: pos_counts[pos][tok] / row_total for tok in range(vocab_size)})

        fingerprints[li] = {
            "token_probs": token_probs,
            "pos_probs": pos_probs,
        }
    return fingerprints


def plot_token_fingerprints(fingerprints: dict, out_dir: str, vocab_size: int = 3):
    """
    Plot bar charts for token probabilities (overall + per position) for each latent.
    """
    _ensure_dir(out_dir)
    colors = ["red", "green", "blue"][:vocab_size]

    for li, fp in fingerprints.items():
        # Overall token probs
        plt.figure(figsize=(5,4))
        tokens, probs = zip(*sorted(fp["token_probs"].items()))
        plt.bar(tokens, probs, color=colors[:len(tokens)])
        plt.xticks(range(len(tokens)), [f"Tok {t}" for t in tokens])
        plt.ylabel("Probability")
        plt.title(f"Latent {li}: overall token distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"latent_{li}_tokens.png"), dpi=200)
        plt.show()
        plt.close()

        # Position-specific
        seq_len = len(fp["pos_probs"])
        fig, axes = plt.subplots(1, seq_len, figsize=(2*seq_len, 4), sharey=True)
        for pos in range(seq_len):
            probs_dict = fp["pos_probs"][pos]
            axes[pos].bar(range(vocab_size), [probs_dict.get(tok, 0.0) for tok in range(vocab_size)], color=colors)
            axes[pos].set_xticks(range(vocab_size))
            axes[pos].set_xticklabels([str(t) for t in range(vocab_size)])
            axes[pos].set_title(f"Pos {pos}")
        axes[0].set_ylabel("Probability")
        plt.suptitle(f"Latent {li}: position-specific token probs")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"latent_{li}_pos_tokens.png"), dpi=200)
        plt.show()
        plt.close()

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

    def _detect_triangle_vertices(self, tol=0.02):
        """Infer simplex triangle vertices from this latent's coords (robust to PCA noise)."""
        if self.coords.shape[0] < 3:
            return None

        all_coords = self.coords
        xmin, xmax = np.min(all_coords[:,0]), np.max(all_coords[:,0])
        ymin, ymax = np.min(all_coords[:,1]), np.max(all_coords[:,1])

        candidates = []

        # Near xmin
        mask = np.isclose(all_coords[:,0], xmin, atol=tol)
        if mask.any():
            ys = all_coords[mask,1]
            candidates.append([xmin, ys.min()])
            candidates.append([xmin, ys.max()])

        # Near xmax
        mask = np.isclose(all_coords[:,0], xmax, atol=tol)
        if mask.any():
            ys = all_coords[mask,1]
            candidates.append([xmax, ys.min()])
            candidates.append([xmax, ys.max()])

        # Near ymin
        mask = np.isclose(all_coords[:,1], ymin, atol=tol)
        if mask.any():
            xs = all_coords[mask,0]
            candidates.append([xs.min(), ymin])
            candidates.append([xs.max(), ymin])

        # Near ymax
        mask = np.isclose(all_coords[:,1], ymax, atol=tol)
        if mask.any():
            xs = all_coords[mask,0]
            candidates.append([xs.min(), ymax])
            candidates.append([xs.max(), ymax])

        candidates = np.unique(np.array(candidates), axis=0)

        if candidates.shape[0] < 3:
            return None

        hull = ConvexHull(candidates)
        verts = candidates[hull.vertices]

        # Force to exactly 3 vertices if noisy hull produced >3
        if verts.shape[0] > 3:
            dists = np.sum((verts[:, None, :] - verts[None, :, :])**2, axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)

            def area(p, q, r):
                return abs(0.5 * ((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])))

            best_k, best_area = None, -1
            for k in range(len(verts)):
                if k in (i, j):
                    continue
                A = area(verts[i], verts[j], verts[k])
                if A > best_area:
                    best_area, best_k = A, k

            verts = np.array([verts[i], verts[j], verts[best_k]])

        return verts



def plot_epdfs(epdfs, mode="3d", grid_size=80):
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

    # Use triangle from first EPDF
    triangle_vertices = epdfs[0].triangle_vertices
    if triangle_vertices is None:
        raise ValueError("Could not detect triangle vertices for plotting.")

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
    colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan"]

    for idx, epdf in enumerate(epdfs):
        if epdf.kde is None:
            epdf.fit_kde()
        xy = np.vstack([pts_x, pts_y])
        z = epdf.kde(xy)

        if z is None or z.size == 0:
            continue
        color = colors[idx % len(colors)]

        # Normalize inside triangle
        z_min, z_max = np.min(z), np.max(z)
        z_norm = (z - z_min) / (z_max - z_min + 1e-12)

        if mode == "3d":
            fig.add_trace(go.Mesh3d(
                x=pts_x, y=pts_y, z=z_norm,
                color=color,
                opacity=0.6,
                name=f"Latent {epdf.latent_idx}",
                alphahull=0
            ))
        elif mode == "transparency":
            fig.add_trace(go.Scatter(
                x=pts_x, y=pts_y,
                mode="markers",
                marker=dict(size=5, color=color, opacity=z_norm),
                name=f"Latent {epdf.latent_idx}"
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
                name=f"Latent {epdf.latent_idx}",
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
        height=700, width=800
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
