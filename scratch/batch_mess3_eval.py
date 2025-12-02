#!/usr/bin/env python3
"""
Batch evaluation for Mess3-only experiments.

For each directory named mess3_x_<x>_a_<a> (across checkpoints/saes/reports),
this script:
  1. Runs model accuracy evaluation using scratch.evaluate_model_accuracy.
  2. Computes conditional entropy of the Mess3 generator using scratch.conditional_entropy.

Outputs aggregated results to JSON/CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scratch.evaluate_model_accuracy import evaluate_model_accuracy
from scratch.conditional_entropy import compute_conditional_entropy


M_PATTERN = re.compile(r"mess3_x_([0-9.]+)_a_([0-9.]+)")
_CONFIG_CACHE: Dict[Path, Dict[str, Any]] = {}


def parse_x_a(folder_name: str) -> Tuple[float, float]:
    match = M_PATTERN.fullmatch(folder_name)
    if not match:
        raise ValueError(f"Folder name '{folder_name}' does not match mess3_x_* pattern")
    x_str, a_str = match.groups()
    return float(x_str), float(a_str)


def list_mess3_runs(base_dir: Path) -> List[str]:
    runs = set()
    for subdir in base_dir.glob("mess3_x_*"):
        if subdir.is_dir() and M_PATTERN.fullmatch(subdir.name):
            runs.add(subdir.name)
    return sorted(runs)


def gather_runs() -> List[str]:
    roots = [
        Path("outputs/checkpoints"),
        Path("outputs/saes"),
        Path("outputs/reports"),
    ]
    run_names = set()
    for root in roots:
        if not root.exists():
            continue
        run_names.update(list_mess3_runs(root))
    return sorted(run_names)


def find_checkpoint(run_name: str) -> Path:
    ckpt_dir = Path("outputs/checkpoints") / run_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing for {run_name}")
    candidates = sorted(ckpt_dir.glob("checkpoint_step_*best.pt"), reverse=True)
    if not candidates:
        candidates = sorted(ckpt_dir.glob("checkpoint_step_*.pt"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found in {ckpt_dir}")
    return candidates[0]


def find_sae_folder(run_name: str) -> Path:
    sae_dir = Path("outputs/saes") / run_name
    if not sae_dir.exists():
        raise FileNotFoundError(f"SAE folder missing for {run_name}")
    return sae_dir


def find_report_dir(run_name: str) -> Path:
    report_dir = Path("outputs/reports") / run_name
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory missing for {run_name}")
    return report_dir


def parse_x_a(folder_name: str) -> Tuple[float, float]:
    match = M_PATTERN.fullmatch(folder_name)
    if not match:
        raise ValueError(f"Folder name '{folder_name}' does not match mess3_x_* pattern")
    raw_x, raw_a = match.groups()
    # Fallback parser if config lookup fails
    value_x = raw_x.replace("..", ".")
    value_a = raw_a.replace("..", ".")
    if value_x.endswith("."):
        value_x = value_x[:-1]
    if value_a.endswith("."):
        value_a = value_a[:-1]
    try:
        return float(value_x), float(value_a)
    except ValueError:
        raise ValueError(f"Could not parse x/a from folder name '{folder_name}'")


def get_params_from_config(run_name: str, config_path: Path) -> Tuple[float, float]:
    config_path = config_path.resolve()
    if config_path not in _CONFIG_CACHE:
        with config_path.open("r", encoding="utf-8") as f:
            _CONFIG_CACHE[config_path] = json.load(f)
    data = _CONFIG_CACHE[config_path]
    if run_name not in data:
        raise KeyError(f"{run_name} not found in {config_path}")
    entry = data[run_name]
    components = entry if isinstance(entry, list) else [entry]
    for comp in components:
        if isinstance(comp, dict) and comp.get("type") == "mess3":
            params = comp.get("params") or {}
            if "x" in params and "a" in params:
                return float(params["x"]), float(params["a"])
    raise ValueError(f"Could not extract mess3 parameters for {run_name} from config")


def run_single_evaluation(run_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.process_config)
    try:
        x_val, a_val = get_params_from_config(run_name, config_path)
    except Exception:
        x_val, a_val = parse_x_a(run_name)
    ckpt_path = find_checkpoint(run_name)

    eval_args = dict(
        model_ckpt=str(ckpt_path),
        process_config=args.process_config,
        process_config_name=run_name,
        process_preset=None,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_head=args.d_head,
        n_ctx=args.n_ctx,
        d_vocab=None,
        act_fn=args.act_fn,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        seq_len=args.seq_len,
        seed=args.seed,
        device=args.device,
        output_dir=str(args.output_dir / run_name) if args.save_artifacts else "",
    )
    eval_results = evaluate_model_accuracy(**eval_args)

    entropy_results = compute_conditional_entropy(
        "mess3",
        seq_length=args.entropy_seq_length,
        n_sequences=args.entropy_n_sequences,
        seed=args.seed,
        a=a_val,
        x=x_val,
    )

    return {
        "run_name": run_name,
        "a": a_val,
        "x": x_val,
        "checkpoint": str(ckpt_path),
        "model_metrics": eval_results["overall_metrics"],
        "entropy": entropy_results,
    }


def load_reconstruction_metrics(run_name: str, site: str) -> Dict[int, float] | None:
    recon_path = Path("outputs/saes") / run_name / "reconstruction_errors.json"
    if not recon_path.exists():
        return None
    with recon_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        topk = data["reconstruction_mse"][site]["sequence"]["top_k"]
    except KeyError:
        return None
    metrics: Dict[int, float] = {}
    for key, value in topk.items():
        if key.startswith("k"):
            try:
                k_val = int(key[1:])
            except ValueError:
                continue
            metrics[k_val] = float(value)
    return metrics or None


def compute_elbow_distances(k_to_error: Dict[int, float]) -> Dict[str, Any]:
    if not k_to_error:
        return {"winner": None, "distances": {}, "points": []}
    points = sorted(k_to_error.items())
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    if len(points) == 1:
        return {"winner": int(points[0][0]), "distances": {int(points[0][0]): 0.0}, "points": points}
    p1 = np.array([xs[0], ys[0]], dtype=np.float64)
    p2 = np.array([xs[-1], ys[-1]], dtype=np.float64)
    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)
    distances: Dict[int, float] = {}
    for k, err in points:
        point = np.array([k, err], dtype=np.float64)
        if line_norm == 0:
            dist = float(abs(err - ys[0]))
        else:
            dist = float(abs(np.cross(line_vec, point - p1)) / line_norm)
        distances[int(k)] = dist
    winner = max(distances, key=lambda kk: distances[kk])
    return {"winner": int(winner), "distances": distances, "points": [(int(k), float(v)) for k, v in points]}


def build_summary_rows(aggregated: List[Dict[str, Any]], site: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in aggregated:
        run = entry.get("run_name")
        metrics = load_reconstruction_metrics(run, site)
        if not metrics or 3 not in metrics:
            continue
        k3_val = metrics[3]
        min_k, min_val = min(metrics.items(), key=lambda kv: kv[1])
        delta = k3_val - min_val
        delta_ratio = delta / k3_val if k3_val != 0 else float("inf")
        elbow = compute_elbow_distances(metrics)
        model_metrics = entry.get("model_metrics", {})
        accuracy = model_metrics.get("accuracy")
        entropy_bits = entry.get("entropy", {}).get("conditional_entropy_bits")
        rows.append(
            {
                "run_name": run,
                "accuracy": accuracy,
                "conditional_entropy_bits": entropy_bits,
                "k3_error": k3_val,
                "min_error": min_val,
                "min_k": min_k,
                "delta": delta,
                "delta_ratio": delta_ratio,
                "elbow": elbow,
            }
        )
    return rows


def plot_scatter_relationships(rows: List[Dict[str, Any]], output_path: Path) -> None:
    accuracies = np.array([row["accuracy"] for row in rows], dtype=np.float64)
    entropies = np.array([row["conditional_entropy_bits"] for row in rows], dtype=np.float64)
    delta = np.array([row["delta"] for row in rows], dtype=np.float64)
    delta_ratio = np.array([row["delta_ratio"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(accuracies, delta)
    axes[0, 0].set_xlabel("Model accuracy")
    axes[0, 0].set_ylabel("k=3 error - min error")
    axes[0, 0].set_title("Accuracy vs delta")

    axes[0, 1].scatter(accuracies, delta_ratio)
    axes[0, 1].set_xlabel("Model accuracy")
    axes[0, 1].set_ylabel("Delta / k=3 error")
    axes[0, 1].set_title("Accuracy vs relative delta")

    axes[1, 0].scatter(entropies, delta)
    axes[1, 0].set_xlabel("Conditional entropy (bits)")
    axes[1, 0].set_ylabel("k=3 error - min error")
    axes[1, 0].set_title("Cond. entropy vs delta")

    axes[1, 1].scatter(entropies, delta_ratio)
    axes[1, 1].set_xlabel("Conditional entropy (bits)")
    axes[1, 1].set_ylabel("Delta / k=3 error")
    axes[1, 1].set_title("Cond. entropy vs relative delta")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Mess3 model evaluation")
    parser.add_argument("--process_config", type=str, default="process_configs.json")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_batches", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--entropy-seq-length", type=int, default=5000)
    parser.add_argument("--entropy-n-sequences", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="outputs/mess3_batch_eval")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Optional subset of run names to process")
    parser.add_argument("--reuse-summary", action="store_true",
                        help="Skip accuracy/entropy evaluation if summary file already exists")
    parser.add_argument("--elbow-site", type=str, default="layer_1",
                        help="Site key (e.g., layer_1) for reconstruction error analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_names = args.runs or gather_runs()
    if not run_names:
        print("No mess3 runs found.")
        return

    summary_path = args.output_dir / "mess3_batch_results.json"
    if args.reuse_summary and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            aggregated = json.load(f)
        print(f"Loaded summary from {summary_path}")
    else:
        aggregated: List[Dict[str, Any]] = []
        for run in run_names:
            print(f"Processing {run} ...")
            res = run_single_evaluation(run, args)
            aggregated.append(res)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        print(f"Saved aggregated results to {summary_path}")

    summaries = build_summary_rows(aggregated, args.elbow_site)
    summary_table_path = args.output_dir / "mess3_topk_summary.json"
    with summary_table_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved top-k summary to {summary_table_path}")

    if summaries:
        plot_path = args.output_dir / "mess3_accuracy_entropy_scatter.png"
        plot_scatter_relationships(summaries, plot_path)
        print(f"Wrote scatter plots to {plot_path}")
    else:
        print("No valid reconstruction summaries found; skipping plotting.")


if __name__ == "__main__":
    main()
