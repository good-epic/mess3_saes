#!/usr/bin/env python3
"""Collect reconstruction/archetypal eval metrics from AAnet runs."""
from __future__ import annotations
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect AAnet eval metrics across Mess3 runs")
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/reports"),
                        help="Root directory containing mess3_x_* folders")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/mess3_batch_eval/aanet_eval_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/mess3_batch_eval/aanet_eval_metrics.csv"))
    parser.add_argument("--glob", type=str, default="mess3_x_*",
                        help="Glob pattern under reports-root to match run directories")
    return parser.parse_args()


def collect_metrics(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for run_dir in sorted(args.reports_root.glob(args.glob)):
        aanet_dir = run_dir / "AAnet"
        if not aanet_dir.exists():
            continue
        for layer_dir in sorted(d for d in aanet_dir.iterdir() if d.is_dir()):
            layer_name = layer_dir.name
            for cluster_dir in sorted(d for d in layer_dir.iterdir() if d.is_dir()):
                cluster_name = cluster_dir.name
                for k_dir in sorted(d for d in cluster_dir.iterdir() if d.is_dir() and d.name.startswith("k_")):
                    metrics_file = k_dir / "metrics.json"
                    if not metrics_file.exists():
                        continue
                    record = json.loads(metrics_file.read_text())
                    metrics = record.get("metrics", record)
                    k_value = k_dir.name.split("_", 1)[-1]
                    aggregated.setdefault(run_dir.name, {}).setdefault(layer_name, {})[k_value] = {
                        "cluster": cluster_name,
                        "reconstruction_mse_eval": metrics.get("reconstruction_mse_eval"),
                        "archetypal_loss_eval": metrics.get("archetypal_loss_eval"),
                        "in_simplex_fraction": metrics.get("in_simplex_fraction"),
                    }
    return aggregated


def save_outputs(data: Dict[str, List[Dict[str, object]]], args: argparse.Namespace) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, indent=2))
    fieldnames = ["run_name", "layer", "cluster", "k",
                  "reconstruction_mse_eval", "archetypal_loss_eval", "in_simplex_fraction"]
    flat_rows: List[Dict[str, object]] = []
    for run_name, layers in data.items():
        for layer_name, k_entries in layers.items():
            for k_value, entry in k_entries.items():
                row = {
                    "run_name": run_name,
                    "layer": layer_name,
                    "k": k_value,
                    "cluster": entry.get("cluster"),
                    "reconstruction_mse_eval": entry.get("reconstruction_mse_eval"),
                    "archetypal_loss_eval": entry.get("archetypal_loss_eval"),
                    "in_simplex_fraction": entry.get("in_simplex_fraction"),
                }
                flat_rows.append(row)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def main():
    args = parse_args()
    data = collect_metrics(args)
    save_outputs(data, args)
    total_rows = sum(len(k_entries) for layers in data.values() for k_entries in layers.values())
    print(f"Collected {total_rows} rows across {len(data)} runs -> {args.output_json}")


if __name__ == "__main__":
    main()
