#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import Dict, List, Any

import matplotlib

# Use a non-interactive backend suitable for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_curves_files(input_dir: str, pattern: str) -> List[str]:
    search_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob(search_pattern))
    return [f for f in files if f.lower().endswith("_curves.json")]


def load_curves(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def infer_series(curves: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Select series that look like 1D numeric lists of equal length.
    """
    candidate_keys: List[str] = []
    lengths: Dict[str, int] = {}
    for key, value in curves.items():
        if isinstance(value, list) and value and all(
            isinstance(x, (int, float)) for x in value
        ):
            candidate_keys.append(key)
            lengths[key] = len(value)
    if not candidate_keys:
        return {}
    # Keep keys with the most common length to avoid mismatched series
    from collections import Counter

    length_counts = Counter(lengths.values())
    if not length_counts:
        return {}
    target_len, _ = length_counts.most_common(1)[0]
    series = {k: curves[k] for k in candidate_keys if lengths[k] == target_len}
    return series


def make_title_from_filename(filename: str) -> str:
    base = os.path.basename(filename).replace("_curves.json", "")
    return base


def plot_curves_file(path: str, output_dir: str) -> str:
    curves = load_curves(path)
    series = infer_series(curves)
    if not series:
        raise ValueError(f"No plottable series found in {path}")
    # Build x-axis
    length = len(next(iter(series.values())))
    x = list(range(1, length + 1))

    plt.figure(figsize=(10, 6))
    for key, values in series.items():
        plt.plot(x, values, label=key)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(make_title_from_filename(path))
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.basename(path).replace("_curves.json", "_curves.png")
    out_path = os.path.join(output_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot *_curves.json files into PNGs."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_curves.json files",
    )
    parser.add_argument(
        "--pattern",
        default="*_curves.json",
        help="Glob pattern to match curve files (default: *_curves.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write PNGs (default: same as input dir)",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or input_dir)

    files = find_curves_files(input_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files matched {os.path.join(input_dir, args.pattern)}")

    print(f"Found {len(files)} files. Writing PNGs to: {output_dir}")
    written: List[str] = []
    errors: List[str] = []
    for fpath in files:
        try:
            out = plot_curves_file(fpath, output_dir)
            written.append(out)
            print(f"OK  {out}")
        except Exception as e:
            errors.append(f"{fpath}: {e}")
            print(f"ERR {fpath}: {e}")
    print(f"\nWritten {len(written)} PNGs.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(err)


if __name__ == "__main__":
    main()


