#!/usr/bin/env python3
"""Generate resume scripts for interrupted sweep runs by inspecting MLflow."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "/Shared/mp_clustering_sweep"
RUN_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "run_scripts"
REFINISH_PREFIX = "sweep_refinish_"

CLUSTER_CONFIGS = [
    "auto_0.83_2.0",
    "auto_0.83_2.5",
    "auto_0.9_2.0",
    "auto_0.9_2.5",
    "manual_6",
    "manual_9",
]
DEDUP_THRESHOLDS = ["0.96", "0.99"]
LATENT_ACTS = ["0.01", "0.001"]
SIM_METRICS = ["cosine", "euclidean"]


@dataclass(frozen=True)
class SweepSpec:
    script_path: Path
    sweep_type: str  # "topk" or "vanilla"
    sae_value: str
    per_point_threshold: str

    @property
    def basename(self) -> str:
        return self.script_path.name

    @property
    def refinish_name(self) -> str:
        return f"{REFINISH_PREFIX}{self.basename}"


@dataclass
class Combination:
    index: int
    cluster_config: str
    dedup: str
    latent: str
    sim_metric: str
    run_name_base: str


def parse_sweep_scripts() -> List[SweepSpec]:
    specs: List[SweepSpec] = []
    pattern = re.compile(r"sweep_(?P<kind>topk|vanilla)_(?P<name>.+)\.sh")

    for path in sorted(RUN_SCRIPTS_DIR.glob("sweep_*.sh")):
        if path.name.startswith(REFINISH_PREFIX):
            continue

        match = pattern.fullmatch(path.name)
        if not match:
            continue

        kind = match.group("kind")
        name_part = match.group("name")
        if kind == "topk":
            match_val = re.match(r"k(?P<value>[0-9]+)_ppf_(?P<ppf>[\d.]+)", name_part)
            if not match_val:
                continue
            sae_value = match_val.group("value")
            ppf = match_val.group("ppf")
        else:
            match_val = re.match(r"l(?P<value>[\d.]+)_ppf_(?P<ppf>[\d.]+)", name_part)
            if not match_val:
                continue
            sae_value = match_val.group("value")
            ppf = match_val.group("ppf")

        specs.append(
            SweepSpec(
                script_path=path,
                sweep_type=kind,
                sae_value=sae_value,
                per_point_threshold=ppf,
            )
        )

    return specs


def build_combinations(spec: SweepSpec) -> List[Combination]:
    combos: List[Combination] = []
    index = 0

    for cluster_config in CLUSTER_CONFIGS:
        for dedup in DEDUP_THRESHOLDS:
            for latent in LATENT_ACTS:
                for sim in SIM_METRICS:
                    if cluster_config.startswith("auto_"):
                        _, var, gap = cluster_config.split("_")
                        cluster_str = f"auto_v{var}_g{gap}"
                    else:
                        _, n_clusters = cluster_config.split("_")
                        cluster_str = f"manual_n{n_clusters}"

                    if spec.sweep_type == "topk":
                        run_base = (
                            f"topk_k{spec.sae_value}_"
                            f"{cluster_str}_"
                            f"dedup{dedup}_"
                            f"lat{latent}_"
                            f"{sim}_"
                            f"ppf{spec.per_point_threshold}"
                        )
                    else:
                        run_base = (
                            f"vanilla_l{spec.sae_value}_"
                            f"{cluster_str}_"
                            f"dedup{dedup}_"
                            f"lat{latent}_"
                            f"{sim}_"
                            f"ppf{spec.per_point_threshold}"
                        )

                    combos.append(
                        Combination(
                            index=index,
                            cluster_config=cluster_config,
                            dedup=dedup,
                            latent=latent,
                            sim_metric=sim,
                            run_name_base=run_base,
                        )
                    )
                    index += 1

    return combos


def collect_runs_by_base(client: MlflowClient, experiment_id: str) -> dict[str, list[mlflow.entities.Run]]:
    runs_by_base: dict[str, list[mlflow.entities.Run]] = {}
    page_token: Optional[str] = None
    total = 0

    while True:
        runs = client.search_runs(
            [experiment_id],
            filter_string="",
            max_results=1000,
            page_token=page_token,
            order_by=["attributes.start_time ASC"],
        )

        if not runs:
            break

        total += len(runs)

        for run in runs:
            run_name = run.info.run_name or run.data.tags.get("mlflow.runName")
            if not run_name:
                continue

            base = run_name
            match_ts = re.match(r"(.+)_\d{8}-\d{6}$", run_name)
            if match_ts:
                base = match_ts.group(1)

            runs_by_base.setdefault(base, []).append(run)

        next_token = getattr(runs, "token", None)
        page_token = next_token
        if not page_token:
            break

    print(f"  Retrieved {total} runs from experiment {experiment_id}.")
    return runs_by_base


def determine_resume_index(combos: List[Combination], runs_by_base: dict[str, list[mlflow.entities.Run]]) -> tuple[int, Optional[int]]:
    last_existing = -1
    start_index = len(combos)

    for combo in combos:
        if combo.run_name_base in runs_by_base:
            last_existing = combo.index
            continue

        start_index = combo.index
        break
    else:
        # loop exhausted without break: all combos exist
        start_index = len(combos)

    guaranteed_idx = last_existing if last_existing >= 0 else None
    return start_index, guaranteed_idx


def format_array(values: Iterable[str], quote: bool = False) -> str:
    items = " ".join(f'"{v}"' if quote else v for v in values)
    return f"({items})"


def write_refinish_script(spec: SweepSpec, combos: List[Combination], start_index: int) -> None:
    refinish_path = RUN_SCRIPTS_DIR / spec.refinish_name
    total_runs = len(combos)

    header_comment = (
        f"# Auto-generated on {datetime.utcnow().isoformat()}Z to resume {spec.basename}\n"
        "# START_INDEX indicates how many combinations to skip (0-based).\n"
    )

    if spec.sweep_type == "topk":
        sae_args_line = 'SAE_ARGS="--sae_type top_k --force_k $SAE_VALUE"'
        run_name_template = 'RUN_NAME="topk_k${SAE_VALUE}_${CLUSTER_STR}_dedup${dedup}_lat${latent_act}_${sim_metric}_ppf${PER_POINT_THRESHOLD}"'
        sweep_label = f"Top K k=$SAE_VALUE, per-point filtering=$PER_POINT_THRESHOLD"
    else:
        sae_args_line = 'SAE_ARGS="--sae_type vanilla --force_lambda $SAE_VALUE"'
        run_name_template = 'RUN_NAME="vanilla_l${SAE_VALUE}_${CLUSTER_STR}_dedup${dedup}_lat${latent_act}_${sim_metric}_ppf${PER_POINT_THRESHOLD}"'
        sweep_label = f"Vanilla lambda=$SAE_VALUE, per-point filtering=$PER_POINT_THRESHOLD"

    script = f"""#!/usr/bin/bash

{header_comment}
SAE_TYPE="{spec.sweep_type}"
SAE_VALUE="{spec.sae_value}"
PER_POINT_THRESHOLD="{spec.per_point_threshold}"
START_INDEX={start_index}

# Parameter arrays for sweep (must match original order)
CLUSTER_CONFIGS={format_array(CLUSTER_CONFIGS, quote=True)}
DEDUP_THRESHOLDS={format_array(DEDUP_THRESHOLDS)}
LATENT_ACTS={format_array(LATENT_ACTS)}
SIM_METRICS={format_array(SIM_METRICS, quote=True)}

TOTAL_RUNS={total_runs}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PYTHON_BIN=${{PYTHON_BIN:-python}}

RUN_INDEX=0

echo "Resuming sweep: {sweep_label}"
echo "Total runs: $TOTAL_RUNS (skipping the first $START_INDEX combinations)"
echo "=================================================="

for cluster_config in "${{CLUSTER_CONFIGS[@]}}"; do
  for dedup in "${{DEDUP_THRESHOLDS[@]}}"; do
    for latent_act in "${{LATENT_ACTS[@]}}"; do
      for sim_metric in "${{SIM_METRICS[@]}}"; do
        if (( RUN_INDEX < START_INDEX )); then
          RUN_INDEX=$((RUN_INDEX + 1))
          continue
        fi

        CURRENT_RUN=$((RUN_INDEX + 1))
        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: cluster=$cluster_config, dedup=$dedup, latent=$latent_act, sim=$sim_metric"

        if [[ $cluster_config == auto_* ]]; then
          IFS='_' read -r _ var_thresh gap_thresh <<< "$cluster_config"
          CLUSTER_ARGS="--subspace_variance_threshold $var_thresh --subspace_gap_threshold $gap_thresh"
          CLUSTER_STR="auto_v${{var_thresh}}_g${{gap_thresh}}"
        else
          IFS='_' read -r _ n_clusters <<< "$cluster_config"
          CLUSTER_ARGS="--subspace_n_clusters $n_clusters"
          CLUSTER_STR="manual_n${{n_clusters}}"
        fi

        {run_name_template}
        {sae_args_line}

        START_TIME=$(date +%s)
        ${{PYTHON_BIN}} -u "${{SCRIPT_DIR}}/../fit_mess3_gmg.py" \\
          $SAE_ARGS \\
          $CLUSTER_ARGS \\
          --sae_folder "/workspace/outputs/saes/multipartite_003e" \\
          --metrics_summary "/workspace/outputs/saes/multipartite_003e/metrics_summary.json" \\
          --output_dir "/workspace/outputs/reports/multipartite_003e" \\
          --model_ckpt "/workspace/outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt" \\
          --device "cpu" \\
          --seed 43 \\
          --process_config "${{SCRIPT_DIR}}/../process_configs.json" \\
          --process_config_name "3xmess3_2xtquant_003" \\
          --d_model 128 \\
          --n_heads 4 \\
          --n_layers 3 \\
          --n_ctx 16 \\
          --d_head 32 \\
          --act_fn "relu" \\
          --sim_metric "$sim_metric" \\
          --clustering_method "k_subspaces" \\
          --min_clusters 6 \\
          --max_clusters 12 \\
          --cosine_dedup_threshold $dedup \\
          --sample_sequences 1024 \\
          --max_activations 50000 \\
          --cluster_activation_threshold 1e-6 \\
          --center_decoder_rows \\
          --latent_activity_threshold $latent_act \\
          --latent_activation_eps 1e-6 \\
          --activation_batches 8 \\
          --refine_with_geometries \\
          --geo_include_circle \\
          --geo_filter_metrics gw_full \\
          --geo_sinkhorn_max_iter 5000 \\
          --geo_sinkhorn_epsilon 0.2 \\
          --geo_threshold_mode raw \\
          --geo_per_point_threshold $PER_POINT_THRESHOLD \\
          --log_to_mlflow \\
          --mlflow_experiment "/Shared/mp_clustering_sweep" \\
          --mlflow_run_name_base "$RUN_NAME"

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "Run completed in $((DURATION / 60)) minutes and $((DURATION % 60)) seconds."
        echo "---"

        RUN_INDEX=$((RUN_INDEX + 1))
      done
    done
  done
done

echo ""
echo "=================================================="
echo "Resume sweep finished."
"""

    refinish_path.write_text(script)
    refinish_path.chmod(0o755)


def main() -> None:
    specs = parse_sweep_scripts()
    if not specs:
        print("No sweep scripts found.")
        return

    tracking_uri = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if not tracking_uri or not token:
        raise SystemExit("DATABRICKS_HOST and DATABRICKS_TOKEN environment variables must be set.")

    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"Experiment {EXPERIMENT_NAME} not found on MLflow server.")

    print(f"Inspecting experiment {EXPERIMENT_NAME} (ID: {experiment.experiment_id})")
    runs_by_base = collect_runs_by_base(client, experiment.experiment_id)
    print(f"  Distinct run name prefixes captured: {len(runs_by_base)}")

    for spec in specs:
        combos = build_combinations(spec)
        start_index, guaranteed_idx = determine_resume_index(combos, runs_by_base)

        guaranteed_base = combos[guaranteed_idx].run_name_base if guaranteed_idx is not None else None
        resume_base = combos[start_index].run_name_base if start_index < len(combos) else None

        print(f"\n{spec.basename}:")
        print(f"  Total combos: {len(combos)}")
        print(f"  Resume from index: {start_index} ({resume_base})")
        if guaranteed_base:
            print(f"  Last guaranteed completed: index {guaranteed_idx} ({guaranteed_base})")
        else:
            print("  No completed runs detected; starting from the beginning.")

        if start_index >= len(combos):
            print("  All combinations appear to be completed; skipping refinish script generation.")
            continue

        write_refinish_script(spec, combos, start_index)
        print(f"  Wrote resume script: {spec.refinish_name}")


if __name__ == "__main__":
    main()
