#%%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import numpy as np
import pandas as pd
import ast
import json
import re
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import traceback
from clustering import SimplexGeometry, CircleGeometry, HypersphereGeometry

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - optional dependency
    scipy_stats = None

try:
    from IPython.display import display
except Exception:  # pragma: no cover - optional dependency
    def display(obj):
        print(obj)


DATA_PATH = Path(__file__).resolve().parents[1] / "mlflow_sweep_results_20251110_181737.csv"
df = pd.read_csv(DATA_PATH)

pd.set_option('display.max_columns', None)


def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}


pprint(df.columns.tolist())
pprint(ast.literal_eval(df['best_geometries'].iloc[0]))

# 1. Find relevant columns for each type
types = ["hard", "soft"]
r2_ass_prefix = "assigned_belief_cluster_r2_"
r2_all_prefix = "all_belief_cluster_r2_"
assigned_col_suffix = "_assignments"
col_dict = {}
for t in types:
    col_dict[t] = [f"{r2_ass_prefix}{t}", f"{r2_all_prefix}{t}", f"{t}{assigned_col_suffix}", "best_geometries", "best_geometry_scores", "all_geo_scores"]
for t, colnames in col_dict.items():
    df[colnames[0]] = df[colnames[0]].apply(safe_eval)
    df[colnames[1]] = df[colnames[1]].apply(safe_eval)
    df[colnames[2]] = df[colnames[2]].apply(safe_eval)
    # n_nan = df[colname].apply(lambda x: x is np.nan).sum()
    # n_none = df[colname].apply(lambda x: x is None).sum()
    # n_empty_dict = df[colname].apply(lambda x: isinstance(x, dict) and len(x) == 0).sum()
    # print(f"For column '{colname}': np.nan={n_nan}, None={n_none}, empty dict={{}}={n_empty_dict}")
df['best_geometries'] = df['best_geometries'].apply(safe_eval)
df['best_geometry_scores'] = df['best_geometry_scores'].apply(safe_eval)
df['all_geo_scores'] = df['all_geo_scores'].apply(safe_eval)

#%%
def circle_norm_test(df):
    n_same = 0
    n_changed = 0
    geo_names = ["circle"] + [f"simplex_{k}" for k in range(1, 9)]
    changed_from = {name: 0 for name in geo_names if name != "circle"}
    orig_geos = {name: 0 for name in geo_names}
    changed_components = {
        "hard": {name: 0 for name in ["mess3", "mess3_1", "mess3_2", "tom_quantum", "tom_quantum_1"]},
        "soft": {name: 0 for name in ["mess3", "mess3_1", "mess3_2", "tom_quantum", "tom_quantum_1"]},
    }

    for _, row in df.iterrows():
        geo_scores = row.get("all_geo_scores")
        if not isinstance(geo_scores, dict):
            continue

        hard_assign = row.get("hard_assignments")
        soft_assign = row.get("soft_assignments")

        for cluster_id, stats in geo_scores.items():
            best_geom = stats.get("best_geometry")
            if best_geom is None:
                continue

            orig_geos.setdefault(best_geom, 0)
            orig_geos[best_geom] += 1

            circle_dist = stats.get("circle")
            best_dist = stats.get("best_distance")

            if best_geom == "circle" or circle_dist is None or best_dist is None:
                n_same += 1
                continue

            transformed = circle_dist / 2.0
            if transformed < best_dist:
                n_changed += 1
                changed_from.setdefault(best_geom, 0)
                changed_from[best_geom] += 1

                for label, assignments in (("hard", hard_assign), ("soft", soft_assign)):
                    if not isinstance(assignments, dict):
                        continue
                    matched_component = None
                    for comp_name, cid in assignments.items():
                        if comp_name == "noise":
                            continue
                        if str(cid) == str(cluster_id):
                            matched_component = comp_name
                            break
                    if matched_component is not None:
                        changed_components[label].setdefault(matched_component, 0)
                        changed_components[label][matched_component] += 1
            else:
                n_same += 1

    return n_same, n_changed, changed_from, orig_geos, changed_components

print("\n**********************\n")

n_same, n_changed, changed_from, orig_geos, changed_components = circle_norm_test(df)
print(f"n_same: {n_same}, n_changed: {n_changed}")
print(f"percentage of changed: {n_changed / (n_same + n_changed)}")
print(f"changed_from:")
pprint(changed_from)
print(f"orig_geos:")
pprint(orig_geos)
print("Changed components (hard):")
pprint(changed_components["hard"])
print("Changed components (soft):")
pprint(changed_components["soft"])


#%%

def norm_circle_dist(x):
    for k, v in x.items():
        v["circle"] = v['circle'] / 2.0
        if v['circle'] < v['best_distance']:
            v['best_distance'] = v['circle']
            v['best_geometry'] = "circle"
    return x

df["all_geo_scores"] = df["all_geo_scores"].apply(norm_circle_dist)

#%%

def _merge_best_from_all_geo(row):
    """Update best geometry maps using the detailed all-geometry scores."""
    all_geo = row.get("all_geo_scores", {})
    if not isinstance(all_geo, dict) or len(all_geo) == 0:
        return row["best_geometries"], row["best_geometry_scores"]

    best_geos = dict(row.get("best_geometries") or {})
    best_scores = dict(row.get("best_geometry_scores") or {})

    for cluster_id, geom_stats in all_geo.items():
        if not isinstance(geom_stats, dict):
            continue
        best_geom = geom_stats.get("best_geometry")
        best_dist = geom_stats.get("best_distance")

        if best_geom is not None:
            if best_geos.get(cluster_id) != best_geom:
                best_geos[cluster_id] = best_geom

        if best_dist is not None:
            if best_scores.get(cluster_id) != best_dist:
                best_scores[cluster_id] = best_dist

    return best_geos, best_scores


df[["best_geometries", "best_geometry_scores"]] = df.apply(
    lambda row: pd.Series(_merge_best_from_all_geo(row)), axis=1
)


 
print("\n**********************\n")

for t, colnames in col_dict.items():
    means = []
    for x in df[colnames].to_records(index=False):
        pprint(x[0])
        pprint(x[1])
        pprint(x[2])
        pprint(x[3])
        pprint(x[4])
        print(type(x))
        print(type(x[0]))
        print(type(x[1]))
        print(type(x[2]))
        print(type(x[3]))
        print(type(x[4]))
        break
    break


#%%
# Get counts per column of NaN, None, or "empty" ("", {}, [], (), etc.)

print("\n**********************\n")

def is_empty(val):
    # Treat NaN, None, empty string, empty list, dict, tuple as empty
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str):
        return val.strip() == ""
    if isinstance(val, (list, dict, tuple, set)) and len(val) == 0:
        return True
    return False

empty_counts = {}
for col in df.columns:
    cnt = df[col].apply(is_empty).sum()
    empty_counts[col] = cnt

# Print out the count of "empty" values per column
print("\nEmpty (NaN/None/empty string/empty list/dict/tuple/set) counts per column:")
for col, cnt in empty_counts.items():
    print(f"{col:40}: {cnt} / {len(df)}")


#%%

#pprint(list(df.columns))
#pprint(list(df["all_belief_cluster_r2_hard"]))
#pprint(list(df["all_belief_cluster_r2_soft"]))
#pprint(list(df["all_belief_cluster_r2_refined"]))
#pprint(df["all_belief_cluster_r2_hard"].iloc[:10].tolist())
#pprint(df["assigned_belief_cluster_r2_hard"].iloc[:10].tolist())

def dim_from_geo(geo):
    if geo == "circle":
        return 2
    elif "simplex" in geo:
        return int(geo.split("_")[1])
    elif "mess" in geo:
        return 2
    elif "tom_quantum" in geo:
        return 2
    else:
        return None

# 2. Compute per-row means (lists in cells, so use ast.literal_eval, fallback to np.nan for empty/malformed)
metrics = {}
for i, (t, colnames) in enumerate(col_dict.items()):
    metrics[t] = {}
    metrics[t]["all_r2_mean"] = []
    metrics[t]["all_other_r2_mean"] = []
    metrics[t]["all_other_r2_max"] = []
    metrics[t]["all_r2_diff"] = []
    metrics[t]["assigned_r2"] = []
    metrics[t]["other_r2_mean"] = []
    metrics[t]["other_r2_max"] = []
    metrics[t]["r2_diff"] = []
    metrics[t]["r2_max_diff"] = []
    metrics[t]["assigned_geo_dim"] = []
    metrics[t]["inferred_geo_dim"] = []
    metrics[t]["geo_dim_diff"] = []
    mismatch_errors = 0
    for j, x in enumerate(df[colnames].to_records(index=False)):
        # print(f"{type(x)=}")
        # print(f"{x[0]=}")
        # print(f"{x[1]=}")
        # print(f"{x[2]=}")
        # print(f"{x[3]=}")
        n_good_rows = 0
        all_r2_mean = np.nanmean(list(x[0].values())) if x and len(x[0]) > 0 else np.nan
        if all_r2_mean > 0.61:
            print("\nMean R² is greater than 0.61")
            pprint(x[0])
            pprint(x[2])
            #for xx in x:
            #    pprint(xx)
        all_other_r2_mean = 0
        for comp, index in x[2].items():
            # print(f"{comp=}")
            # print(f"{index=}")
            if comp == "noise":
                continue
            cluster_r2_map = x[1].get(str(index), {})
            other_vals = [v for k, v in cluster_r2_map.items() if k != comp]
            if other_vals:
                other_r2_mean = float(np.mean(other_vals))
                other_r2_max = float(np.max(other_vals))
            else:
                other_r2_mean = np.nan
                other_r2_max = np.nan
            all_other_r2_mean += other_r2_mean
            agd = dim_from_geo(comp)
            try:
                igd = dim_from_geo(x[3][str(index)])
            except Exception as e:
                tb_str = traceback.format_exc()
                #print(tb_str)
                if "igd = dim_from_geo(x[3][str(index)])" in tb_str:
                    # Check for KeyError: 'some_int_as_str'
                    import re
                    m = re.search(r"KeyError: '(\d+)'", tb_str)
                    if m:
                        mismatch_errors += 1
                continue
            assigned_r2_val = x[0].get(str(index), np.nan)
            metrics[t]["other_r2_mean"].append(other_r2_mean)
            metrics[t]["other_r2_max"].append(other_r2_max)
            metrics[t]["assigned_r2"].append(assigned_r2_val)
            metrics[t]["r2_diff"].append(assigned_r2_val - other_r2_mean if not np.isnan(other_r2_mean) else np.nan)
            metrics[t]["r2_max_diff"].append(assigned_r2_val - other_r2_max if not np.isnan(other_r2_max) else np.nan)
            metrics[t]["assigned_geo_dim"].append(agd)
            metrics[t]["inferred_geo_dim"].append(igd)
            metrics[t]["geo_dim_diff"].append(agd - igd)
            n_good_rows += 1
        metrics[t]["all_r2_mean"].extend([all_r2_mean] * n_good_rows)
        metrics[t]["all_other_r2_mean"].extend([all_other_r2_mean / len(x[2])] * n_good_rows)
        metrics[t]["all_r2_diff"].extend([(all_r2_mean - all_other_r2_mean / len(x[2]))] * n_good_rows)
            # break
    metrics[t]["mismatch_errors"] = mismatch_errors

#%%

def find_top_row(df, assignment_type="hard"):
    """Return the first row (and its metadata) with the highest mean cluster R²."""
    col = f"assigned_belief_cluster_r2_{assignment_type}"
    top_idx = None
    top_score = -np.inf

    for idx, row in df.iterrows():
        cluster_dict = row.get(col)
        if not isinstance(cluster_dict, dict) or not cluster_dict:
            continue  # skip rows without valid data

        cluster_mean_r2 = np.mean(list(cluster_dict.values()))
        if not cluster_mean_r2:
            continue

        if cluster_mean_r2 > top_score:
            top_score = cluster_mean_r2
            top_idx = idx

    if top_idx is None:
        return None, None

    row = df.loc[top_idx]
    meta = {
        "index": top_idx,
        "layer": row.get("layer"),
        "assignment_type": assignment_type,
        "mean_cluster_r2": top_score,
    }
    return meta, row

# Usage example:
meta, best_row = find_top_row(df, assignment_type="hard")
if best_row is not None:
    print(meta)
    with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
        display(best_row)
#%%
print(meta)
pprint(df.loc[meta["index"]].to_dict())

#%%

pprint(best_row.to_dict())


#%%
for t in metrics:
    print(f"\nMetrics for '{t}':")
    for k, v in metrics[t].items():
        if isinstance(v, list):
            print(f"  {k:25}: len={len(v)}")
    print(f"  mismatch_errors: {metrics[t]['mismatch_errors']}")

#%%

for t in metrics:
    if "other_r2_mean" not in metrics[t] or "geo_dim_diff" not in metrics[t]:
        continue
    other_r2_mean = metrics[t]["other_r2_mean"]
    geo_dim_diff = metrics[t]["geo_dim_diff"]

    # Group r2_diff values by the value of geo_dim_diff
    from collections import defaultdict

    grouped = defaultdict(list)
    for gdd, or2d in zip(geo_dim_diff, other_r2_mean):
        grouped[gdd].append(or2d)

    # Sort groups by geo_dim_diff value
    sorted_keys = sorted(grouped.keys())
    data = [grouped[k] for k in sorted_keys]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=[str(k) for k in sorted_keys], showfliers=False)
    plt.xlabel("geo_dim_diff (assigned_geo_dim - inferred_geo_dim)")
    plt.ylabel("other_r2_mean")
    plt.title(f"Boxplot of other_r2_mean grouped by geo_dim_diff for '{t}'")
    plt.tight_layout()
    plt.show()


#%%

for t in metrics:
    if "other_r2_max" not in metrics[t] or "geo_dim_diff" not in metrics[t]:
        continue
    other_r2_max = metrics[t]["other_r2_max"]
    geo_dim_diff = metrics[t]["geo_dim_diff"]

    # Group r2_diff values by the value of geo_dim_diff
    from collections import defaultdict

    grouped = defaultdict(list)
    for gdd, or2d in zip(geo_dim_diff, other_r2_max):
        grouped[gdd].append(or2d)

    # Sort groups by geo_dim_diff value
    sorted_keys = sorted(grouped.keys())
    data = [grouped[k] for k in sorted_keys]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=[str(k) for k in sorted_keys], showfliers=False)
    plt.xlabel("geo_dim_diff (assigned_geo_dim - inferred_geo_dim)")
    plt.ylabel("other_r2_max")
    plt.title(f"Boxplot of other_r2_max grouped by geo_dim_diff for '{t}'")
    plt.tight_layout()
    plt.show()



#%%

for t in metrics:
    if "r2_max_diff" not in metrics[t] or "geo_dim_diff" not in metrics[t]:
        continue
    r2_max_diff = metrics[t]["r2_max_diff"]
    geo_dim_diff = metrics[t]["geo_dim_diff"]

    # Group r2_diff values by the value of geo_dim_diff
    from collections import defaultdict

    grouped = defaultdict(list)
    for gdd, or2d in zip(geo_dim_diff, r2_max_diff):
        grouped[gdd].append(or2d)

    # Sort groups by geo_dim_diff value
    sorted_keys = sorted(grouped.keys())
    data = [grouped[k] for k in sorted_keys]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=[str(k) for k in sorted_keys], showfliers=False)
    plt.xlabel("geo_dim_diff (assigned_geo_dim - inferred_geo_dim)")
    plt.ylabel("r2_max_diff")
    plt.title(f"Boxplot of r2_max_diff grouped by geo_dim_diff for '{t}'")
    plt.tight_layout()
    plt.show()


#%%

# Pairwise scatter plots: assigned_r2 vs [other_r2_mean, other_r2_max, r2_diff, r2_max_diff], colored by geo_dim_diff
scatter_metrics = [
    ("other_r2_mean", "Assigned R² vs Other R² Mean"),
    ("other_r2_max", "Assigned R² vs Other R² Max"),
    ("r2_diff", "Assigned R² vs R² Diff"),
    ("r2_max_diff", "Assigned R² vs R² Max Diff")
]

for t in metrics:
    assigned_r2 = metrics[t].get("assigned_r2")
    geo_dim_diff = np.array(metrics[t].get("geo_dim_diff"))

    if assigned_r2 is None or geo_dim_diff is None:
        continue

    for metric_key, plot_title in scatter_metrics:
        yvals = metrics[t].get(metric_key)
        if yvals is None:
            continue
        # Convert to arrays
        assigned = np.array(assigned_r2)
        y = np.array(yvals)
        gdd = np.array(geo_dim_diff)
        if len(assigned) != len(y) or len(assigned) != len(gdd):
            continue  # skip mismatched data

        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(assigned, y, c=gdd, cmap="viridis", alpha=0.7, s=32)
        plt.xlabel("assigned_r2")
        plt.ylabel(metric_key)
        cbar = plt.colorbar(scatter)
        cbar.set_label("geo_dim_diff")
        plt.title(f"{plot_title} ({t})")
        plt.tight_layout()
        plt.show()


#%%

# Make scatterplots: (other_r2_mean, r2_diff), (other_r2_mean, r2_max_diff),
#                    (other_r2_max, r2_diff), (other_r2_max, r2_max_diff)
cross_axis_pairs = [
    ("other_r2_mean", "r2_diff", "Other R² Mean", "R² Diff"),
    ("other_r2_mean", "r2_max_diff", "Other R² Mean", "R² Max Diff"),
    ("other_r2_max", "r2_diff", "Other R² Max", "R² Diff"),
    ("other_r2_max", "r2_max_diff", "Other R² Max", "R² Max Diff")
]
for t in metrics:
    geo_dim_diff = np.array(metrics[t].get("geo_dim_diff"))
    if geo_dim_diff is None:
        continue

    for xkey, ykey, xlabel, ylabel in cross_axis_pairs:
        xvals = metrics[t].get(xkey)
        yvals = metrics[t].get(ykey)
        gdd = metrics[t].get("geo_dim_diff")
        if xvals is None or yvals is None or gdd is None:
            continue
        x = np.array(xvals)
        y = np.array(yvals)
        c = np.array(gdd)
        # Skip if lengths don't match
        if len(x) != len(y) or len(x) != len(c):
            continue

        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(x, y, c=c, cmap="viridis", alpha=0.7, s=32)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cbar = plt.colorbar(scatter)
        cbar.set_label("geo_dim_diff")
        plt.title(f"{xlabel} vs {ylabel} ({t})")
        plt.tight_layout()
        plt.show()


#%%



# Find the average geo_dim_diff when r2_max_diff is negative versus positive

for t in metrics:
    geo_dim_diff = np.array(metrics[t].get("geo_dim_diff"))
    r2_max_diff = np.array(metrics[t].get("r2_max_diff"))
    if geo_dim_diff is None or r2_max_diff is None:
        continue

    # Make sure arrays have the same length
    if len(geo_dim_diff) != len(r2_max_diff):
        print(f"Skipping {t} due to length mismatch.")
        continue

    neg_mask = r2_max_diff < 0
    pos_mask = r2_max_diff >= 0

    avg_gdd_neg = np.nanmean(geo_dim_diff[neg_mask]) if np.any(neg_mask) else np.nan
    avg_gdd_pos = np.nanmean(geo_dim_diff[pos_mask]) if np.any(pos_mask) else np.nan

    print(f"\nFor '{t}':")
    print(f"  Average geo_dim_diff where r2_max_diff < 0: {avg_gdd_neg}")
    print(f"  Average geo_dim_diff where r2_max_diff >= 0: {avg_gdd_pos}")



#%%

import numpy as np

# Summarize all_r2_mean statistics separately for each t (task/key in metrics)
for t in metrics:
    arr = metrics[t].get("all_r2_mean")
    if arr is None:
        print(f"Skipping '{t}' (no all_r2_mean).")
        continue
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        print(f"Skipping '{t}' (all all_r2_mean NaN).")
        continue

    print(f"\nSummary of all_r2_mean for '{t}':")
    print(f"  min: {np.min(arr)}")
    print(f"  max: {np.max(arr)}")
    # Deciles: 0%, 10%, ..., 100%
    deciles = np.percentile(arr, np.arange(0, 110, 10))
    print(f"  deciles (0%, 10%, ..., 100%):")
    for i, v in enumerate(deciles):
        print(f"    {i*10}%: {v}")
    for p in [1, 5, 95, 99]:
        val = np.percentile(arr, p)
        print(f"  {p}th percentile: {val}")


#%%
######## Non Ground Truth Metrics ############################
##############################################################

KTH_LOWEST_PRINCIPAL_ANGLE = 4  # Default rank when averaging principal angles


def _ensure_dict(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        parsed = safe_eval(val)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_float(val):
    if val is None:
        return np.nan
    if isinstance(val, (int, float, np.number)):
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return np.nan
        return float(val)
    try:
        num = float(val)
    except (TypeError, ValueError):
        return np.nan
    return num if not np.isnan(num) else np.nan


def _filter_numeric(values):
    nums = []
    for value in values:
        num = _as_float(value)
        if not np.isnan(num):
            nums.append(num)
    return nums


def _mean_from_dict(mapping):
    if not isinstance(mapping, dict) or not mapping:
        return np.nan
    values = _filter_numeric(mapping.values())
    return float(np.mean(values)) if values else np.nan


def _assignment_r2_stats(payload):
    stats = {"r2_min": np.nan, "r2_mean": np.nan, "r2_max": np.nan}
    if not isinstance(payload, dict):
        return stats
    scores_dict = payload.get("assignment_scores")
    if not isinstance(scores_dict, dict):
        return stats
    scores = _filter_numeric(scores_dict.values())
    if not scores:
        return stats
    stats["r2_min"] = float(np.min(scores))
    stats["r2_mean"] = float(np.mean(scores))
    stats["r2_max"] = float(np.max(scores))
    return stats


def _principal_angle_kth_mean(angle_dict, kth):
    if not isinstance(angle_dict, dict):
        return np.nan
    idx = max(int(kth) - 1, 0)
    kth_vals = []
    for values in angle_dict.values():
        if not isinstance(values, (list, tuple)):
            continue
        numeric_vals = sorted(_filter_numeric(values))
        if len(numeric_vals) > idx:
            kth_vals.append(numeric_vals[idx])
    return float(np.mean(kth_vals)) if kth_vals else np.nan


def _ordinal(value):
    value = int(value)
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _empty_dict_series(length, index=None):
    return pd.Series([{} for _ in range(length)], index=index)


num_rows = len(df)

# Normalize assignment payloads so we always have the same structure
hard_assignments_series = (
    df["hard_assignments"]
    if "hard_assignments" in df.columns
    else _empty_dict_series(num_rows, df.index)
).apply(_ensure_dict)

hard_assignment_scores_series = (
    df["hard_assignment_scores"]
    if "hard_assignment_scores" in df.columns
    else _empty_dict_series(num_rows, df.index)
).apply(_ensure_dict)

if "hard_assignment_scores" in df.columns:
    df["hard_assignment_scores"] = hard_assignment_scores_series

if "component_assignments_hard" in df.columns:
    component_assign_series = df["component_assignments_hard"].apply(_ensure_dict)
else:
    component_assign_series = pd.Series(
        [
            {"assignments": assigns, "assignment_scores": scores}
            for assigns, scores in zip(hard_assignments_series, hard_assignment_scores_series)
        ],
        index=df.index,
    )
df["component_assignments_hard"] = component_assign_series

# Prepare dict-based metric columns
if "coherence_metrics_hard" in df.columns:
    coherence_series = df["coherence_metrics_hard"].apply(_ensure_dict)
else:
    print("Warning: 'coherence_metrics_hard' column missing; coherence plots may be empty.")
    coherence_series = _empty_dict_series(num_rows, df.index)
principal_angles_series = (
    df["principal_angles_deg"].apply(_ensure_dict)
    if "principal_angles_deg" in df.columns
    else _empty_dict_series(num_rows, df.index)
)
min_principal_series = (
    df["min_principal_angles_deg"].apply(_ensure_dict)
    if "min_principal_angles_deg" in df.columns
    else _empty_dict_series(num_rows, df.index)
)
within_energy_series = (
    df["within_projection_energy"].apply(_ensure_dict)
    if "within_projection_energy" in df.columns
    else _empty_dict_series(num_rows, df.index)
)

# Assemble plotting dataframe
assignment_stats = component_assign_series.apply(
    lambda payload: pd.Series(_assignment_r2_stats(payload))
)
plot_df = assignment_stats.copy()

layer_series = (
    df["layer"].fillna("unknown")
    if "layer" in df.columns
    else pd.Series(["unknown"] * num_rows, index=df.index)
)
plot_df["layer"] = layer_series

plot_df["coherence_within_mean"] = coherence_series.apply(
    lambda c: _as_float(c.get("within_cluster_correlation_mean")) if isinstance(c, dict) else np.nan
)
plot_df["coherence_between_mean"] = coherence_series.apply(
    lambda c: _as_float(c.get("between_cluster_correlation_mean")) if isinstance(c, dict) else np.nan
)
plot_df["coherence_diff"] = (
    plot_df["coherence_between_mean"] - plot_df["coherence_within_mean"]
)

plot_df["overall_min_principal_angle_deg"] = (
    pd.to_numeric(df["overall_min_principal_angle_deg"], errors="coerce")
    if "overall_min_principal_angle_deg" in df.columns
    else np.nan
)
plot_df["principal_angle_kth_mean"] = principal_angles_series.apply(
    lambda angles: _principal_angle_kth_mean(angles, KTH_LOWEST_PRINCIPAL_ANGLE)
)
plot_df["mean_min_principal_angle"] = min_principal_series.apply(_mean_from_dict)

plot_df["within_projection_energy_mean"] = within_energy_series.apply(_mean_from_dict)
plot_df["between_projection_energy"] = (
    pd.to_numeric(df["between_projection_energy"], errors="coerce")
    if "between_projection_energy" in df.columns
    else np.nan
)
plot_df["energy_contrast_ratio"] = (
    pd.to_numeric(df["energy_contrast_ratio"], errors="coerce")
    if "energy_contrast_ratio" in df.columns
    else np.nan
)

GROUND_TRUTH_AXES = [
    ("r2_min", "Min assigned R²"),
    ("r2_mean", "Mean assigned R²"),
    ("r2_max", "Max assigned R²"),
]

principal_angle_label = (
    f"Mean {_ordinal(KTH_LOWEST_PRINCIPAL_ANGLE)} lowest principal angle (deg)"
)

PLOT_GROUPS = [
    {
        "title": "Coherence metrics vs assigned R² (hard)",
        "metrics": [
            ("coherence_within_mean", "Within-cluster corr (mean)"),
            ("coherence_between_mean", "Between-cluster corr (mean)"),
            ("coherence_diff", "Between - within corr"),
        ],
    },
    {
        "title": "Principal angles vs assigned R²",
        "metrics": [
            ("overall_min_principal_angle_deg", "Overall min principal angle (deg)"),
            ("principal_angle_kth_mean", principal_angle_label),
            ("mean_min_principal_angle", "Mean of min principal angles (deg)"),
        ],
    },
    {
        "title": "Projection energy metrics vs assigned R²",
        "metrics": [
            ("within_projection_energy_mean", "Mean within projection energy"),
            ("between_projection_energy", "Between projection energy"),
            ("energy_contrast_ratio", "Energy contrast ratio"),
        ],
    },
]

layer_values = plot_df["layer"].fillna("unknown")
layer_order = list(dict.fromkeys(layer_values.tolist()))
if not layer_order:
    layer_order = ["unknown"]
cmap = plt.get_cmap("tab20")
layer_colors = {layer: cmap(i % cmap.N) for i, layer in enumerate(layer_order)}
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color=color,
        label=layer,
        markersize=6,
        markeredgecolor="k",
        markeredgewidth=0.25,
    )
    for layer, color in layer_colors.items()
]

has_ground_truth = any(
    plot_df[col].notna().any() for col, _ in GROUND_TRUTH_AXES if col in plot_df.columns
)

if not has_ground_truth:
    print("Skipping non ground truth metric plots: no assignment score data available.")
else:
    for group in PLOT_GROUPS:
        metric_keys = [key for key, _ in group["metrics"]]
        has_metric_data = any(
            plot_df.get(key, pd.Series(dtype=float)).notna().any() for key in metric_keys
        )
        if not has_metric_data:
            print(f"Skipping '{group['title']}' plots (no metric data).")
            continue

        n_rows = len(metric_keys)
        n_cols = len(GROUND_TRUTH_AXES)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.8 * n_cols, 3.2 * n_rows),
            sharex="col",
            squeeze=False,
        )

        for row_idx, (metric_key, metric_label) in enumerate(group["metrics"]):
            y_vals = plot_df.get(metric_key, pd.Series(dtype=float))
            for col_idx, (x_key, x_label) in enumerate(GROUND_TRUTH_AXES):
                ax = axes[row_idx][col_idx]
                x_vals = plot_df.get(x_key, pd.Series(dtype=float))
                valid_mask = x_vals.notna() & y_vals.notna()

                plotted = False
                for layer_name in layer_order:
                    layer_mask = valid_mask & (layer_values == layer_name)
                    if layer_mask.any():
                        ax.scatter(
                            x_vals[layer_mask],
                            y_vals[layer_mask],
                            alpha=0.75,
                            s=36,
                            color=layer_colors[layer_name],
                            edgecolor="white",
                            linewidth=0.4,
                        )
                        plotted = True

                if not plotted:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="0.4",
                    )

                if row_idx == n_rows - 1:
                    ax.set_xlabel(x_label)
                if col_idx == 0:
                    ax.set_ylabel(metric_label)
                ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)

        fig.suptitle(group["title"], fontsize=14)
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                labels=[h.get_label() for h in legend_handles],
                title="Layer",
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
            )
            layout_rect = [0, 0, 0.88, 0.95]
        else:
            layout_rect = [0, 0, 1, 0.96]
        plt.tight_layout(rect=layout_rect)
        plt.show()



#%%
# Layer-wise regression: predict mean assigned R² using non-ground-truth metrics
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

PLOT_GROUPS = [
    {
        "title": "Coherence metrics vs assigned R² (hard)",
        "metrics": [
            ("coherence_within_mean", "Within-cluster corr (mean)"),
            ("coherence_between_mean", "Between-cluster corr (mean)"),
            ("coherence_diff", "Between - within corr"),
        ],
    },
    {
        "title": "Principal angles vs assigned R²",
        "metrics": [
            ("overall_min_principal_angle_deg", "Overall min principal angle (deg)"),
            ("principal_angle_kth_mean", principal_angle_label),
            ("mean_min_principal_angle", "Mean of min principal angles (deg)"),
        ],
    },
    {
        "title": "Projection energy metrics vs assigned R²",
        "metrics": [
            ("within_projection_energy_mean", "Mean within projection energy"),
            ("between_projection_energy", "Between projection energy"),
            ("energy_contrast_ratio", "Energy contrast ratio"),
        ],
    },
]

REGRESSION_TARGET_COL = "r2_mean"
REGRESSION_FEATURES = sorted(
    {metric_key for group in PLOT_GROUPS for metric_key, _ in group["metrics"] \
        if metric_key not in ['coherence_within_mean', 'coherence_between_mean', 'mean_min_principal_angle', 'within_projection_energy_mean']}
)
#REGRESSION_FEATURES = ["coherence_diff", "principal_angle_kth_mean", "overall_min_principal_angle_deg", "energy_contrast_ratio"]
REGRESSION_FEATURES = ["coherence_diff", "principal_angle_kth_mean", "overall_min_principal_angle_deg"]
REGRESSION_MIN_FEATURE_COVERAGE = 0.3
REGRESSION_MIN_ROWS = 5
# Collect ground-truth R² stats for plotting/analysis
r2_stats_df = plot_df[["r2_min", "r2_mean", "r2_max", "layer"]].copy()
# Now r2_stats_df contains the min/mean/max assigned R² and layer for each row for manual use.


def _run_layer_regression(layer_name, layer_df):
    """Fit LinearRegression (sklearn) and report coefficients with p-values."""
    available_features = []
    coverage_stats = {}
    for feat in REGRESSION_FEATURES:
        if feat not in layer_df.columns:
            continue
        coverage = layer_df[feat].notna().mean()
        coverage_stats[feat] = coverage
        if coverage >= REGRESSION_MIN_FEATURE_COVERAGE:
            available_features.append(feat)

    if not available_features:
        print(
            f"Skipping regression for {layer_name}: no feature met "
            f"coverage threshold {REGRESSION_MIN_FEATURE_COVERAGE:.0%} "
            f"(coverage={coverage_stats})."
        )
        return

    target_series = layer_df[REGRESSION_TARGET_COL]
    if target_series.notna().sum() < REGRESSION_MIN_ROWS:
        print(f"Skipping regression for {layer_name}: target '{REGRESSION_TARGET_COL}' has insufficient non-null rows.")
        return

    feature_df = layer_df[available_features].astype(float)
    # Impute remaining NaNs with column means to avoid dropping all rows
    feature_means = feature_df.mean(skipna=True)
    feature_df = feature_df.fillna(feature_means)

    mask = target_series.notna()
    y = target_series[mask].to_numpy(dtype=float)
    X = feature_df[mask].to_numpy(dtype=float)

    min_samples = len(available_features) + 1  # intercept + features
    if len(y) < max(min_samples, REGRESSION_MIN_ROWS):
        print(
            f"Skipping regression for {layer_name}: insufficient usable samples "
            f"({len(y)}) after aligning target and features."
        )
        return

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    coef = np.concatenate(([model.intercept_], model.coef_))
    ones = np.ones((len(y), 1), dtype=float)
    design_matrix = np.hstack([ones, X])
    resid = y - y_pred

    n_samples = len(y)
    n_params = design_matrix.shape[1]
    dof = n_samples - n_params
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - rss / tss if tss > 0 else np.nan

    p_values = [np.nan] * len(coef)
    if dof > 0 and scipy_stats is not None:
        sigma_sq = rss / dof
        xtx = design_matrix.T @ design_matrix
        xtx_inv = np.linalg.pinv(xtx)
        cov = sigma_sq * xtx_inv
        std_err = np.sqrt(np.clip(np.diag(cov), a_min=0, a_max=None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.divide(coef, std_err, out=np.full_like(coef, np.nan), where=std_err > 0)
        p_values = 2 * scipy_stats.t.sf(np.abs(t_stats), df=dof)
    elif scipy_stats is None:
        print("  (SciPy not available; coefficient p-values omitted.)")

    header = ["intercept"] + available_features
    print(f"\nLayer {layer_name} regression results using features {available_features} "
          f"(n={n_samples}, R²={r_squared:.4f}):")
    print(f"{'Coefficient':<35}{'Estimate':>12}{'p-value':>14}")
    for name, beta, p_val in zip(header, coef, p_values):
        p_str = f"{p_val:.3e}" if np.isfinite(p_val) else "   n/a"
        print(f"{name:<35}{beta:>12.5f}{p_str:>14}")

    plt.figure(figsize=(5.5, 5))
    plt.scatter(
        y,
        y_pred,
        alpha=0.8,
        color=layer_colors.get(layer_name, "#1f77b4"),
        edgecolor="white",
        linewidth=0.4,
    )
    line_min = float(np.nanmin([y.min(), y_pred.min()]))
    line_max = float(np.nanmax([y.max(), y_pred.max()]))
    plt.plot([line_min, line_max], [line_min, line_max], color="k", linestyle="--", linewidth=1)
    plt.xlabel("Actual mean assigned R²")
    plt.ylabel("Predicted mean assigned R²")
    plt.title(f"{layer_name}: actual vs predicted assigned R²")
    plt.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    plt.tight_layout()
    plt.show()
    
    return model, y_pred


reg_models = []
y_preds = []
if LinearRegression is None:
    print("scikit-learn is not available; skipping regression plots.")
else:
    missing_features = [feat for feat in REGRESSION_FEATURES if feat not in plot_df.columns]
    if missing_features:
        print(f"Cannot run regressions; missing features in plot_df: {missing_features}")
    elif REGRESSION_TARGET_COL not in plot_df.columns:
        print(f"Cannot run regressions; missing target column '{REGRESSION_TARGET_COL}'.")
    else:
        for layer_name in sorted(plot_df["layer"].dropna().unique()):
            reg_model, y_pred = _run_layer_regression(layer_name, plot_df[plot_df["layer"] == layer_name])
            reg_models.append(reg_model)
            y_preds.append(y_pred)


#%%
# Create a percentile version of plot_df where all columns are converted to percentiles
def df_to_percentiles(df):
    percentile_df = df.copy()
    for col in percentile_df.columns:
        if np.issubdtype(percentile_df[col].dtype, np.number):
            col_values = percentile_df[col].values
            # Handle nan by ignoring them in the ranking
            non_nan = ~np.isnan(col_values)
            ranks = np.full_like(col_values, np.nan, dtype=np.float64)
            if non_nan.sum() > 0:
                # Compute ranks for non-nan values
                non_nan_vals = col_values[non_nan]
                order = np.argsort(non_nan_vals)
                ranks_raw = order.argsort()
                # Percentile is rank / (N-1)
                n = len(non_nan_vals)
                if n == 1:
                    percentiles = np.zeros(1)
                else:
                    percentiles = ranks_raw / (n - 1)
                ranks[non_nan] = percentiles
            percentile_df[col] = ranks
    return percentile_df

plot_df_percentiles = df_to_percentiles(plot_df)


#%%
############### Older code below ###############################
################################################################

cols_to_drop = [
    "layer_0_component_assignment_refined_assignments",
    "layer_0_component_assignment_refined_noise_clusters",
    "layer_0_component_assignment_soft_assignments",
    "layer_0_component_assignment_soft_noise_clusters",
    "layer_1_component_assignment_refined_assignments",
    "layer_1_component_assignment_refined_noise_clusters",
    "layer_1_component_assignment_soft_assignments",
    "layer_1_component_assignment_soft_noise_clusters",
    "layer_2_component_assignment_refined_assignments",
    "layer_2_component_assignment_refined_noise_clusters",
    "layer_2_component_assignment_soft_assignments",
    "layer_2_component_assignment_soft_noise_clusters",
    "embeddings_component_assignment_refined_assignments",
    "embeddings_component_assignment_refined_noise_clusters",
    "embeddings_component_assignment_soft_assignments",
    "embeddings_component_assignment_soft_noise_clusters",
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

cols = list(df.columns)
for i in range(len(cols)):
    print(cols[i])
#for i in range(0, len(cols), 5):
#    print(" | ".join(cols[i:i+5]))


#%%
# Display all rows for the selected columns and show full cell content (no ... elision)
with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
    display(df.loc[0:3, ("run_name", "layer","refined_assignments", "refined_assignment_scores")])


#%%
import matplotlib.pyplot as plt

geo_types = ["hard", "soft", "refined"]
layer_names = ["layer_0", "layer_1", "layer_2"]
n_bins = 6  # 0 through 5 inclusive
bar_labels = [str(i) for i in range(n_bins)]
bar_x = list(range(n_bins))

for geo in geo_types:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, layer in enumerate(layer_names):
        ax = axes[idx]
        layer_df = df[df["layer"] == layer]
        col = f"n_geo_matches_{geo}"
        # Default to all zeros if column is missing
        if col in layer_df.columns:
            hist = layer_df[col].value_counts().reindex(range(n_bins), fill_value=0).sort_index()
        else:
            hist = pd.Series([0]*n_bins, index=range(n_bins))
        ax.bar(bar_x, hist.values, color="#6baed6")
        ax.set_title(f"{layer}")
        ax.set_xticks(bar_x)
        ax.set_xticklabels(bar_labels)
        ax.set_xlabel("Number of Geometry Matches")
        if idx == 0:
            ax.set_ylabel("Count")
        for i, v in enumerate(hist.values):
            ax.text(i, v + 0.05 * max(hist.values, default=1), str(v), ha='center', va='bottom', fontsize=11)
    fig.suptitle(f'Distribution of Geometry Matches ({geo} assignment)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


#%%
rows = df[(df["layer"] == "layer_2") & (df["n_geo_matches_refined"] == 2)]
with pd.option_context('display.max_colwidth', None):
    display(rows)


#%%
# Find runs where n_geo_matches_soft == 2 for layer_1 or layer_2
# Display sweep parameters for these runs

# Define the sweep parameter columns
sweep_params = [
    "sae_type",  # topk or vanilla
    "force_k",  # for topk
    "force_lambda",  # for vanilla
    "subspace_variance_threshold",  # for auto cluster detection
    "subspace_gap_threshold",  # for auto cluster detection
    "subspace_n_clusters",  # for manual cluster setting
    "cosine_dedup_threshold",  # deduplication
    "latent_activity_threshold",  # latent filtering
    "sim_metric",  # cosine or euclidean
    "geo_per_point_threshold",  # per-point filtering
]

# Filter for the condition
filtered_df = df[
    ((df["layer"] == "layer_1") | (df["layer"] == "layer_2")) &
    ((df["n_geo_matches_soft"] == 2) | (df["n_geo_matches_soft"] == 3))
]

print(f"Found {len(filtered_df)} runs where n_geo_matches_soft == 2 for layer_1 or layer_2")
print(f"Number of unique run_ids: {filtered_df['run_id'].nunique()}")
print("\n" + "="*80)

# Display sweep parameters for these runs
if len(filtered_df) > 0:
    # Select columns to display
    display_cols = ["run_name", "layer", "n_geo_matches_soft"] + sweep_params
    display_cols = [col for col in display_cols if col in filtered_df.columns]

    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        display(filtered_df[display_cols].sort_values(["run_name", "layer"]))

    # Show summary statistics for sweep parameters
    print("\n" + "="*80)
    print("Summary of sweep parameters for matching runs:")
    print("="*80)
    for param in sweep_params:
        if param in filtered_df.columns:
            unique_vals = filtered_df[param].dropna().unique()
            if len(unique_vals) > 0:
                print(f"\n{param}:")
                for val in sorted(unique_vals, key=lambda x: (x is None, x)):
                    count = len(filtered_df[filtered_df[param] == val])
                    print(f"  {val}: {count} runs")
else:
    print("No matching runs found.")


#%%
# Find runs where n_geo_matches_refined == 2 for layer_1 or layer_2
# Display sweep parameters for these runs

# Define the sweep parameter columns
sweep_params = [
    "sae_type",  # topk or vanilla
    "force_k",  # for topk
    "force_lambda",  # for vanilla
    "subspace_variance_threshold",  # for auto cluster detection
    "subspace_gap_threshold",  # for auto cluster detection
    "subspace_n_clusters",  # for manual cluster setting
    "cosine_dedup_threshold",  # deduplication
    "latent_activity_threshold",  # latent filtering
    "sim_metric",  # cosine or euclidean
    "geo_per_point_threshold",  # per-point filtering
]

# Filter for the condition
filtered_df = df[
    ((df["layer"] == "layer_1") | (df["layer"] == "layer_2")) &
    (df["n_geo_matches_refined"] == 2)
]

print(f"Found {len(filtered_df)} runs where n_geo_matches_refined == 2 for layer_1 or layer_2")
print(f"Number of unique run_ids: {filtered_df['run_id'].nunique()}")
print("\n" + "="*80)

# Display sweep parameters for these runs
if len(filtered_df) > 0:
    # Select columns to display
    display_cols = ["run_name", "layer", "n_geo_matches_refined"] + sweep_params
    display_cols = [col for col in display_cols if col in filtered_df.columns]

    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        display(filtered_df[display_cols].sort_values(["run_name", "layer"]))

    # Show summary statistics for sweep parameters
    print("\n" + "="*80)
    print("Summary of sweep parameters for matching runs:")
    print("="*80)
    for param in sweep_params:
        if param in filtered_df.columns:
            unique_vals = filtered_df[param].dropna().unique()
            if len(unique_vals) > 0:
                print(f"\n{param}:")
                for val in sorted(unique_vals, key=lambda x: (x is None, x)):
                    count = len(filtered_df[filtered_df[param] == val])
                    print(f"  {val}: {count} runs")
else:
    print("No matching runs found.")


#%%
# Component-level geometry accuracy analysis

EXPECTED_GEOMETRIES = {
    "tom_quantum": "circle",
    "tom_quantum_1": "circle",
    "mess3": "simplex_2",
    "mess3_1": "simplex_2",
    "mess3_2": "simplex_2",
}

ASSIGNMENT_COLUMNS = {
    "hard": "hard_assignments",
    "soft": "soft_assignments",
    "refined": "refined_assignments",
}

def _safe_parse(value):
    if isinstance(value, str) and value.strip():
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
    return {} if value is None or (isinstance(value, float) and np.isnan(value)) else value


component_records = []

for _, row in df.iterrows():
    best_geoms = _safe_parse(row.get("best_geometries"))
    if not isinstance(best_geoms, dict):
        best_geoms = {}

    for assign_type, col in ASSIGNMENT_COLUMNS.items():
        if col not in df.columns:
            continue
        assignments = _safe_parse(row.get(col))
        if not isinstance(assignments, dict):
            continue

        for component, expected_geom in EXPECTED_GEOMETRIES.items():
            cluster = assignments.get(component)
            if cluster is None or isinstance(cluster, list):
                continue

            predicted_geom = best_geoms.get(str(cluster))
            is_correct = predicted_geom == expected_geom

            component_records.append(
                {
                    "run_id": row.get("run_id"),
                    "run_name": row.get("run_name"),
                    "layer": row.get("layer"),
                    "sae_type": row.get("sae_type"),
                    "sae_param": row.get("force_k") if row.get("sae_type") == "top_k" else row.get("force_lambda"),
                    "assignment_type": assign_type,
                    "component": component,
                    "predicted_geometry": predicted_geom,
                    "expected_geometry": expected_geom,
                    "is_correct": bool(is_correct),
                }
            )


component_df = pd.DataFrame(component_records)

if component_df.empty:
    print("No component-level records were produced; skipping geometry accuracy plots.")
else:
    components_order = list(EXPECTED_GEOMETRIES.keys())
    component_df["component"] = pd.Categorical(component_df["component"], categories=components_order, ordered=True)

    # Overall accuracy per component
    overall_acc = (
        component_df.groupby(["assignment_type", "component"]) ["is_correct"]
        .mean()
        .reset_index()
    )

    for assign_type, subset in overall_acc.groupby("assignment_type"):
        values = subset.set_index("component")["is_correct"].reindex(components_order).fillna(0.0)
        plt.figure(figsize=(8, 4))
        bars = plt.bar(range(len(components_order)), values.values * 100.0, color="#74add1")
        plt.xticks(range(len(components_order)), components_order, rotation=30)
        plt.ylim(0, 100)
        plt.ylabel("Accuracy (%)")
        plt.title(f"Overall geometry accuracy per component ({assign_type})")
        for idx, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{values.values[idx]*100:.1f}%", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.show()

    # Accuracy by layer
    layers = sorted(component_df["layer"].dropna().unique())
    for assign_type in component_df["assignment_type"].unique():
        fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), sharey=True)
        if len(layers) == 1:
            axes = [axes]
        for ax, layer in zip(axes, layers):
            layer_subset = component_df[(component_df["assignment_type"] == assign_type) & (component_df["layer"] == layer)]
            acc = (
                layer_subset.groupby("component")["is_correct"].mean()
                .reindex(components_order)
                .fillna(0.0)
            )
            bars = ax.bar(range(len(components_order)), acc.values * 100.0, color="#fdae61")
            ax.set_xticks(range(len(components_order)))
            ax.set_xticklabels(components_order, rotation=30)
            ax.set_title(layer)
            if ax is axes[0]:
                ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            for idx, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc.values[idx]*100:.1f}%", ha="center", va="bottom", fontsize=9)
        fig.suptitle(f"Geometry accuracy by layer ({assign_type})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

    # Accuracy by SAE type
    sae_types = sorted(component_df["sae_type"].dropna().unique())
    for assign_type in component_df["assignment_type"].unique():
        fig, axes = plt.subplots(1, len(sae_types), figsize=(4 * len(sae_types), 4), sharey=True)
        if len(sae_types) == 1:
            axes = [axes]
        for ax, sae_type in zip(axes, sae_types):
            sae_subset = component_df[(component_df["assignment_type"] == assign_type) & (component_df["sae_type"] == sae_type)]
            acc = (
                sae_subset.groupby("component")["is_correct"].mean()
                .reindex(components_order)
                .fillna(0.0)
            )
            bars = ax.bar(range(len(components_order)), acc.values * 100.0, color="#abdda4")
            ax.set_xticks(range(len(components_order)))
            ax.set_xticklabels(components_order, rotation=30)
            ax.set_title(sae_type)
            if ax is axes[0]:
                ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            for idx, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc.values[idx]*100:.1f}%", ha="center", va="bottom", fontsize=9)
        fig.suptitle(f"Geometry accuracy by SAE type ({assign_type})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

    # Accuracy by layer and SAE type (grid)
    for assign_type in component_df["assignment_type"].unique():
        fig, axes = plt.subplots(len(layers), len(sae_types), figsize=(4 * len(sae_types), 3.5 * len(layers)), sharey=True)
        if len(layers) == 1 and len(sae_types) == 1:
            axes = np.array([[axes]])
        elif len(layers) == 1:
            axes = axes[np.newaxis, :]
        elif len(sae_types) == 1:
            axes = axes[:, np.newaxis]

        for i, layer in enumerate(layers):
            for j, sae_type in enumerate(sae_types):
                ax = axes[i, j]
                subset = component_df[
                    (component_df["assignment_type"] == assign_type)
                    & (component_df["layer"] == layer)
                    & (component_df["sae_type"] == sae_type)
                ]
                acc = (
                    subset.groupby("component")["is_correct"].mean()
                    .reindex(components_order)
                    .fillna(0.0)
                )
                bars = ax.bar(range(len(components_order)), acc.values * 100.0, color="#fee08b")
                ax.set_xticks(range(len(components_order)))
                ax.set_xticklabels(components_order, rotation=30)
                if i == 0:
                    ax.set_title(sae_type)
                if j == 0:
                    ax.set_ylabel(f"{layer}\nAccuracy (%)")
                ax.set_ylim(0, 100)
                for idx, bar in enumerate(bars):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc.values[idx]*100:.1f}%", ha="center", va="bottom", fontsize=8)
        fig.suptitle(f"Geometry accuracy by layer & SAE type ({assign_type})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Summary table
    summary = (
        component_df.groupby(["assignment_type", "component", "layer", "sae_type"])["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary.rename(columns={"mean": "accuracy", "count": "n_evaluated"}, inplace=True)
    with pd.option_context('display.max_rows', None):
        display(summary.head(50))

#%%
