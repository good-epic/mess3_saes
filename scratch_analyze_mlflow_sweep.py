#%%
import numpy as np
import pandas as pd
import ast
from pprint import pprint
import matplotlib.pyplot as plt
import traceback
from clustering import SimplexGeometry, CircleGeometry, HypersphereGeometry


df = pd.read_csv("mlflow_sweep_results_20251018_174112.csv")

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
        all_other_r2_mean = 0
        for comp, index in x[2].items():
            # print(f"{comp=}")
            # print(f"{index=}")
            if comp == "noise":
                continue
            other_r2_mean = np.mean([v for k,v in x[1][str(index)].items() if k != comp])
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
            metrics[t]["other_r2_mean"].append(other_r2_mean)
            metrics[t]["r2_diff"].append(x[0][str(index)] - other_r2_mean)
            metrics[t]["assigned_r2"].append(x[0][str(index)])
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
