from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class ClusterDescriptor:
    cluster_id: int
    label: str
    latent_indices: List[int]
    is_noise: bool
    component_names: Sequence[str]


def load_cluster_summary(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_cluster_descriptors(
    summary: Mapping[str, object],
    *,
    include_noise: bool = True,
) -> List[ClusterDescriptor]:
    clusters_raw: Mapping[str, Mapping[str, object]] = summary.get("clusters", {})  # type: ignore[assignment]
    component_block: Mapping[str, object] = summary.get("component_assignment_hard", {})  # type: ignore[assignment]
    component_to_cluster: Mapping[str, int] = component_block.get("assignments", {})  # type: ignore[assignment]
    noise_clusters: Iterable[int] = component_block.get("noise_clusters", [])  # type: ignore[assignment]

    cluster_to_components: Dict[int, List[str]] = {}
    for component_name, cluster_value in component_to_cluster.items():
        cluster_id = int(cluster_value)
        cluster_to_components.setdefault(cluster_id, []).append(str(component_name))

    descriptors: List[ClusterDescriptor] = []
    for cluster_key, entry in clusters_raw.items():
        cluster_id = int(cluster_key)
        latent_indices = [int(idx) for idx in entry.get("latent_indices", [])]  # type: ignore[arg-type]
        is_noise = cluster_id in set(int(val) for val in noise_clusters)
        if is_noise and not include_noise:
            continue
        component_names = cluster_to_components.get(cluster_id, [])
        if component_names:
            label = "_".join(component_names)
        elif is_noise:
            label = f"noise_{cluster_id}"
        else:
            label = f"cluster_{cluster_id}"
        descriptors.append(
            ClusterDescriptor(
                cluster_id=cluster_id,
                label=label,
                latent_indices=latent_indices,
                is_noise=is_noise,
                component_names=component_names,
            )
        )
    descriptors.sort(key=lambda desc: desc.cluster_id)
    return descriptors


def build_latent_assignment_list(
    summary: Mapping[str, object],
    *,
    dict_size: int,
    default_cluster: int = -1,
) -> List[int]:
    assignments = [int(default_cluster) for _ in range(dict_size)]
    latent_map: Mapping[str, int] = summary.get("latent_cluster_assignments", {})  # type: ignore[assignment]
    for key, cluster_value in latent_map.items():
        idx = int(key)
        if 0 <= idx < dict_size:
            assignments[idx] = int(cluster_value)
    return assignments
