from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class AAnetDescriptor:
    cluster_id: int
    label: str
    latent_indices: List[int]
    is_noise: bool
    component_names: Sequence[str]


def load_aanet_summary(path: str) -> List[AAnetDescriptor]:
    """Load AAnet descriptors from a JSON summary file."""
    with open(path, "r") as f:
        data = json.load(f)
    return parse_aanet_descriptors(data)


def parse_aanet_descriptors(data: Dict) -> List[AAnetDescriptor]:
    """Parse dictionary data into AAnetDescriptor objects."""
    descriptors = []
    # Handle different formats if needed, but assuming standard format for now
    # Format: {"cluster_0": {"label": "...", "latent_indices": [...]}, ...}
    # Or list format
    
    if isinstance(data, list):
        # List of dicts
        for item in data:
            descriptors.append(AAnetDescriptor(
                cluster_id=item.get("cluster_id", -1),
                label=item.get("label", str(item.get("cluster_id", -1))),
                latent_indices=item.get("latent_indices", []),
                component_names=item.get("component_names", []),
                is_noise=item.get("is_noise", False)
            ))
    elif isinstance(data, dict):
        # Dict mapping ID/label to content
        for key, val in data.items():
            # Try to infer ID from key if not in val
            cid = val.get("cluster_id")
            if cid is None and key.startswith("cluster_"):
                try:
                    cid = int(key.split("_")[1])
                except:
                    cid = -1
            
            descriptors.append(AAnetDescriptor(
                cluster_id=cid if cid is not None else -1,
                label=val.get("label", key),
                latent_indices=val.get("latent_indices", []),
                component_names=val.get("component_names", []),
                is_noise=val.get("is_noise", False)
            ))
            
    return sorted(descriptors, key=lambda x: x.cluster_id)


def build_latent_assignment_list(descriptors: List[AAnetDescriptor], total_latents: int) -> List[int]:
    """Build a list mapping each latent index to its AAnet group ID."""
    assignment = [-1] * total_latents
    for desc in descriptors:
        for idx in desc.latent_indices:
            if 0 <= idx < total_latents:
                assignment[idx] = desc.cluster_id
    return assignment
