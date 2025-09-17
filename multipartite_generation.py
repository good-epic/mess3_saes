"""Utilities for sampling from multipartite generative processes.

This module provides lightweight helpers that combine multiple simplexity
generative processes (e.g. mess3 and tom_quantum) into a single sampler that
emits product-space tokens while keeping track of the component belief states.
The intent is to enable SAE training on stacked processes without having to
re-implement a full joint HMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp

from simplexity.generative_processes.builder import (
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
)
from simplexity.generative_processes.generative_process import GenerativeProcess


PROCESS_BUILDERS = {
    "mess3": build_hidden_markov_model,
    "tom_quantum": build_generalized_hidden_markov_model,
}


@dataclass(frozen=True)
class ProcessComponent:
    """Container describing a single generative process instance."""

    name: str
    process: GenerativeProcess

    @property
    def vocab_size(self) -> int:  # pragma: no cover - simple property
        return self.process.vocab_size

    @property
    def state_dim(self) -> int:  # pragma: no cover - simple property
        return self.process.num_states


def _ensure_unique_name(base: str, counts: dict[str, int]) -> str:
    idx = counts[base]
    counts[base] += 1
    return f"{base}_{idx}" if idx else base


def build_components_from_config(config: Sequence[Mapping[str, Any]]) -> list[ProcessComponent]:
    """Instantiate processes from a simple configuration structure.

    Args:
        config: iterable where each item has at minimum a ``type`` field which
            must be one of the keys in :data:`PROCESS_BUILDERS`. Each item can
            optionally provide ``instances`` (list of parameter dictionaries)
            or ``params``/``count``. Examples::

                [
                    {"type": "mess3", "instances": [{"x": 0.1, "a": 0.7}, {"x": 0.2, "a": 0.6}]},
                    {"type": "tom_quantum", "params": {"alpha": 1.0, "beta": 7.14}, "count": 2},
                ]

    Returns:
        A list of :class:`ProcessComponent` instances.
    """

    components: list[ProcessComponent] = []
    type_counts: dict[str, int] = {key: 0 for key in PROCESS_BUILDERS.keys()}

    for entry in config:
        process_type = entry.get("type")
        if process_type not in PROCESS_BUILDERS:
            available = ", ".join(PROCESS_BUILDERS.keys())
            raise KeyError(f"Unknown process type '{process_type}'. Available types: {available}")

        builder = PROCESS_BUILDERS[process_type]
        name_prefix = entry.get("name_prefix", process_type)

        instances: Iterable[Mapping[str, Any]]
        if "instances" in entry:
            instances = entry["instances"]
        else:
            params = entry.get("params", {})
            count = int(entry.get("count", 1))
            instances = [params for _ in range(count)]

        for inst in instances:
            params = dict(inst)
            explicit_name = params.pop("name", None)
            name = explicit_name or _ensure_unique_name(name_prefix, type_counts)
            process = builder(process_type, **params)  # type: ignore[arg-type]
            components.append(ProcessComponent(name=name, process=process))

    if not components:
        raise ValueError("Configuration produced zero process components")

    return components


class MultipartiteSampler:
    """Generate sequences from the Cartesian product of independent processes."""

    def __init__(self, components: Sequence[ProcessComponent]):
        if not components:
            raise ValueError("MultipartiteSampler requires at least one component")
        self.components = list(components)
        self.component_names = [c.name for c in self.components]
        self.component_state_dims = [c.state_dim for c in self.components]
        self.component_vocab_sizes = [c.vocab_size for c in self.components]
        self._bases = self._compute_bases(self.component_vocab_sizes)
        self.vocab_size = reduce(mul, self.component_vocab_sizes, 1)
        self.belief_dim = sum(self.component_state_dims)

    @staticmethod
    def _compute_bases(vocab_sizes: Sequence[int]) -> list[int]:
        bases: list[int] = []
        running = 1
        for size in reversed(vocab_sizes[1:]):
            running *= size
        for idx, size in enumerate(vocab_sizes):
            if idx == len(vocab_sizes) - 1:
                bases.append(1)
            else:
                bases.append(running)
                running //= vocab_sizes[idx + 1]
        return bases

    def sample(
        self,
        rng_key: jax.Array,
        batch_size: int,
        sequence_len: int,
    ) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
        """Return ``(new_rng_key, beliefs, tokens, component_tokens)``.

        ``beliefs`` has shape ``(batch, sequence_len, belief_dim)`` formed by
        concatenating the belief states of each component along the last axis.
        ``tokens`` contains the product-space observations encoded lexicographically.
        ``component_tokens`` retains each component's raw observations.
        """

        keys = jax.random.split(rng_key, len(self.components) + 1)
        new_key, component_keys = keys[0], keys[1:]

        beliefs_list: list[jnp.ndarray] = []
        tokens_list: list[jnp.ndarray] = []

        for comp, comp_key in zip(self.components, component_keys, strict=True):
            batch_keys = jax.random.split(comp_key, batch_size)
            initial_state = jnp.tile(comp.process.initial_state, (batch_size, 1))
            states, obs = comp.process.generate(initial_state, batch_keys, sequence_len, True)
            if states.ndim == 2:
                states = states[:, None, :]
            beliefs_list.append(states)
            tokens_list.append(obs)

        beliefs = jnp.concatenate(beliefs_list, axis=-1)

        tokens = jnp.zeros_like(tokens_list[0])
        for obs, base in zip(tokens_list, self._bases, strict=True):
            tokens = tokens + obs * base

        return new_key, beliefs, tokens, tokens_list


def build_multipartite_sampler(config: Sequence[Mapping[str, Any]]) -> MultipartiteSampler:
    """Convenience helper that instantiates components and wraps them in a sampler."""

    components = build_components_from_config(config)
    return MultipartiteSampler(components)


__all__ = [
    "MultipartiteSampler",
    "ProcessComponent",
    "build_components_from_config",
    "build_multipartite_sampler",
]
