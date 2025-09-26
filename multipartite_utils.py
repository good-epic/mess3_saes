 """Utilities for sampling from multipartite generative processes.

This module provides lightweight helpers that combine multiple simplexity
generative processes (e.g. mess3 and tom_quantum) into a single sampler that
emits product-space tokens while keeping track of the component belief states.
The intent is to enable SAE training on stacked processes without having to
re-implement a full joint HMM.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig

from simplexity.generative_processes.builder import (
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
)
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.torch_generator import generate_data_batch

from mess3_gmg_analysis_utils import sae_encode_features


# Canonical order matching legacy training pipeline: tom components precede mess3
PROCESS_BUILDERS = {
    "tom_quantum": build_generalized_hidden_markov_model,
    "mess3": build_hidden_markov_model,
}

COMPONENT_TYPE_PRIORITY: dict[str, int] = {"tom_quantum": 0, "mess3": 1}


@dataclass(frozen=True)
class ProcessComponent:
    """Container describing a single generative process instance."""

    name: str
    process_type: str
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
            components.append(ProcessComponent(name=name, process_type=process_type, process=process))

    if not components:
        raise ValueError("Configuration produced zero process components")

    return components


class MultipartiteSampler:
    """Generate sequences from the Cartesian product of independent processes."""

    def __init__(self, components: Sequence[ProcessComponent]):
        if not components:
            raise ValueError("MultipartiteSampler requires at least one component")
        enumerated = list(enumerate(components))
        default_priority = len(COMPONENT_TYPE_PRIORITY)
        ordered = sorted(
            enumerated,
            key=lambda pair: (
                COMPONENT_TYPE_PRIORITY.get(pair[1].process_type, default_priority),
                pair[0],
            ),
        )
        self.components = [comp for _, comp in ordered]
        self.component_names = [c.name for c in self.components]
        self.component_state_dims = [c.state_dim for c in self.components]
        self.component_vocab_sizes = [c.vocab_size for c in self.components]
        self._bases = self._compute_bases(self.component_vocab_sizes)
        self.vocab_size = reduce(mul, self.component_vocab_sizes, 1)
        self.belief_dim = sum(self.component_state_dims)
        # Keep tuples for JIT so component count remains static
        self._components_tuple = tuple(self.components)
        self._bases_tuple = tuple(int(b) for b in self._bases)
        self._jax_sampler = self._build_jax_sampler()

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

    def sample_python(
        self,
        rng_key: jax.Array,
        batch_size: int,
        sequence_len: int,
    ) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
        """Return ``(new_rng_key, beliefs, tokens, component_tokens)``.

        ``beliefs`` has shape ``(batch, sequence_len - 1, belief_dim)`` formed by
        concatenating the belief states of each component along the last axis.
        ``tokens`` contains the product-space "input" observations (last time step dropped)
        encoded lexicographically. ``component_tokens`` retains each component's raw
        "inputs" so downstream code mirrors :func:`generate_data_batch` semantics.
        """
        if sequence_len < 2:
            raise ValueError("sequence_len must be at least 2 so inputs and next tokens can be formed")

        keys = jax.random.split(rng_key, len(self.components) + 1)
        new_key, component_keys = keys[0], keys[1:]

        beliefs_list: list[jnp.ndarray] = []
        inputs_list: list[jnp.ndarray] = []

        for comp, comp_key in zip(self.components, component_keys, strict=True):
            batch_keys = jax.random.split(comp_key, batch_size)
            initial_state = jnp.tile(comp.process.initial_state, (batch_size, 1))
            states, obs = comp.process.generate(initial_state, batch_keys, sequence_len, True)

            if states.ndim == 2:  # single-step fallback shouldn't trigger, but guard just in case
                states = states[:, None, :]

            # Drop the final time step so the returned tokens align with the "inputs"
            # convention used throughout the simplexity training utilities.
            states = states[:, :-1, :]
            inputs = obs[:, :-1]

            beliefs_list.append(states)
            inputs_list.append(inputs)

        beliefs = jnp.concatenate(beliefs_list, axis=-1)

        tokens = jnp.zeros_like(inputs_list[0])
        for obs, base in zip(inputs_list, self._bases, strict=True):
            tokens += obs * base

        return new_key, beliefs, tokens, inputs_list

    def _build_jax_sampler(self):
        components = self._components_tuple
        bases = self._bases_tuple
        num_components = len(components)

        def _jax_sample_impl(
            rng_key: jax.Array,
            batch_size: int,
            sequence_len: int,
        ) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, ...]]:
            keys = jax.random.split(rng_key, num_components + 1)
            new_key = keys[0]
            component_keys = keys[1:]

            beliefs_list = []
            inputs_list = []
            for comp, comp_key in zip(components, component_keys, strict=True):
                initial_state = jnp.tile(comp.process.initial_state, (batch_size, 1))
                batch_keys = jax.random.split(comp_key, batch_size)
                states, obs = comp.process.generate(initial_state, batch_keys, sequence_len, True)
                states = states[:, :-1, :]
                inputs = obs[:, :-1]
                beliefs_list.append(states)
                inputs_list.append(inputs)

            beliefs = jnp.concatenate(beliefs_list, axis=-1)
            tokens = jnp.zeros_like(inputs_list[0])
            for obs, base in zip(inputs_list, bases, strict=True):
                tokens = tokens + obs * base

            return new_key, beliefs, tokens, tuple(inputs_list)

        return jax.jit(_jax_sample_impl, static_argnums=(1, 2))

    def sample(
        self,
        rng_key: jax.Array,
        batch_size: int,
        sequence_len: int,
    ) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
        if sequence_len < 2:
            raise ValueError("sequence_len must be at least 2 so inputs and next tokens can be formed")
        new_key, beliefs, tokens, inputs_tuple = self._jax_sampler(rng_key, batch_size, sequence_len)
        inputs_list = [jnp.asarray(arr) for arr in inputs_tuple]
        return new_key, beliefs, tokens, inputs_list


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

def _resolve_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def _load_process_stack(args: argparse.Namespace, preset_process_configs) -> tuple[list[dict], list, object]:
    if args.process_config:
        with open(args.process_config, "r", encoding="utf-8") as f:
            process_cfg_raw = json.load(f)
    elif args.process_preset:
        if args.process_preset not in preset_process_configs:
            raise ValueError(f"Unknown process preset '{args.process_preset}'")
        process_cfg_raw = deepcopy(preset_process_configs[args.process_preset])
    else:
        process_cfg_raw = deepcopy(preset_process_configs["single_mess3"])

    components = build_components_from_config(process_cfg_raw)
    data_source: object
    if len(components) == 1:
        data_source = components[0].process
    else:
        data_source = MultipartiteSampler(components)
    return process_cfg_raw, components, data_source


def _load_transformer(args: argparse.Namespace, device: str, vocab_size: int):
    cfg = HookedTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_ctx=args.n_ctx,
        d_vocab=args.d_vocab if args.d_vocab is not None else vocab_size,
        act_fn=args.act_fn,
        device=device,
        d_head=args.d_head,
    )
    model = HookedTransformer(cfg).to(device)
    ckpt = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt.get("config"), dict):
        cfg_loaded = HookedTransformerConfig.from_dict(ckpt["config"])
        model = HookedTransformer(cfg_loaded).to(device)
        cfg = cfg_loaded

    state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state_dict is None:
        available = ", ".join(sorted(ckpt.keys()))
        raise KeyError(
            "Checkpoint does not contain 'state_dict' or 'model_state_dict'. "
            f"Available keys: {available}"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
    if missing or unexpected:
        print(
            "Warning: load_state_dict reported issues",
            {"missing": missing, "unexpected": unexpected},
        )
    model.eval()
    return model, cfg


def _select_sites(
    metrics_summary: Mapping[str, Any],
    requested_sites: Sequence[str] | None,
    site_hook_map: Mapping[str, str],
) -> list[str]:
    """Return the subset of ``requested_sites`` present in the metrics summary.

    If ``requested_sites`` is ``None`` all sites defined in ``site_hook_map``
    that have entries in ``metrics_summary`` are returned in the map order.
    Unknown site names are ignored with a warning.
    """

    available = [site for site in site_hook_map if site in metrics_summary]
    if requested_sites is None:
        return available

    valid = [site for site in requested_sites if site in site_hook_map]
    missing = sorted(set(requested_sites) - set(valid))
    if missing:
        print(f"Warning: ignoring unknown sites {missing}")

    return [site for site in valid if site in metrics_summary]


def _sample_tokens(
    data_source,
    batch_size: int,
    sample_len: int,
    target_len: int | None,
    seed: int,
    device: str,
) -> torch.Tensor:
    if sample_len < 2:
        raise ValueError("sample_len must be at least 2 to provide input/target pairs")
    key = jax.random.PRNGKey(seed)
    if isinstance(data_source, MultipartiteSampler):
        key, beliefs, tokens, _ = data_source.sample(key, batch_size, sample_len)
        _ = beliefs  # unused, but kept for clarity
        arr = np.array(tokens)
    else:
        gen_states = jnp.repeat(data_source.initial_state[None, :], batch_size, axis=0)
        _, inputs, _ = generate_data_batch(gen_states, data_source, batch_size, sample_len, key)
        arr = np.array(inputs)
    effective_len = arr.shape[1]
    if target_len is None:
        target_len = sample_len - 1
    if effective_len > target_len:
        arr = arr[:, :target_len]
        effective_len = target_len
    elif effective_len < target_len:
        if target_len - effective_len > 1:
            raise ValueError(
                f"Sampled sequence length {effective_len} smaller than target length {target_len}"
            )
        # Allow a single-token discrepancy (sequence dropped final token upstream).
    if arr.shape[1] < 2:
        raise ValueError("Sampled sequences must provide at least two tokens")
    tokens = torch.from_numpy(arr).long().to(device)
    return tokens


def collect_latent_activity_data(
    model,
    sae,
    data_source,
    hook_name: str,
    *,
    batch_size: int,
    sample_len: int,
    target_len: int,
    n_batches: int,
    seed: int,
    device: str,
    activation_eps: float,
    collect_matrix: bool = False,
):
    if n_batches <= 0:
        raise ValueError("activation sampling requires n_batches >= 1")

    dict_size = sae.W_dec.shape[0]
    active_counts = torch.zeros(dict_size, dtype=torch.float64)
    mean_abs_sum = torch.zeros(dict_size, dtype=torch.float64)
    total_samples = 0
    binary_batches: list[torch.Tensor] = [] if collect_matrix else []

    for batch_idx in range(n_batches):
        tokens = _sample_tokens(
            data_source,
            batch_size,
            sample_len,
            target_len,
            seed + batch_idx + 1,
            device,
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None)
            acts = cache[hook_name].reshape(-1, cache[hook_name].shape[-1]).to(device)
            feature_acts, _, _ = sae_encode_features(sae, acts)
        mask = (feature_acts.abs() > activation_eps)
        active_counts += mask.sum(dim=0).to(torch.float64).cpu()
        mean_abs_sum += feature_acts.abs().sum(dim=0).to(torch.float64).cpu()
        total_samples += mask.shape[0]

        if collect_matrix:
            binary_batches.append(mask.cpu())

        del cache
        del acts
        del feature_acts
        del mask
        del tokens

    activity_rates = (active_counts / max(total_samples, 1)).numpy()
    mean_abs_activation = (mean_abs_sum / max(total_samples, 1)).numpy()
    latent_matrix = None
    if collect_matrix:
        latent_matrix = torch.cat(binary_batches, dim=0).numpy()
    return {
        "activity_rates": activity_rates,
        "mean_abs_activation": mean_abs_activation,
        "latent_matrix": latent_matrix,
        "total_samples": total_samples,
    }
