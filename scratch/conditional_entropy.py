#!/usr/bin/env python3
"""
Estimate conditional entropy of synthetic generators (e.g., Mess3, Tom Quantum).

This script can be used via CLI or imported to call `compute_conditional_entropy`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplexity.generative_processes.builder import build_hidden_markov_model, build_generalized_hidden_markov_model


def estimate_mess3_entropy_mc(
    transition_matrices,
    seq_length: int = 10000,
    n_sequences: int = 1000,
    seed: int | None = None,
) -> float:
    """
    Estimate the entropy rate H(X_t | X_{<t}) using Monte Carlo sampling.
    """
    if seed is not None:
        np.random.seed(seed)

    transition_matrices = np.array(transition_matrices, dtype=np.float64)
    n_tokens, n_states, _ = transition_matrices.shape

    marginal_transitions = transition_matrices.sum(axis=0)
    eigvals, eigvecs = np.linalg.eig(marginal_transitions.T)
    stationary_idx = np.argmax(np.isclose(eigvals, 1, atol=1e-6))
    pi_states = np.real(eigvecs[:, stationary_idx])
    pi_states = pi_states / pi_states.sum()

    total_log_prob = 0.0
    total_steps = 0

    for _ in range(n_sequences):
        state = np.random.choice(n_states, p=pi_states)
        belief = pi_states.copy()

        for _ in range(seq_length):
            joint_probs = transition_matrices[:, state, :].reshape(-1)
            joint_probs = joint_probs / joint_probs.sum()

            joint_idx = np.random.choice(len(joint_probs), p=joint_probs)
            token = joint_idx // n_states
            next_state = joint_idx % n_states

            pred_prob = 0.0
            for s in range(n_states):
                pred_prob += belief[s] * transition_matrices[token, s, :].sum()

            if pred_prob > 0:
                total_log_prob += np.log(pred_prob)
                total_steps += 1

            new_belief = np.zeros(n_states)
            for s_next in range(n_states):
                for s_prev in range(n_states):
                    new_belief[s_next] += belief[s_prev] * transition_matrices[token, s_prev, s_next]
            if new_belief.sum() > 0:
                belief = new_belief / new_belief.sum()

            state = next_state

    return float(-total_log_prob / total_steps)


def estimate_tom_quantum_entropy_mc(
    alpha: float,
    beta: float,
    *,
    seq_length: int = 1000,
    n_sequences: int = 10000,
    seed: int | None = None,
) -> float:
    """
    Monte Carlo conditional entropy estimator for Tom Quantum process
    (Bloch walk). Uses explicit transition matrices derived from alpha/beta.
    """
    if seed is not None:
        np.random.seed(seed)

    gamma = 1.0 / (2.0 * np.sqrt(alpha**2 + beta**2))

    T = [
        np.array([[1/4, 0, 2*alpha*beta*gamma**2],
                  [0, (alpha**2 - beta**2)*gamma**2, 0],
                  [2*alpha*beta*gamma**2, 0, 1/4]]),
        np.array([[1/4, 0, -2*alpha*beta*gamma**2],
                  [0, (alpha**2 - beta**2)*gamma**2, 0],
                  [-2*alpha*beta*gamma**2, 0, 1/4]]),
        np.array([[1/4, 2*alpha*beta*gamma**2, 0],
                  [2*alpha*beta*gamma**2, 1/4, 0],
                  [0, 0, (alpha**2 - beta**2)*gamma**2]]),
        np.array([[1/4, -2*alpha*beta*gamma**2, 0],
                  [-2*alpha*beta*gamma**2, 1/4, 0],
                  [0, 0, (alpha**2 - beta**2)*gamma**2]])
    ]
    right_vec = np.array([1.0, 0.0, 0.0])

    total_log_prob = 0.0
    total_steps = 0

    for _ in range(n_sequences):
        belief = np.array([1.0, 0.0, 0.0])
        for _ in range(seq_length):
            token_probs = np.array([belief @ T[i] @ right_vec for i in range(4)])
            token = np.random.choice(4, p=token_probs)

            total_log_prob += np.log(token_probs[token])
            total_steps += 1

            belief = (belief @ T[token]) / token_probs[token]

    return float(-total_log_prob / total_steps)


def build_generator(generator: str, **params):
    if generator == "mess3":
        required = {"a", "x"}
        missing = required - params.keys()
        if missing:
            raise ValueError(f"Missing Mess3 parameters: {missing}")
        return build_hidden_markov_model("mess3", a=params["a"], x=params["x"])
    elif generator == "tom_quantum":
        required = {"alpha", "beta"}
        missing = required - params.keys()
        if missing:
            raise ValueError(f"Missing Tom Quantum parameters: {missing}")
        return build_generalized_hidden_markov_model("tom_quantum", alpha=params["alpha"], beta=params["beta"])
    else:
        raise ValueError(f"Unsupported generator '{generator}'")


def compute_conditional_entropy(
    generator: str,
    *,
    seq_length: int = 10000,
    n_sequences: int = 1000,
    seed: int | None = None,
    **params,
) -> Dict[str, Any]:
    hmm = build_generator(generator, **params)
    if generator == "mess3":
        entropy_nats = estimate_mess3_entropy_mc(
            hmm.transition_matrices,
            seq_length=seq_length,
            n_sequences=n_sequences,
            seed=seed,
        )
    elif generator == "tom_quantum":
        entropy_nats = estimate_tom_quantum_entropy_mc(
            alpha=params["alpha"],
            beta=params["beta"],
            seq_length=seq_length,
            n_sequences=n_sequences,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported generator {generator}")
    entropy_bits = entropy_nats / np.log(2)
    return {
        "generator": generator,
        "parameters": params,
        "seq_length": seq_length,
        "n_sequences": n_sequences,
        "seed": seed,
        "conditional_entropy_nats": entropy_nats,
        "conditional_entropy_bits": entropy_bits,
        "perplexity": float(np.exp(entropy_nats)),
        "vocab_size": hmm.vocab_size,
    }


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate conditional entropy of Mess3/Tom-Quantum generators")
    parser.add_argument("--generator", choices=["mess3", "tom_quantum"], required=True)
    parser.add_argument("--a", type=float, help="Mess3 asymmetry parameter")
    parser.add_argument("--x", type=float, help="Mess3 noise parameter")
    parser.add_argument("--alpha", type=float, help="Tom Quantum alpha parameter")
    parser.add_argument("--beta", type=float, help="Tom Quantum beta parameter")
    parser.add_argument("--seq-length", type=int, default=10000)
    parser.add_argument("--n-sequences", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON results")
    return parser.parse_args()


def args_to_params(args: argparse.Namespace) -> Dict[str, float]:
    if args.generator == "mess3":
        return {"a": args.a, "x": args.x}
    elif args.generator == "tom_quantum":
        return {"alpha": args.alpha, "beta": args.beta}
    else:
        raise ValueError


def main() -> None:
    args = parse_cli_args()
    params = args_to_params(args)
    if None in params.values():
        missing = [k for k, v in params.items() if v is None]
        raise ValueError(f"Missing required parameters for {args.generator}: {missing}")

    results = compute_conditional_entropy(
        args.generator,
        seq_length=args.seq_length,
        n_sequences=args.n_sequences,
        seed=args.seed,
        **params,
    )

    print(json.dumps(results, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
