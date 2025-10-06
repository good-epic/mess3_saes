#!/usr/bin/env python
# coding: utf-8

#%%   8]:


import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from simplexity.generative_processes.builder import build_hidden_markov_model, build_generalized_hidden_markov_model
from simplexity.generative_processes.mixed_state_presentation import LogMixedStateTreeGenerator, MyopicEntropies


#%%   9]:


def estimate_entropy_rate_monte_carlo(transition_matrices, seq_length=100000, n_sequences=10):
    """
    Estimate entropy rate by sampling sequences and computing empirical entropy.
    """
    transition_matrices = np.array(transition_matrices, dtype=np.float64)
    n_tokens, n_states, _ = transition_matrices.shape

    # Find stationary distribution over hidden states
    marginal_transitions = transition_matrices.sum(axis=0)
    eigenvalues, eigenvectors = np.linalg.eig(marginal_transitions.T)
    stationary_idx = np.argmax(np.isclose(eigenvalues, 1, atol=1e-6))
    pi_states = np.real(eigenvectors[:, stationary_idx])
    pi_states = pi_states / pi_states.sum()

    total_log_prob = 0.0
    total_steps = 0

    for _ in range(n_sequences):
        # Start from stationary distribution
        state = np.random.choice(n_states, p=pi_states)
        belief = pi_states.copy()

        for t in range(seq_length):
            # Sample (token, next_state) pair jointly from current state
            # P(token, next_state | current_state)
            joint_probs = transition_matrices[:, state, :].reshape(-1)
            joint_probs = joint_probs / joint_probs.sum()

            joint_idx = np.random.choice(len(joint_probs), p=joint_probs)
            token = joint_idx // n_states
            next_state = joint_idx % n_states

            # Compute P(token | belief) - marginalize over all states and next states
            pred_prob = 0.0
            for s in range(n_states):
                pred_prob += belief[s] * transition_matrices[token, s, :].sum()

            if pred_prob > 0:
                total_log_prob += np.log(pred_prob)
                total_steps += 1

            # Bayesian update: P(S_t | X_t = token, belief_{t-1})
            # P(S_t = s' | token, belief) ∝ Σ_s belief[s] × P(token, s' | s)
            new_belief = np.zeros(n_states)
            for s_next in range(n_states):
                for s_prev in range(n_states):
                    new_belief[s_next] += belief[s_prev] * transition_matrices[token, s_prev, s_next]

            # Normalize
            if new_belief.sum() > 0:
                belief = new_belief / new_belief.sum()

            state = next_state

    entropy_rate = -total_log_prob / total_steps
    return entropy_rate


# entropy_rate_1 = estimate_entropy_rate_monte_carlo(h1.transition_matrices, seq_length=10000, n_sequences=5)
# entropy_rate_2 = estimate_entropy_rate_monte_carlo(h2.transition_matrices, seq_length=10000, n_sequences=5)
# entropy_rate_3 = estimate_entropy_rate_monte_carlo(h3.transition_matrices, seq_length=10000, n_sequences=5)

# print(f"Estimated Entropy Rate: {entropy_rate_1:.4f} bits per token")
# print(f"Estimated Entropy Rate: {entropy_rate_2:.4f} bits per token")
# print(f"Estimated Entropy Rate: {entropy_rate_3:.4f} bits per token")
# print(f"This is the theoretical minimum cross-entropy for a predictive model")


#%%   10]:


def estimate_bloch_walk_entropy_mc(alpha=1, beta=np.sqrt(51), 
                                   n_sequences=10000, seq_length=1000):
    """
    Estimate conditional entropy via Monte Carlo sampling.
    Much faster than exhaustive exploration.
    """
    gamma = 1 / (2 * np.sqrt(alpha**2 + beta**2))

    # Transition matrices
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
        # Start from stationary belief
        belief = np.array([1.0, 0.0, 0.0])

        for _ in range(seq_length):
            # Compute P(token | belief)
            token_probs = np.array([belief @ T[i] @ right_vec for i in range(4)])

            # Sample token
            token = np.random.choice(4, p=token_probs)

            # Log probability for entropy calculation
            total_log_prob += np.log(token_probs[token])
            total_steps += 1

            # Update belief
            belief = (belief @ T[token]) / token_probs[token]

    # Entropy = -E[log P(X_t | X_{1:t-1})]
    entropy = -total_log_prob / total_steps

    return entropy


# tqe1 = estimate_bloch_walk_entropy_mc(alpha=1.12, beta=5.64, n_sequences=3000, seq_length=1000)
# tqe2 = estimate_bloch_walk_entropy_mc(alpha=0.88, beta=8.64, n_sequences=3000, seq_length=1000)
# print(f"Estimated Conditional Entropy: {tqe1:.4f} bits per token")
# print(f"Estimated Conditional Entropy: {tqe2:.4f} bits per token")


#%%   11]:


# Grid scan for Tom Quantum parameters
tq_alpha_values = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
tq_beta_values = [3.0, 5.0, 7.0, 9.0, 11.0]

tq_results = []
ghmm = build_generalized_hidden_markov_model("tom_quantum", alpha=1.0, beta=7.1)
tq_d_vocab = ghmm.vocab_size

print("\nScanning Tom Quantum parameter space...")
for alpha in tqdm(tq_alpha_values):
    for beta in tq_beta_values:
        try:
            # Estimate conditional entropy
            ce = estimate_bloch_walk_entropy_mc(
                alpha=alpha, 
                beta=beta, 
                n_sequences=2000, 
                seq_length=500
            )

            # Build transition matrices for analysis
            gamma = 1 / (2 * np.sqrt(alpha**2 + beta**2))

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

            # Aggregate transition matrix metrics
            all_probs = np.concatenate([t.flatten() for t in T])

            # Compute average transition matrix for spectral analysis
            T_avg = np.mean(T, axis=0)

            # Eigenvalue analysis of average matrix
            eigvals, eigvecs = np.linalg.eig(T_avg)
            eigvals_sorted = np.sort(np.abs(eigvals))[::-1]
            eigvals_real = np.sort(np.real(eigvals))[::-1]
            eigvals_imag_max = np.max(np.abs(np.imag(eigvals)))

            # Singular value decomposition
            singular_vals = np.linalg.svd(T_avg, compute_uv=False)
            cond_number = singular_vals[0] / singular_vals[-1] if singular_vals[-1] > 1e-10 else np.inf

            # Matrix structure metrics
            diag = np.diag(T_avg)
            upper_tri = np.triu(T_avg, k=1).sum()
            lower_tri = np.tril(T_avg, k=-1).sum()

            # Per-token matrix metrics
            eigenvalues_per_token = []
            for t_mat in T:
                eigs = np.linalg.eigvals(t_mat)
                eigenvalues_per_token.append(np.sort(np.abs(eigs))[::-1])
            eigenvalues_per_token = np.array(eigenvalues_per_token)

            tq_results.append({
                'alpha': alpha,
                'beta': beta,
                'conditional_entropy': ce,
                'gamma': gamma,
                'alpha_sq_minus_beta_sq_term': (alpha**2 - beta**2)*gamma**2,
                'alpha_beta_term': 2*alpha*beta*gamma**2,
                'min_transition_prob': all_probs.min(),
                'max_transition_prob': all_probs.max(),
                'mean_transition_prob': all_probs.mean(),
                'std_transition_prob': all_probs.std(),
                'nonzero_prob_count': (all_probs > 1e-10).sum(),
                # Spectral metrics for average matrix
                'spectral_radius': eigvals_sorted[0],
                'largest_eigenval': eigvals_sorted[0],
                'second_eigenval': eigvals_sorted[1] if len(eigvals_sorted) > 1 else 0.0,
                'third_eigenval': eigvals_sorted[2] if len(eigvals_sorted) > 2 else 0.0,
                'largest_real_eigenval': eigvals_real[0],
                'max_imag_eigenval': eigvals_imag_max,
                'condition_number': cond_number,
                'determinant': np.linalg.det(T_avg),
                'trace': diag.sum(),
                'frobenius_norm': np.linalg.norm(T_avg, 'fro'),
                'max_diagonal': diag.max(),
                'min_diagonal': diag.min(),
                'upper_triangle_sum': upper_tri,
                'lower_triangle_sum': lower_tri,
                # Per-token eigenvalue statistics
                'mean_max_eigenval_per_token': eigenvalues_per_token[:, 0].mean(),
                'std_max_eigenval_per_token': eigenvalues_per_token[:, 0].std(),
                'mean_second_eigenval_per_token': eigenvalues_per_token[:, 1].mean(),
            })
        except Exception as e:
            print(f"Error for alpha={alpha}, beta={beta}: {e}")
            continue

tq_df = pd.DataFrame(tq_results)
tq_df.to_csv('tom_quantum_parameter_scan.csv', index=False)
print(f"\nTom Quantum scan complete: {len(tq_df)} parameter combinations")
print(tq_df.head())


#%%   12]:


# Grid scan for Mess3 parameters
mess3_a_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
mess3_x_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45]

mess3_results = []
mess3_d_vocab = None

print("Scanning Mess3 parameter space...")
for a in tqdm(mess3_a_values):
    for x in mess3_x_values:
        try:
            # Build HMM
            hmm = build_hidden_markov_model("mess3", a=a, x=x)
            if mess3_d_vocab is None:
                mess3_d_vocab = hmm.vocab_size

            # Get marginal transition matrix
            tmat = hmm.transition_matrices.sum(axis=0)
            tmat = np.array(tmat, dtype=np.float64)

            # Monte Carlo estimate (faster, fewer sequences for grid scan)
            mc_ce = estimate_entropy_rate_monte_carlo(
                hmm.transition_matrices,
                seq_length=500,
                n_sequences=2000
            )

            # Eigenvalue analysis
            eigvals, eigvecs = np.linalg.eig(tmat)
            eigvals_sorted = np.sort(np.abs(eigvals))[::-1]
            non_unit_eig = eigvals_sorted[1] if len(eigvals_sorted) > 1 else 0.0

            # Matrix metrics
            diag = np.diag(tmat)
            upper_tri = np.triu(tmat, k=1).sum()
            lower_tri = np.tril(tmat, k=-1).sum()

            # Singular values for condition number
            singular_vals = np.linalg.svd(tmat, compute_uv=False)
            cond_number = singular_vals[0] / singular_vals[-1] if singular_vals[-1] > 1e-10 else np.inf

            # Off-diagonal variance
            off_diag = tmat.copy()
            np.fill_diagonal(off_diag, 0)
            off_diag_var = np.var(off_diag[off_diag > 0]) if (off_diag > 0).any() else 0.0

            # 1. Interaction features
            a_times_x = a * x
            a_squared_times_x = a**2 * x
            a_times_x_squared = a * x**2
            a_over_x_ratio = a / (x + 1e-10)

            # 2. Eigenvalue information metrics
            # Normalize eigenvalues to sum to 1 (treating as probability distribution)
            eigvals_abs = np.abs(eigvals)
            eigvals_normalized = eigvals_abs / eigvals_abs.sum()
            # Entropy of eigenvalue distribution
            eigval_entropy = -np.sum(eigvals_normalized * np.log(eigvals_normalized + 1e-10))
            # Effective rank
            effective_rank = (eigvals_abs.sum())**2 / (eigvals_abs**2).sum()

            # 3. Matrix asymmetry metrics
            # Asymmetry norm: ||T - T^T||_F / ||T||_F
            asymmetry_matrix = tmat - tmat.T
            matrix_asymmetry_norm = np.linalg.norm(asymmetry_matrix, 'fro') / (np.linalg.norm(tmat, 'fro') + 1e-10)
            # Diagonal dominance: max diagonal / mean of off-diagonals
            off_diag_mean = off_diag[off_diag > 0].mean() if (off_diag > 0).any() else 1e-10
            diagonal_dominance = diag.max() / off_diag_mean

            # 4. Probability distribution metrics
            # Flatten all transition probabilities
            all_probs = tmat.flatten()
            all_probs = all_probs[all_probs > 1e-10]  # Filter out zeros

            # Gini coefficient
            sorted_probs = np.sort(all_probs)
            n = len(sorted_probs)
            cumsum = np.cumsum(sorted_probs)
            gini_coefficient = (2 * np.sum((np.arange(1, n+1)) * sorted_probs)) / (n * cumsum[-1]) - (n + 1) / n

            # Participation ratio
            participation_ratio = 1.0 / (all_probs**2).sum()

            # Transition entropy (effective number of transitions)
            prob_entropy = -np.sum(all_probs * np.log(all_probs + 1e-10))
            transition_entropy = np.exp(prob_entropy)

            mess3_results.append({
                'a': a,
                'x': x,
                'conditional_entropy': mc_ce,
                'max_diagonal': diag.max(),
                'trace': diag.sum(),
                'upper_triangle_sum': upper_tri,
                'lower_triangle_sum': lower_tri,
                'non_unit_eigenvalue': non_unit_eig,
                'frobenius_norm': np.linalg.norm(tmat, 'fro'),
                'spectral_radius': eigvals_sorted[0],
                'condition_number': cond_number,
                'determinant': np.linalg.det(tmat),
                'off_diagonal_variance': off_diag_var,
                # Interaction features
                'a_times_x': a_times_x,
                'a_squared_times_x': a_squared_times_x,
                'a_times_x_squared': a_times_x_squared,
                'a_over_x_ratio': a_over_x_ratio,
                # Eigenvalue information metrics
                'eigenvalue_entropy': eigval_entropy,
                'effective_rank': effective_rank,
                # Matrix asymmetry metrics
                'matrix_asymmetry_norm': matrix_asymmetry_norm,
                'diagonal_dominance': diagonal_dominance,
                # Probability distribution metrics
                'gini_coefficient': gini_coefficient,
                'participation_ratio': participation_ratio,
                'transition_entropy': transition_entropy,
            })
        except Exception as e:
            print(f"Error for a={a}, x={x}: {e}")
            continue

mess3_df = pd.DataFrame(mess3_results)
mess3_df.to_csv('mess3_parameter_scan.csv', index=False)
print(f"\nMess3 scan complete: {len(mess3_df)} parameter combinations")
print(mess3_df.head())


#%%    ]:


# Visualizations for Tom Quantum
print("\nGenerating Tom Quantum visualizations...")

# 1. Heatmap of conditional entropy vs (alpha, beta)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

tq_ce_matrix = tq_df.pivot(index='beta', columns='alpha', values='conditional_entropy')

# Custom annotation: show value and value/tq_d_vocab as percentage
def make_annot(val):
    if pd.isnull(val):
        return ""
    percent = 100 * val / np.log(tq_d_vocab)
    return f"{val:.3f}\n({percent:.1f}%)"

annot = tq_ce_matrix.applymap(make_annot)

sns.heatmap(tq_ce_matrix, annot=annot, fmt='', cmap='plasma', ax=ax)
ax.set_title('Tom Quantum Conditional Entropy')
ax.set_xlabel('alpha')
ax.set_ylabel('beta')

plt.tight_layout()
plt.savefig('tom_quantum_entropy_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. Scatter plots: CE vs each metric
tq_metric_cols = [col for col in tq_df.columns if col not in ['alpha', 'beta', 'conditional_entropy']]

n_metrics = len(tq_metric_cols)
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

for idx, metric in enumerate(tq_metric_cols):
    ax = axes[idx]
    scatter = ax.scatter(tq_df[metric], tq_df['conditional_entropy'], 
                        c=tq_df['alpha'], cmap='coolwarm', s=50, alpha=0.7)
    ax.set_xlabel(metric)
    ax.set_ylabel('Conditional Entropy')
    ax.set_title(f'CE vs {metric}')
    plt.colorbar(scatter, ax=ax, label='alpha')

    # Add correlation
    corr = np.corrcoef(tq_df[metric], tq_df['conditional_entropy'])[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide unused subplots
for idx in range(len(tq_metric_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('tom_quantum_ce_vs_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print("Tom Quantum visualizations saved!")


#%%    ]:


# Visualizations for Mess3
print("\nGenerating Mess3 visualizations...")

# 1. Heatmap of conditional entropy vs (a, x)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Reshape for heatmap
ce_matrix = mess3_df.pivot(index='x', columns='a', values='conditional_entropy')

# Prepare custom annotation: main value, then (percentage of d_vocab)
annot = ce_matrix.copy().astype(str)
for i in range(ce_matrix.shape[0]):
    for j in range(ce_matrix.shape[1]):
        val = ce_matrix.iloc[i, j]
        if pd.isnull(val):
            annot.iloc[i, j] = ""
        else:
            percent = 100 * val / np.log(mess3_d_vocab)
            annot.iloc[i, j] = f"{val:.3f}\n({percent:.1f}%)"

sns.heatmap(ce_matrix, annot=annot.values, fmt='', cmap='viridis', ax=ax)
ax.set_title('Mess3 Conditional Entropy (Monte Carlo)')
ax.set_xlabel('a (asymmetry)')
ax.set_ylabel('x (noise)')

plt.tight_layout()
plt.savefig('mess3_entropy_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. Scatter plots: CE vs each matrix metric
metric_cols = [col for col in mess3_df.columns if col not in ['a', 'x', 'conditional_entropy']]

n_metrics = len(metric_cols)
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

for idx, metric in enumerate(metric_cols):
    ax = axes[idx]
    scatter = ax.scatter(mess3_df[metric], mess3_df['conditional_entropy'], 
                        c=mess3_df['a'], cmap='coolwarm', s=50, alpha=0.7)
    ax.set_xlabel(metric)
    ax.set_ylabel('Conditional Entropy')
    ax.set_title(f'CE vs {metric}')
    plt.colorbar(scatter, ax=ax, label='a (asymmetry)')

    # Add correlation
    corr = np.corrcoef(mess3_df[metric], mess3_df['conditional_entropy'])[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide unused subplots
for idx in range(len(metric_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('mess3_ce_vs_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print("Mess3 visualizations saved!")


#%%    ]:


# Summary statistics and correlation analysis
print("\n" + "="*60)
print("MESS3 SUMMARY STATISTICS")
print("="*60)
print(f"\nConditional Entropy range: [{mess3_df['conditional_entropy'].min():.4f}, {mess3_df['conditional_entropy'].max():.4f}]")
print(f"Mean CE: {mess3_df['conditional_entropy'].mean():.4f}")
print(f"Std CE: {mess3_df['conditional_entropy'].std():.4f}")

print("\nTop 5 correlations with CE:")
correlations = {}
for col in mess3_df.columns:
    if col not in ['a', 'x', 'conditional_entropy']:
        corr = np.corrcoef(mess3_df[col], mess3_df['conditional_entropy'])[0, 1]
        correlations[col] = corr

for metric, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"  {metric:30s}: r = {corr:7.4f}")

print("\n" + "="*60)
print("TOM QUANTUM SUMMARY STATISTICS")
print("="*60)
print(f"\nConditional Entropy range: [{tq_df['conditional_entropy'].min():.4f}, {tq_df['conditional_entropy'].max():.4f}]")
print(f"Mean CE: {tq_df['conditional_entropy'].mean():.4f}")
print(f"Std CE: {tq_df['conditional_entropy'].std():.4f}")

print("\nTop 5 correlations with CE:")
tq_correlations = {}
for col in tq_df.columns:
    if col not in ['alpha', 'beta', 'conditional_entropy']:
        corr = np.corrcoef(tq_df[col], tq_df['conditional_entropy'])[0, 1]
        tq_correlations[col] = corr

for metric, corr in sorted(tq_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"  {metric:30s}: r = {corr:7.4f}")

print("\n" + "="*60)
print("FILES SAVED")
print("="*60)
print("  - mess3_parameter_scan.csv")
print("  - tom_quantum_parameter_scan.csv")
print("  - mess3_entropy_heatmap.png")
print("  - mess3_ce_vs_metrics.png")
print("  - tom_quantum_entropy_heatmap.png")
print("  - tom_quantum_ce_vs_metrics.png")
print("="*60)

#%%

test_hmm = build_hidden_markov_model("mess3", a=0.9, x=0.05)
print(test_hmm.transition_matrices)
