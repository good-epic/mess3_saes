import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools



import numpy as np
import itertools

def phi_similarity(latent_acts, efficient=True, eps=1e-12):
    """
    Compute the Phi (Pearson) correlation coefficient between all pairs of binary
    latent activations.

    Args:
        latent_acts : (N_samples, N_latents) array-like
            Binary activations (0/1 or bool). Nonzero is treated as 1.
        efficient : bool
            If True, use matrix multiplies (fast, memory-heavy).
            If False, use a simple double loop (slow, memory-light).
        eps : float
            Numerical guard for denominators.

    Returns:
        Phi : (N_latents, N_latents) float64 array, symmetric, diag=1.
    """
    A = (latent_acts > eps).astype(np.int64)            # N x p
    N, p = A.shape

    # This is slower but more memory efficient
    if not efficient:
        Phi = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            ai = A[:, i]
            for j in range(i, p):
                aj = A[:, j]
                n11 = np.sum((ai == 1) & (aj == 1))
                n10 = np.sum((ai == 1) & (aj == 0))
                n01 = np.sum((ai == 0) & (aj == 1))
                n00 = np.sum((ai == 0) & (aj == 0))
                denom = np.sqrt((n11+n10)*(n11+n01)*(n00+n10)*(n00+n01)) + eps
                phi = (n11*n00 - n10*n01) / denom
                Phi[i, j] = Phi[j, i] = phi
        np.fill_diagonal(Phi, 1.0)
        return Phi

    # Efficient path: matrix multiplies
    Ac = 1 - A

    # Co-activations
    N11 = (A.T @ A).astype(np.float64)

    # i=1, j=0
    N10 = (A.T @ Ac).astype(np.float64)

    # Symmetry: N01 = N10.T
    # N01 = N10.T

    # Both off
    N00 = N - (N11 + N10 + N10.T)

    # Phi coefficient
    denom = np.sqrt((N11+N10) * (N00+N10.T) * (N11+N10.T) * (N00+N10)) + eps
    Phi = (N11 * N00 - N10 * N10.T) / denom

    # Clean up numerical junk
    Phi[~np.isfinite(Phi)] = 0.0
    np.fill_diagonal(Phi, 1.0)
    np.clip(Phi, -1.0, 1.0, out=Phi)
    return Phi



def build_similarity_matrix(data, method="cosine", latent_acts=None, phi_compute_efficient=True):
    if method == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(data)

    elif method == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        D = euclidean_distances(data)
        return 1.0 / (1.0 + D)

    elif method == "phi":
        if latent_acts is None:
            raise ValueError("Phi similarity requires latent_acts.")
        return phi_similarity(latent_acts, efficient=phi_compute_efficient)

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def spectral_clustering_with_eigengap(sim_matrix, max_clusters=10, random_state=0, plot=False, plot_path=None):
    """
    Run spectral clustering using dense similarity matrix and eigengap heuristic.

    Args:
        sim_matrix (ndarray): n x n similarity matrix (cosine sim).
        max_clusters (int): check eigengaps among first max_clusters eigenvalues.
        random_state (int): for k-means stability.
        plot (bool): if True, plot eigenvalues and highlight the chosen gap.

    Returns:
        labels (ndarray): cluster assignments for each point.
        best_k (int): chosen number of clusters.
    """
    # Degree matrix and normalized Laplacian
    diag = np.diag(sim_matrix.sum(axis=1))
    laplacian = diag - sim_matrix
    sqrt_deg = np.diag(1.0 / np.sqrt(np.maximum(diag.diagonal(), 1e-12)))
    norm_lap = sqrt_deg @ laplacian @ sqrt_deg

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(norm_lap)
    eigvals = np.real(eigvals)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Eigengap heuristic
    usable_count = min(eigvals.shape[0], max_clusters + 1)
    gaps = np.diff(eigvals[:usable_count])
    if gaps.size == 0:
        best_k = 1
    else:
        best_k = int(np.argmax(gaps) + 1)
    print(f"Chosen number of clusters via eigengap: k={best_k}")

    if plot or plot_path is not None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        x_len = usable_count
        ax.plot(range(1, x_len + 1), eigvals[:usable_count], marker="o")
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Eigenvalue spectrum (normalized Laplacian)")
        # Highlight chosen eigengap
        ax.axvline(best_k, color="red", linestyle="--", label=f"Chosen k={best_k}")
        ax.legend()
        fig.tight_layout()
        if plot_path is not None:
            import os
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            fig.savefig(plot_path)
            plt.close(fig)
        elif plot:
            plt.show()

    # Spectral embedding
    embedding = eigvecs[:, :best_k]
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    # K-means on embedding
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(embedding)

    return labels, best_k


def generate_mp_emissions(
    key,
    n_tom_quantum,
    n_mess3,
    tom_stationaries,
    mess3_stationaries,
    batch_size,
    seq_len,
    tom_quantum_processes,
    mess3_processes,
    tom_quantum_vocab_size,
    mess3_vocab_size,
    product_vocab_size,
    device,
):
    """
    Generate a new batch of factored inputs for tom_quantum and mess3 processes,
    and combine them into product space tokens.

    Args:
        key: JAX PRNGKey
        n_tom_quantum: int, number of tom_quantum processes
        n_mess3: int, number of mess3 processes
        tom_stationaries: list of stationary distributions for tom_quantum
        mess3_stationaries: list of stationary distributions for mess3
        batch_size: int, batch size
        seq_len: int, sequence length
        tom_quantum_processes: list of tom_quantum process objects
        mess3_processes: list of mess3 process objects
        tom_quantum_vocab_size: int, vocab size for tom_quantum
        mess3_vocab_size: int, vocab size for mess3
        product_vocab_size: int, total product vocab size
        device: torch device

    Returns:
        key: new JAX PRNGKey after split
        tom_inputs_list: list of np.ndarray, one per tom_quantum process
        mess3_inputs_list: list of np.ndarray, one per mess3 process
        tokens: torch.LongTensor, combined product tokens on device
    """
    key, *subkeys = jax.random.split(key, 1 + n_tom_quantum + n_mess3)

    tom_inputs_list = []
    for i in range(n_tom_quantum):
        tom_states = jnp.repeat(tom_stationaries[i][None, :], batch_size, axis=0)
        _, tom_inputs, _ = generate_data_batch(
            tom_states, tom_quantum_processes[i], batch_size, seq_len, subkeys[i]
        )
        if isinstance(tom_inputs, torch.Tensor):
            tom_inputs_list.append(tom_inputs.cpu().numpy())
        else:
            tom_inputs_list.append(np.array(tom_inputs))

    mess3_inputs_list = []
    for i in range(n_mess3):
        mess3_states = jnp.repeat(mess3_stationaries[i][None, :], batch_size, axis=0)
        _, mess3_inputs, _ = generate_data_batch(
            mess3_states, mess3_processes[i], batch_size, seq_len, subkeys[n_tom_quantum + i]
        )
        if isinstance(mess3_inputs, torch.Tensor):
            mess3_inputs_list.append(mess3_inputs.cpu().numpy())
        else:
            mess3_inputs_list.append(np.array(mess3_inputs))

    # Combine into product space: token = sum(input_i * base_i)
    # The base for each process will be the product of the vocab sizes of the processes that come after it in the combination.
    # For example, if we have tom1, tom2, mess3_1, mess3_2 with vocab sizes T1, T2, M1, M2
    # token = tom1 * (T2 * M1 * M2) + tom2 * (M1 * M2) + mess3_1 * M2 + mess3_2
    # Assuming all tom_quantum have the same vocab size (tom_quantum_vocab_size) and all mess3 have the same (mess3_vocab_size)
    # token = tom1 * (tom_quantum_vocab_size**(n_tom_quantum-1) * mess3_vocab_size**n_mess3) + ... + mess3_2

    # Initialize product_tokens based on whether there are any inputs
    if n_tom_quantum > 0:
        product_tokens = np.zeros_like(tom_inputs_list[0])
    elif n_mess3 > 0:
        product_tokens = np.zeros_like(mess3_inputs_list[0])
    else:
        raise ValueError("Both n_tom_quantum and n_mess3 are zero. Cannot generate data.")

    # Add tom_quantum contributions
    for i in range(n_tom_quantum):
        base_for_tom = (tom_quantum_vocab_size ** (n_tom_quantum - 1 - i)) * (mess3_vocab_size ** n_mess3)
        product_tokens += tom_inputs_list[i] * base_for_tom

    # Add mess3 contributions
    for i in range(n_mess3):
        base_for_mess3 = (mess3_vocab_size ** (n_mess3 - 1 - i))
        product_tokens += mess3_inputs_list[i] * base_for_mess3

    tokens = torch.from_numpy(product_tokens).long().to(device)

    return key, tom_inputs_list, mess3_inputs_list, tokens
