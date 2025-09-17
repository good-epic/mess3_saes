#%%
# === Import and Config === #
#############################
import os, re, glob, json, sys, argparse
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import torch
import numpy as np
import jax, jax.numpy as jnp
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from transformer_lens import HookedTransformer, HookedTransformerConfig
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch

from BatchTopK.sae import TopKSAE, VanillaSAE

from mess3_gmg import spectral_clustering_with_eigengap, build_similarity_matrix

#%%
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Mess3 GMG: load SAEs, cluster decoder directions, sample latents")
    # Paths and runtime
    parser.add_argument("--sae_folder", type=str, default="outputs/saes", help="Folder containing saved SAE .pt files")
    parser.add_argument("--model_ckpt", type=str, default="outputs/mess3_transformer.pt", help="Transformer checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device override")
    #parser.add_argument("--jax_platform", type=str, default="cpu", choices=["cpu", "gpu", "tpu"], help="JAX platform")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Mess3 process
    parser.add_argument("--process_name", type=str, default="mess3", help="Process name for build_hidden_markov_model")
    parser.add_argument("--mess3_x", type=float, default=0.1, help="Mess3 x parameter")
    parser.add_argument("--mess3_a", type=float, default=0.7, help="Mess3 a parameter")

    # Transformer config
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_ctx", type=int, default=10)
    parser.add_argument("--d_head", type=int, default=8)
    parser.add_argument("--act_fn", type=str, default="relu")

    # Clustering + similarity
    parser.add_argument("--sim_metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="Similarity metric for decoder directions")
    parser.add_argument("--cluster_method", type=str, default="eigengap", choices=["eigengap"], help="Clustering method to use")
    parser.add_argument("--max_clusters", type=int, default=10, help="Max clusters for spectral eigengap search")
    parser.add_argument("--plot_eigengap", action="store_true", help="Plot eigengap diagnostics")

    # SAE selection
    parser.add_argument("--k", type=int, nargs="+", default=[3], help="TopK values to include")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[], help="Vanilla lambda values to include")
    parser.add_argument("--sites", type=str, nargs="+", default=None, help="Subset of site names (e.g., layer_0 layer_1)")

    # Sampling and features
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length for sampling; defaults to n_ctx if None")
    parser.add_argument("--layer_index", type=int, default=1, help="Transformer layer index for resid_post activations")

    # Be robust in interactive contexts (e.g., #%% cells): ignore unknown args
    args, _ = parser.parse_known_args()
    return args


# Parse args first to configure environment
ARGS = _parse_args()

if ARGS.device == "auto":
    _device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    _device = ARGS.device
device = _device
torch.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)

print(f"Device: {device}")

import jax
print(jax.devices())  # should show only CPU now


#%%
# === Load Mess3 process and transformer === #
##############################################
mess3 = build_hidden_markov_model(ARGS.process_name, x=ARGS.mess3_x, a=ARGS.mess3_a)
cfg = HookedTransformerConfig(
    d_model=ARGS.d_model, n_heads=ARGS.n_heads, n_layers=ARGS.n_layers, n_ctx=ARGS.n_ctx,
    d_vocab=mess3.vocab_size, act_fn=ARGS.act_fn, device=device, d_head=ARGS.d_head
)
model = HookedTransformer(cfg).to(device)
ckpt = torch.load(ARGS.model_ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])  # type: ignore
model.eval()


#%%
# === Load SAEs === #
#####################
loaded_saes = {}
for f in glob.glob(os.path.join(ARGS.sae_folder, "*.pt")):
    base = os.path.basename(f)
    if "top_k_k" in base:
        m = re.match(r"(.+)_top_k_k(\d+)\.pt$", base)
        if not m: continue
        site_name, k = m.group(1), int(m.group(2))
        ckpt = torch.load(f, map_location=device, weights_only=False)
        loaded_saes.setdefault(site_name, {})[(k, None)] = ckpt
    elif "vanilla_lambda_" in base:
        m = re.match(r"(.+)_vanilla_lambda_([0-9.eE+-]+)\.pt$", base)
        if not m: continue
        site_name, lam = m.group(1), float(m.group(2))
        ckpt = torch.load(f, map_location=device, weights_only=False)
        loaded_saes.setdefault(site_name, {})[(None, lam)] = ckpt
print(f"Loaded SAEs for {len(loaded_saes)} sites")


#%%
## === Choose SAE(s) === #
#########################

## In the real code we'll need to choose which SAE to use. Potentially try a few?
## One idea is the elbow in the L2 penalty plot for top K SAE reconstruction error
## by k. In this single code case it's k = 3, so we'll hardcode that. 
site_names = list(loaded_saes.keys())
k_lambdas = [(3, None)]

#%%
# === Collect and cluster decoder directions === #
##################################################
decoder_dirs = {}
sim_matrices = {}
cluster_labels = {}
spectral_ks = {}
sae_objs = {}  # reuse later for encoding
# Later want to add in procrustes analysis to align decoder directions across
# layers and cluster them all together?
#
# Also when expanding to large dimensions we'll have to avoid the full cosine sim
# matrix and instead calculate a sparse matrix of top k nearest neighbors. Then we'll
# do spectral clustering on the sparse matrix.
for (k, lam) in k_lambdas:
    for site_name in site_names:
        print(f"\n\n{site_name=}, {k=}, {lam=}")
        ckpt = loaded_saes[site_name][(k, lam)]
        if k is not None:
            sae_class = TopKSAE
            sae_type = "top_k"
        else:
            sae_class = VanillaSAE
            sae_type = "L1"
        sae = sae_class(ckpt["cfg"]).to(device)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()
        d_key = (site_name, sae_type, k if sae_type == "top_k" else lam)
        sae_objs[d_key] = sae
        decoder_dirs[d_key] = sae.W_dec.detach().cpu().numpy()
        # I think cosine makes most sense here because we're clustering decoder directions
        sim_matrices[d_key] = build_similarity_matrix(data=decoder_dirs[d_key], method=ARGS.sim_metric)
        if ARGS.cluster_method == "eigengap":
            cluster_labels[d_key], spectral_ks[d_key] = spectral_clustering_with_eigengap(sim_matrices[d_key], max_clusters=10, plot=True)
        else:
            raise ValueError(f"Unknown cluster method: {ARGS.cluster_method}")
        # Worry about other cluster methods later.
        # elif cluster_method == "co-occurence":
        #     cluster_labels[d_key], spectral_ks[d_key] = spectral_clustering_with_eigengap(sim_matrices[d_key], max_clusters=10, plot=True)

        # print(f"{labels=}")
        # print(f"SAE={d_key}")
        # print("\nCluster assignments for decoder directions:")
        # for idx, lbl in enumerate(labels):
        #     print(f"SAE={d_key}, latent {idx} -> cluster {lbl}")



## WORKING HERE 9/14/25
## Double check the code below to make sure it works with the way all the data is collected.
## Next steps:
## 1) Do a co-occurence clustering like in Tegmark's paper. I think that means do clustering
##    with their Phi as the similarity metric. Then they did spectral clustering there too.
##
## 2) Bring in Adam's multipartite code so we can try a Mess3 multipartite model, and then a
##    Mess3 + Tom Quantum multipartite model.
##
## 3) Work with GPT to implement GMG algorithm and run it here on the multipartite models.
##    Should also spend some tim just poking around the spectral clustering once we have
##    the multipartite models working.




#%%
### This needs to be edited to work with the multipartite models.

# === Generate Mess3 samples and compute latent activations === #
#################################################################
batch_size = ARGS.batch_size
seq_len = (ARGS.seq_len if ARGS.seq_len is not None else cfg.n_ctx)
key = jax.random.PRNGKey(ARGS.seed)
gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)
_, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len+1, key)
tokens = torch.tensor(np.array(inputs)).long().to(device)

# Run through transformer
with torch.no_grad():
    _, cache = model.run_with_cache(tokens, return_type=None)

# Example: use configurable layer index residuals
acts = cache[f"blocks.{ARGS.layer_index}.hook_resid_post"].reshape(-1, cfg.d_model).to(device)

latent_samples = {}
for (site_name, sae_type, kval_or_lam), sae in sae_objs.items():
    with torch.no_grad():
        z = sae.encode(acts)  # latent activations
        latent_samples[(site_name, sae_type, kval_or_lam)] = z.cpu().numpy()
        print(f"Latents for {site_name}, SAE=({sae_type},{kval_or_lam}): {z.shape}")

print("\nDone: loaded SAEs, clustered decoder dirs, and generated latent samples.")

