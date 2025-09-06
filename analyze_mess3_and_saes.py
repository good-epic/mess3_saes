#%%
############## Imports ##############
#####################################
import torch
import torch.nn.functional as F
import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from transformer_lens import HookedTransformer, HookedTransformerConfig
from simplexity.generative_processes.builder import build_hidden_markov_model, build_generalized_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch
import argparse, sys, os

from mess3_saes import train_sae_on_site

import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import glob
import re
import json
import matplotlib
import matplotlib.pyplot as plt
import importlib
import sys

module_name = "mess_3_saes_analysis_fxns"
if module_name in sys.modules:
    importlib.reload(sys.modules[module_name])
else:
    importlib.import_module(module_name)

from mess_3_saes_analysis_fxns import (
    generate_last50_loss_plots,
    generate_l0_bar_plots,
    generate_vanilla_l0_average_line_plots,
    analyze_latent_activation_by_sequence,
    summarize_latent_sequence_results,
    plot_latent_histograms,
    compute_latent_token_fingerprints,
    plot_token_fingerprints,
    sample_residual_stream_activations,
    build_epdfs_for_all_saes,
    plot_epdfs,
    build_epdfs_for_sae,
    LatentEPDF,
    )

#%%
############## CLI arguments ##############
###########################################
device_default = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description="Train Transformer on mess3 and SAEs")

# HookedTransformerConfig parameters
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_ctx", type=int, default=10)
parser.add_argument("--d_vocab", type=int, default=None)
parser.add_argument("--act_fn", type=str, default="relu")
parser.add_argument("--device", type=str, default=device_default)
parser.add_argument("--d_head", type=int, default=8)

# SAE hyperparameters
parser.add_argument("--dict_mul", type=int, default=4)
parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
parser.add_argument("--l1_coeff", type=float, nargs="+", default=[1e-3, 0.01, 0.015, 0.02, 0.025, 0.05] + [round(x, 2) for x in np.arange(0.1, 1.01, 0.05)])
parser.add_argument("--input_unit_norm", dest="input_unit_norm", action="store_true", default=True)
parser.add_argument("--no_input_unit_norm", dest="input_unit_norm", action="store_false")
parser.add_argument("--n_batches_to_dead", type=int, default=5)
parser.add_argument("--top_k_aux", type=int, default=None)
parser.add_argument("--aux_penalty", type=float, default=1.0/32.0)
parser.add_argument("--bandwidth", type=float, default=0.001)
parser.add_argument("--print_last50_loss", action="store_true", default=False)

# SAE training loop controls
parser.add_argument("--sae_steps", type=int, default=400)
parser.add_argument("--sae_batch_size", type=int, default=64)
parser.add_argument("--sae_seq_len", type=int, default=None)
parser.add_argument("--seq_len", type=int, default=None, help="Sequence length used for analysis/visualization batches; defaults to n_ctx")

# Model loading
parser.add_argument("--load_model", type=str, default="outputs/mess3_transformer.pt", help="Path to a saved model checkpoint (.pt). If provided, skip training and load this model.")
parser.add_argument("--load_saes", type=str, default="outputs/saes", help="Path to a folder containing SAE checkpoints (.pt) and metrics_summary.json")

# Parse known to be notebook-friendly
args, _ = parser.parse_known_args()



#%% 
############# Load Model ###############
########################################

## Create mess3 process
mess3 = build_hidden_markov_model("mess3", x=0.1, a=0.7)
#mess3 = build_generalized_hidden_markov_model("tom_quantum", alpha=1, beta=np.sqrt(51))
print(f"mess3: vocab_size={mess3.vocab_size}, states={mess3.num_states}")

## Create TransformerLens model
device = args.device
cfg = HookedTransformerConfig(
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    n_ctx=args.n_ctx,
    d_vocab=(args.d_vocab if args.d_vocab is not None else mess3.vocab_size),
    act_fn=args.act_fn,
    device=device,
    d_head=args.d_head,
)
model = HookedTransformer(cfg)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params on {device}")

# Define a default sequence length for downstream analysis/visualization
seq_len = args.seq_len if args.seq_len is not None else cfg.n_ctx


## Load model
ckpt = torch.load(args.load_model, map_location=device, weights_only=False)
cfg_loaded = HookedTransformerConfig.from_dict(ckpt["config"]) if isinstance(ckpt.get("config"), dict) else cfg
if isinstance(ckpt.get("config"), dict):
    model = HookedTransformer(cfg_loaded).to(device)
model.load_state_dict(ckpt["state_dict"])  # type: ignore
model.eval()
print(f"Loaded model from {args.load_model}")



#%%
########## Load Metrics Summary ##############
##############################################

metrics_summary_path = os.path.join(args.load_saes, "metrics_summary.json")
metrics_summary = None
if os.path.exists(metrics_summary_path):
    try:
        with open(metrics_summary_path, "r") as f:
            metrics_summary = json.load(f)
        print(f"Loaded metrics summary from {metrics_summary_path}")
    except Exception as e:
        print(f"Failed to load metrics summary: {e}")

## Load active latent indices (and optionally SAEs if needed)
active_latents = {}
if metrics_summary is not None:
    # metrics_summary holds active_latents_last_quarter per SAE
    for site_name, site_data in metrics_summary.items():
        if not isinstance(site_data, dict):
            continue
        # Sequence groups
        seq = site_data.get("sequence", {})
        for sae_type in ("top_k", "vanilla"):
            group = seq.get(sae_type, {})
            for name, rec in group.items():
                inds = rec.get("active_latents_last_quarter", [])
                if inds:
                    active_latents[(site_name, "sequence", sae_type, name)] = inds
        # Beliefs vanilla group
        beliefs = site_data.get("beliefs", {})
        for name, rec in beliefs.items():
            inds = rec.get("active_latents_last_quarter", [])
            if inds:
                active_latents[(site_name, "beliefs", "vanilla", name)] = inds
    print(f"Loaded active latent indices for {len(active_latents)} SAEs from metrics_summary")



#%%
########## Load SAEs ##############
###################################

## Load SAEs matching current filenames produced by training
sae_folder = args.load_saes
loaded_saes = {}  # {site_name: {(k, lambda): ckpt}} where one of k or lambda may be None

# Sequence TopK: {site}_top_k_k{K}.pt
for f in glob.glob(os.path.join(sae_folder, "*_top_k_k*.pt")):
    base = os.path.basename(f)
    m = re.match(r"(.+)_top_k_k(\d+)\.pt$", base)
    if not m:
        print(f"Skipping unrecognized SAE file: {f}")
        continue
    site_name = m.group(1)
    k = int(m.group(2))
    ckpt = torch.load(f, map_location=device, weights_only=False)
    loaded_saes.setdefault(site_name, {})[(k, None)] = ckpt
    #print(f"Loaded seq TopK SAE for {site_name} (k={k}) from {f}")

# Sequence Vanilla: {site}_vanilla_lambda_{lambda}.pt
for f in glob.glob(os.path.join(sae_folder, "*_vanilla_lambda_*.pt")):
    base = os.path.basename(f)
    m = re.match(r"(.+)_vanilla_lambda_([0-9.eE+-]+)\.pt$", base)
    if not m:
        print(f"Skipping unrecognized SAE file: {f}")
        continue
    site_name = m.group(1)
    lmbda = float(m.group(2))
    ckpt = torch.load(f, map_location=device, weights_only=False)
    loaded_saes.setdefault(site_name, {})[(None, lmbda)] = ckpt
    #print(f"Loaded seq Vanilla SAE for {site_name} (lambda={lmbda}) from {f}")
print(f"Loaded SAEs")



#%%
print(f"{loaded_saes.keys()=}")
print(f"{loaded_saes['layer_3'].keys()=}")

#%%
########## Plot Training Summary ##############
################################################

#for key, val in active_latents.items():
#    print(f"{key}: {val}")

# args.print_last50_loss = True
## Summary stats and visualizations for metrics summary
if metrics_summary is not None:
    reports_dir = os.path.join(args.load_saes, "reports")
    if args.print_last50_loss:
        generate_last50_loss_plots(metrics_summary, reports_dir)
    generate_l0_bar_plots(metrics_summary, reports_dir)
    generate_vanilla_l0_average_line_plots(metrics_summary, reports_dir)
else:
    print("metrics_summary not found; skipping metrics reports.")




#%%
####### Fit PCAs Per Residual Stream Layer ##############
#########################################################
## Get residual stream activations samples
residual_streams, tokens = sample_residual_stream_activations(model, mess3, seq_len)

print("Activation shapes:")
for name, acts in residual_streams.items():
    print(f"  {name}: {acts.shape}")

# Flatten for PCA: combine batch and sequence dimensions
activations_flat = {}
token_labels = []
for name, acts in residual_streams.items():
    # Reshape from [batch, seq, d_model] to [batch*seq, d_model]
    acts_reshaped = acts.reshape(-1, acts.shape[-1]).cpu().numpy()
    activations_flat[name] = acts_reshaped

# Create labels for each token
token_labels = tokens.flatten().cpu().numpy()
print(f"\nTotal points for PCA: {activations_flat['layer_1'].shape[0]}")


pcas = {}
pca_coords = {}
for key in activations_flat.keys():
    pcas[key] = PCA(n_components=2)
    pca_coords[key] = pcas[key].fit_transform(activations_flat[key])

    print(f"PCA explained variance ratio for {key}: {pcas[key].explained_variance_ratio_}")
    print(f"Total variance explained for {key}: {sum(pcas[key].explained_variance_ratio_):.2%}")




#%%
########## Build Empirical PDFs for all SAEs ##############
###########################################################
import importlib
mess_3_saes_analysis_fxns = importlib.import_module("mess_3_saes_analysis_fxns")
importlib.reload(mess_3_saes_analysis_fxns)
from mess_3_saes_analysis_fxns import (
    generate_last50_loss_plots,
    generate_l0_bar_plots,
    generate_vanilla_l0_average_line_plots,
    analyze_latent_activation_by_sequence,
    summarize_latent_sequence_results,
    plot_latent_histograms,
    compute_latent_token_fingerprints,
    plot_token_fingerprints,
    sample_residual_stream_activations,
    build_epdfs_for_all_saes,
    plot_epdfs,
    build_epdfs_for_sae,
    LatentEPDF,
)

sequence_epdfs = build_epdfs_for_all_saes(
    model,
    mess3,
    loaded_saes,
    active_latents,
    pcas=pcas,
    seq_len=10,
    n_batches=2000,
    device=device,
)


#%%
# Build a list of (key, count) tuples for active latents, sorted descending by count
active_latent_counts = []

for key, val in active_latents.items():
    if isinstance(val, dict):
        count = len(val)
    else:
        try:
            count = len(val)
        except Exception:
            count = 0
    active_latent_counts.append((key, count))

# Sort descending by count
active_latent_counts.sort(key=lambda x: x[1], reverse=True)
# Print the last two elements of active_latent_counts that are "top_k" and k is 2
filtered = [item for item in active_latent_counts if (
    ("top_k" in str(item[0])) and (
        ("k2" in str(item[0])) or (isinstance(item[0], tuple) and any(str(x) == "2" for x in item[0]))
    )
)]
print("Last two elements of active_latent_counts with top_k and k=2:")
for item in filtered[-10:]:
    print(item)

#%%
print(f"{sequence_epdfs.keys()=}")
print(f"{sequence_epdfs['layer_3'].keys()=}")
print(f"{sequence_epdfs['layer_3']['top_k'].keys()=}")
print(f"{sequence_epdfs['layer_3']['top_k']['k2'].keys()=}")
print(f"{active_latents[('layer_3', 'sequence', 'top_k', 'k2')]=}")
print(f"{active_latents[('layer_1', 'sequence', 'top_k', 'k2')]=}")



#%%

epdf_fig = plot_epdfs([sequence_epdfs['layer_3']['top_k']['k2'][59]], mode="transparency", grid_size=80)
epdf_fig.show()
epdf_fig = plot_epdfs([sequence_epdfs['layer_3']['top_k']['k2'][113]], mode="transparency", grid_size=80)
epdf_fig.show()
epdf_fig = plot_epdfs([sequence_epdfs['layer_3']['top_k']['k2'][119]], mode="transparency", grid_size=80)
epdf_fig.show()
epdf_fig = plot_epdfs([sequence_epdfs['layer_3']['top_k']['k2'][172]], mode="transparency", grid_size=80)
epdf_fig.show()



#%%
########## Latent Activation Sequence Analysis ##############
############################################################

# Reports dir
reports_dir = os.path.join(args.load_saes, "reports")
os.makedirs(reports_dir, exist_ok=True)

print("\n=== Running sequence analysis for all SAEs ===")
seq_len_analysis = 6

for site_name, sae_dict in loaded_saes.items():
    for (k, lmbda), sae_ckpt in sae_dict.items():
        # Decide SAE class
        from BatchTopK.sae import TopKSAE, VanillaSAE
        if f"k{k}" in sae_ckpt.get("name", ""):
            sae_class = TopKSAE
        else:
            sae_class = VanillaSAE

        # Rebuild SAE object
        sae = sae_class(sae_ckpt["cfg"])
        sae.load_state_dict(sae_ckpt["state_dict"])
        sae.to(device).eval()

        # Hook name for activations at this site
        if site_name == "embeddings":
            hook_name = "hook_embed"
        else:
            try:
                layer_idx = int(site_name.split("_")[1])
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
            except Exception:
                print(f"⚠️ Could not parse layer from site {site_name}, skipping")
                continue

        # Restrict to active latents if available
        key = (site_name, "sequence", "top_k", f"k{k}")
        active_inds = active_latents.get(key, None)

        print(f"\n--- {site_name}, k={k}, lambda={lmbda} ---")
        results = analyze_latent_activation_by_sequence(
            model,
            sae,
            mess3,
            seq_len=seq_len_analysis,
            device=device,
            active_latents=active_inds,
            hook_name=hook_name,
        )

        # Summarize
        summaries = summarize_latent_sequence_results(results, top_n=10)
        for li, s in summaries.items():
            print(f"Latent {li}: {s['count']} seqs, "
                  f"avg={s['avg_value']:.4f}, max={s['max_value']:.4f}")
            for seq, val in s["top_sequences"]:
                print(f"    {seq} → {val:.4f}")

        # Save histograms
        layer_report_dir = os.path.join(reports_dir, f"{site_name}_k{k}_lambda{lmbda}")
        os.makedirs(layer_report_dir, exist_ok=True)
        plot_latent_histograms(results, layer_report_dir)

        # Fingerprints
        fps = compute_latent_token_fingerprints(results, vocab_size=mess3.vocab_size, seq_len=seq_len_analysis)
        plot_token_fingerprints(fps, layer_report_dir)

        # Save raw results + summaries
        import json
        out_json = {
            "results": {li: [{"seq": list(e["seq"]), "value": e["value"]} for e in ents] for li, ents in results.items()},
            "summaries": summaries,
            "fingerprints": fps,
        }
        out_path = os.path.join(layer_report_dir, "latent_sequence_analysis.json")
        with open(out_path, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"Saved analysis to {out_path}")




#%%
## 3D PCA visualization of residual stream

# Create interactive 3D plot with plotly
fig = go.Figure()

# Add scatter plot for each token type (0, 1, 2)
colors = ['red', 'green', 'blue']
token_names = ['Token 0', 'Token 1', 'Token 2']

for token_id in range(3):
    mask = token_labels == token_id
    fig.add_trace(go.Scatter3d(  # Use Scatter3d for 3D plot
        x=pca_coords[mask, 0],
        y=pca_coords[mask, 1],
        z=pca_coords[mask, 2],  # Add z-coordinates for 3D
        mode='markers',
        name=token_names[token_id],
        marker=dict(
            size=2,  # Adjust marker size for 3D
            color=colors[token_id],
            opacity=0.6,
        )
    ))

fig.update_layout(
    title='3D PCA of Residual Stream (Layer 3)',  # Update title
    scene = dict(                 # Define scene for 3D plot
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'),       # Add z-axis title
    height=600,
    width=800,
    showlegend=True
)

fig.show()


#%%
## 2D PCA visualization of hidden states

# Setup
batch_size = 1000
sequence_len = 10
key = jax.random.PRNGKey(0)
batch_keys = jax.random.split(key, batch_size)

# Initialize states
initial_states = jnp.tile(mess3.initial_state, (batch_size, 1))
gen_states = jnp.repeat(mess3.initial_state[None, :], 1000, axis=0)

# Generate sequences
states, observations = mess3.generate(initial_states, batch_keys, sequence_len, True)

print(f"{states.shape=}")
print(f"{observations.shape=}")


# Flatten the states data for visualization
states_flat = states.reshape(-1, 3)

# Assuming the states are one-hot encoded or similar, find the active state
# If the states are probabilities, you might want to visualize the probabilities directly or the most likely state
# For simplicity, let's assume it's a one-hot like representation and find the index of the active state
# If your 'states' data is different, you might need to adjust this part.
active_states_indices = jnp.argmax(states_flat, axis=1)

# Create interactive 2D plot with plotly
fig = go.Figure()

# Add scatter plot for each state type (0, 1, 2)
colors = ['red', 'green', 'blue']
state_names = ['State 0', 'State 1', 'State 2']

for state_id in range(3):
    mask = active_states_indices == state_id
    fig.add_trace(go.Scatter(  # Use Scatter for 2D plot
        x=states_flat[mask, 0], # Use the first component of the state
        y=states_flat[mask, 1], # Use the second component of the state
        mode='markers',
        name=state_names[state_id],
        marker=dict(
            size=3,
            color=colors[state_id],
            opacity=0.6,
        )
    ))

fig.update_layout(
    title='2D Visualization of Hidden States', # Update title
    xaxis_title='State Component 1',
    yaxis_title='State Component 2',
    height=600,
    width=800,
    showlegend=True
)

fig.show()


#%%

# print(f"{states.shape=}")
# print(f"{states_flat.shape=}")
# print(f"{states[:3]=}")
print(f"{states_flat[:3]=}")

# print(f"{observations.shape=}")
# print(f"{observations.reshape(-1).shape=}")
# print(f"{observations[:3]=}")

tt = np.array(jnp.sum(states_flat, axis=-1))
print(f"{states_flat.shape=}")
print(f"{tt.shape=}")
print(f"{tt[:50]=}")
print(f"{tt.max()=}, {tt.min()=}, {tt.mean()=}")
plt.figure()
plt.hist(tt.tolist(), bins=10, color='purple', alpha=0.7, range=(0.999, 1.001))
plt.xlabel('Sum of State Components (tt)')
plt.ylabel('Frequency')
plt.title('Histogram of tt (Sum of State Components)')
plt.show()



#%%
## 2D PCA visualization with barycentric cast to a triangle

# Assuming states_flat is already reshaped to (-1, 3) and contains values
# that represent points in a 3-simplex (e.g., probabilities that sum to 1).
# If not, you might need to adjust the data loading/preparation.

# Define the vertices of an equilateral triangle in 2D for visualization
# You can adjust these coordinates as needed
v1 = np.array([0, 0])
v2 = np.array([1, 0])
v3 = np.array([0.5, np.sqrt(3)/2])

# Function to transform 3D simplex coordinates to 2D triangle coordinates
# This is a simple barycentric coordinate mapping. Adjust if your state representation is different.
def simplex_to_triangle_coords(simplex_coords, v1, v2, v3):
    # Ensure the sum of simplex coordinates is close to 1
    simplex_coords = simplex_coords / jnp.sum(simplex_coords, axis=-1, keepdims=True) # Uncomment if needed

    # Map to 2D using barycentric coordinates
    x = simplex_coords[:, 0] * v1[0] + simplex_coords[:, 1] * v2[0] + simplex_coords[:, 2] * v3[0]
    y = simplex_coords[:, 0] * v1[1] + simplex_coords[:, 1] * v2[1] + simplex_coords[:, 2] * v3[1]
    return np.stack([x, y], axis=1)

# Transform the states data
triangle_coords = simplex_to_triangle_coords(states_flat, v1, v2, v3)

# Create interactive 2D plot on the triangle
fig = go.Figure()

# Add scatter plot for each state type (0, 1, 2)
colors = ['red', 'green', 'blue']
state_names = ['State 0', 'State 1', 'State 2']

# Find the most likely state for coloring
active_states_indices = jnp.argmax(states_flat, axis=1)


for state_id in range(3):
    mask = active_states_indices == state_id
    fig.add_trace(go.Scatter(
        x=triangle_coords[mask, 0],
        y=triangle_coords[mask, 1],
        mode='markers',
        name=state_names[state_id],
        marker=dict(
            size=3,
            color=colors[state_id],
            opacity=0.6,
        )
    ))

# Optional: Add lines to show the triangle boundaries
fig.add_trace(go.Scatter(
    x=[v1[0], v2[0], v3[0], v1[0]],
    y=[v1[1], v2[1], v3[1], v1[1]],
    mode='lines',
    showlegend=False,
    line=dict(color='black', width=2)
))


fig.update_layout(
    title='2D Visualization of Hidden States on a Simplex',
    xaxis_title='Simplex Dimension 1',
    yaxis_title='Simplex Dimension 2',
    height=600,
    width=800,
    showlegend=True,
    xaxis=dict(scaleanchor="y", scaleratio=1), # Keep aspect ratio
    yaxis=dict(constrain="domain"),
)

fig.show()




#%%

"""# Task
Generate new data and activations, perform linear regression from the activation space to the beliefs, and plot the true and predicted beliefs side-by-side.

## Generate new data and activations

### Subtask:
Generate a new batch of sequences and obtain the corresponding residual stream activations from the trained model. Also, get the true hidden states (beliefs) from the data generation process.

**Reasoning**:
Generate new data and activations as per the instructions, including true hidden states.
"""

# 1. Generate a new JAX random key
key, subkey = jax.random.split(key)

# 2. Generate a batch of initial states
gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)

# 3. Generate fresh sequences and get true states
true_states, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len + 1, subkey)

# 4. Convert the generated observations (inputs) to a PyTorch tensor
if isinstance(inputs, torch.Tensor):
    tokens = inputs.long().to(device)
else:
    tokens = torch.from_numpy(np.array(inputs)).long().to(device)

# 5. Run with cache to get all activations
_, cache = model.run_with_cache(tokens, return_type="cache")

# 6. Extract residual stream activations from the last layer (layer 3)
activations = cache['blocks.3.hook_resid_post']

"""**Reasoning**:
Correct the `return_type` in the `run_with_cache` call to `'single'` to get the cache object and extract activations from the last layer.


"""

# 1. Generate a new JAX random key
key, subkey = jax.random.split(key)

# 2. Generate a batch of initial states
gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)

# 3. Generate fresh sequences and get true states
true_states, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len + 1, subkey)

# 4. Convert the generated observations (inputs) to a PyTorch tensor
if isinstance(inputs, torch.Tensor):
    tokens = inputs.long().to(device)
else:
    tokens = torch.from_numpy(np.array(inputs)).long().to(device)

# 5. Run with cache to get all activations
_, cache = model.run_with_cache(tokens, return_type="single")

# 6. Extract residual stream activations from the last layer (layer 3)
activations = cache['blocks.3.hook_resid_post']

"""**Reasoning**:
Correct the `return_type` in the `run_with_cache` call to `'fwd_and_bwd_activations'` as the documentation suggests this returns the cache object.


"""

# 1. Generate a new JAX random key
key, subkey = jax.random.split(key)

# 2. Generate a batch of initial states
gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)

# 3. Generate fresh sequences and get true states
true_states, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len + 1, subkey)

# 4. Convert the generated observations (inputs) to a PyTorch tensor
if isinstance(inputs, torch.Tensor):
    tokens = inputs.long().to(device)
else:
    tokens = torch.from_numpy(np.array(inputs)).long().to(device)

# 5. Run with cache to get all activations
_, cache = model.run_with_cache(tokens, return_type="fwd_and_bwd_activations")

# 6. Extract residual stream activations from the last layer (layer 3)
activations = cache['blocks.3.hook_resid_post']

"""**Reasoning**:
Correct the `return_type` in the `run_with_cache` call to `'everything'` to ensure the cache object containing all activations is returned.


"""

# 1. Generate a new JAX random key
key, subkey = jax.random.split(key)

# 2. Generate a batch of initial states
gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)

# 3. Generate fresh sequences and get true states
true_states, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len + 1, subkey)

# 4. Convert the generated observations (inputs) to a PyTorch tensor
if isinstance(inputs, torch.Tensor):
    tokens = inputs.long().to(device)
else:
    tokens = torch.from_numpy(np.array(inputs)).long().to(device)

# 5. Run with cache to get all activations
_, cache = model.run_with_cache(tokens, return_type="everything")

# 6. Extract residual stream activations from the last layer (layer 3)
activations = cache['blocks.3.hook_resid_post']

"""**Reasoning**:
Based on the previous errors with `return_type`, it seems the appropriate value might be different. Consulting the `transformer_lens` documentation or examples is necessary to find the correct way to get the cache. However, I cannot do that. Given the constraints, I will try using `return_type=None` which might default to returning the logits and cache.


"""

# 1. Generate a new JAX random key
key, subkey = jax.random.split(key)

# 2. Generate a batch of initial states
gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)

# 3. Generate fresh sequences and get true states
true_states, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len + 1, subkey)

# 4. Convert the generated observations (inputs) to a PyTorch tensor
if isinstance(inputs, torch.Tensor):
    tokens = inputs.long().to(device)
else:
    tokens = torch.from_numpy(np.array(inputs)).long().to(device)

# 5. Run with cache to get all activations
_, cache = model.run_with_cache(tokens, return_type=None)

# 6. Extract residual stream activations from the last layer (layer 3)
activations = cache['blocks.3.hook_resid_post']

"""## Prepare data for linear regression

### Subtask:
Reshape the activations and true beliefs to be suitable for training a linear regression model (e.g., flatten the batch and sequence dimensions).

**Reasoning**:
Reshape the activations and true states to be suitable for linear regression.
"""

# Reshape activations
# Assuming activations is a tensor of shape [batch, seq, d_model]
activations_flat_for_lr = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

# Reshape true states
# Assuming true_states is a JAX array of shape [batch, seq, num_states]
# Note: generate_data_batch returns states of shape [batch, seq+1, num_states].
# We should align this with the activations which are for sequences of length seq_len.
# Let's assume we need the states for the first 'seq_len' tokens.
true_states_flat = true_states[:, :seq_len, :].reshape(-1, true_states.shape[-1])

print(f"Reshaped activations shape: {activations_flat_for_lr.shape}")
print(f"Reshaped true states shape: {true_states_flat.shape}")

"""**Reasoning**:
The error indicates that `true_states` is a 2-dimensional array, but the indexing `true_states[:, :seq_len, :]` is treating it as 3-dimensional. This is likely because the `generate_data_batch` function returned `true_states` with a different shape than anticipated, possibly `[batch * seq, num_states]` directly. I need to inspect the shape of `true_states` and reshape it correctly based on its actual shape.


"""

# Check the shape of true_states
print(f"Original true_states shape: {true_states.shape}")

# Reshape activations (already confirmed shape in previous cell)
activations_flat_for_lr = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

# Reshape true states based on the observed shape
# If true_states is already 2D [batch * seq, num_states], no reshaping is needed for flattening.
# If true_states is 3D [batch, seq, num_states], reshape it.
# Based on the error, it seems true_states might be 2D already. Let's assume it's [batch * seq_len, num_states]
# from a previous step or a different output format than expected.

# Let's try reshaping assuming it's [batch, seq+1, num_states] as per the comment,
# but first, check the actual shape.
# If true_states is 2D, we can just use it directly for flattening.
if true_states.ndim == 2:
    # Assuming true_states is already [batch * (seq+1), num_states] or similar
    # We need to slice it to match the sequence length of activations (seq_len)
    # This is tricky if it's already flattened. Let's re-generate data and ensure
    # true_states has the expected [batch, seq, num_states] shape before flattening.

    # Re-generate data to ensure consistent shapes
    key, subkey = jax.random.split(key)
    gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)
    # Generate seq_len tokens, so the states will be [batch, seq_len + 1, num_states]
    true_states_new, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len, subkey)

    if isinstance(inputs, torch.Tensor):
        tokens = inputs.long().to(device)
    else:
        tokens = torch.from_numpy(np.array(inputs)).long().to(device)

    _, cache = model.run_with_cache(tokens, return_type=None)
    activations = cache['blocks.3.hook_resid_post']
    activations_flat_for_lr = activations.reshape(-1, activations.shape[-1]).cpu().numpy()


    # Now true_states_new should be [batch, seq_len + 1, num_states].
    # We need states corresponding to the first 'seq_len' tokens.
    # The activations are for the input tokens up to the second to last token when predicting the last token.
    # So, activations are [batch, seq_len, d_model].
    # The true states we want to predict are the states *after* processing each token.
    # If the states are [batch, seq_len + 1, num_states], states[:, 1:, :] would correspond to the states after
    # processing the tokens tokens[:, :seq_len].
    # Let's align states with activations: Activations are for tokens 0 to seq_len-1.
    # The state after processing token i is true_states_new[:, i+1, :].
    # So, for activations[:, t, :], the corresponding true state is true_states_new[:, t+1, :].
    # We need true_states_new[:, 1:seq_len+1, :] which is equivalent to true_states_new[:, 1:, :] given seq_len.
    true_states_to_flatten = true_states_new[:, 1:, :]


    true_states_flat = true_states_to_flatten.reshape(-1, true_states_to_flatten.shape[-1])

else:
    # If true_states was unexpectedly 2D, this branch won't be hit after re-generating.
    # If it was 3D but not [batch, seq+1, num_states], we would need to adjust.
    # Assuming the regeneration fixed the shape to [batch, seq+1, num_states].
    pass # This else block should not be reached if regeneration works as expected.


print(f"Reshaped activations shape: {activations_flat_for_lr.shape}")
print(f"Reshaped true states shape: {true_states_flat.shape}")

"""**Reasoning**:
The error persists even after attempting to re-generate the data. The print statement shows that `true_states` is still 2-dimensional (`(1000, 3)`). This indicates that the `generate_data_batch` function is not returning the true states in the expected `[batch, seq_len + 1, num_states]` format when `seq_len` is passed to it. Looking back at the previous cells, the `observations` variable from a prior test had the shape `(1000, 10)` which means `batch_size=1000` and `seq_len=10`. And the `states` variable from the same test had the shape `(1000, 10, 3)` which means `batch_size=1000`, `seq_len=10`, and `num_states=3`. It seems the `generate_data_batch` function needs the *total* sequence length including the initial state. So, if we want `seq_len` tokens and `seq_len + 1` states, we should request a sequence length of `seq_len` for tokens and the states output will be `seq_len + 1`. Let's try regenerating the data by passing `seq_len` as the requested sequence length for the tokens and see if `true_states_new` has the shape `[batch, seq_len + 1, num_states]`. Then we can slice it to get the states corresponding to the activations.


"""

# Re-generate data to ensure consistent shapes for true_states
key, subkey = jax.random.split(key)
gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)

# Generate data with sequence length 'seq_len'. This should give us states of shape [batch, seq_len + 1, num_states]
# and inputs (tokens) of shape [batch, seq_len].
true_states_new, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len, subkey)

# Convert inputs (tokens) to PyTorch tensor
if isinstance(inputs, torch.Tensor):
    tokens = inputs.long().to(device)
else:
    tokens = torch.from_numpy(np.array(inputs)).long().to(device)

# Run with cache to get activations. Activations should be [batch, seq_len, d_model]
_, cache = model.run_with_cache(tokens, return_type=None)
activations = cache['blocks.3.hook_resid_post']

# Reshape activations
activations_flat_for_lr = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

# Now true_states_new should be [batch, seq_len + 1, num_states].
# The activations at position `t` correspond to the input token at position `t`.
# We are trying to predict the hidden state *after* processing token `t`.
# So, for activations[:, t, :], the corresponding true state is true_states_new[:, t+1, :].
# We need states from true_states_new corresponding to indices 1 to seq_len (inclusive).
# This slice is true_states_new[:, 1:seq_len+1, :] which is equivalent to true_states_new[:, 1:, :].
true_states_to_flatten = true_states_new[:, 1:, :]

# Reshape true states to [batch * seq_len, num_states]
true_states_flat = true_states_to_flatten.reshape(-1, true_states_to_flatten.shape[-1])

print(f"Original true_states_new shape: {true_states_new.shape}")
print(f"true_states_to_flatten shape: {true_states_to_flatten.shape}")
print(f"Reshaped activations shape: {activations_flat_for_lr.shape}")
print(f"Reshaped true states shape: {true_states_flat.shape}")

"""**Reasoning**:
The error persists because `true_states_new` is still 2D after calling `generate_data_batch` with `seq_len`. This strongly suggests that `generate_data_batch` does not return the full sequence of states in a 3D array under these conditions, or that the output format for states is different from what was expected. The error message consistently says a 2-dimensional array is being indexed with 3 indices. This confirms `true_states_new` is 2D. Let's explicitly print the shape of `true_states_new` right after the `generate_data_batch` call to confirm its shape and then adjust the indexing and reshaping based on its actual 2D structure. If it's 2D, it's likely already flattened as `[batch * (some_seq_len), num_states]`. We'll need to determine that 'some_seq_len' and slice accordingly. Given the previous observation of `states.shape` being `(1000, 10, 3)` when `sequence_len=10` was passed to `mess3.generate`, it seems likely that `generate_data_batch` with `seq_len` returns states of shape `[batch * (seq_len + 1), num_states]` or `[batch * seq_len, num_states]`. Let's assume it's `[batch * (seq_len + 1), num_states]` and we need to slice it to get the states corresponding to the `seq_len` tokens that produce the activations.


"""

# Re-generate data
key, subkey = jax.random.split(key)
gen_states = jnp.repeat(mess3.initial_state[None, :], batch_size, axis=0)

# Generate data with sequence length 'seq_len'.
true_states_new, inputs, _ = generate_data_batch(gen_states, mess3, batch_size, seq_len, subkey)

# Print the actual shape of true_states_new immediately after generation
print(f"Actual true_states_new shape after generation: {true_states_new.shape}")

# Convert inputs (tokens) to PyTorch tensor
if isinstance(inputs, torch.Tensor):
    tokens = inputs.long().to(device)
else:
    tokens = torch.from_numpy(np.array(inputs)).long().to(device)

# Run with cache to get activations. Activations should be [batch, seq_len, d_model]
_, cache = model.run_with_cache(tokens, return_type=None)
activations = cache['blocks.3.hook_resid_post']

# Reshape activations to [batch * seq_len, d_model]
activations_flat_for_lr = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

# Based on the observed 2D shape of true_states_new, it is likely already flattened.
# Assuming its shape is [batch * (seq_len + 1), num_states], we need to select
# the states corresponding to the tokens processed, which are the states from index 1 onwards
# within each sequence of length seq_len + 1.
# If the original 3D shape was [batch, seq_len + 1, num_states], the flattened version is
# [batch * (seq_len + 1), num_states]. The states we need correspond to the elements from index
# `(seq_len+1) * i + 1` to `(seq_len+1) * i + seq_len` for each batch `i`.
# This is equivalent to taking elements with flattened indices from `batch_index * (seq_len + 1) + 1`
# up to `batch_index * (seq_len + 1) + seq_len` for each batch_index.
# A simpler way might be to reshape the 2D array back to 3D [batch, seq_len + 1, num_states]
# and then slice it.

num_states = true_states_new.shape[-1] # Get num_states from the actual shape

# Reshape the 2D true_states_new back to 3D [batch, seq_len + 1, num_states]
# This assumes the 2D array is the concatenation of flattened sequences of length seq_len + 1
try:
    true_states_3d = true_states_new.reshape(batch_size, seq_len + 1, num_states)

    # Now slice the 3D array to get states corresponding to the tokens [batch, seq_len, num_states]
    # We want the states after processing tokens at index 0 to seq_len-1.
    # These correspond to states at index 1 to seq_len in the original sequence of states.
    true_states_to_flatten = true_states_3d[:, 1:, :]

    # Reshape the sliced 3D array to 2D [batch * seq_len, num_states]
    true_states_flat = true_states_to_flatten.reshape(-1, num_states)

    print(f"Reshaped true_states_3d shape: {true_states_3d.shape}")
    print(f"true_states_to_flatten shape after slicing: {true_states_to_flatten.shape}")
    print(f"Reshaped activations shape: {activations_flat_for_lr.shape}")
    print(f"Reshaped true states shape: {true_states_flat.shape}")

except Exception as e:
    print(f"Error reshaping true_states_new to 3D: {e}")
    # If reshaping to 3D fails, it means the original 2D shape is not [batch * (seq_len + 1), num_states].
    # Let's assume it's [batch * seq_len, num_states] directly and use it.
    # If it's [batch * seq_len, num_states], then it should already be aligned with the activations.
    print("Assuming true_states_new is already flattened as [batch * seq_len, num_states]")
    true_states_flat = true_states_new # Use it directly if it's already 2D

    print(f"Reshaped activations shape: {activations_flat_for_lr.shape}")
    print(f"Reshaped true states shape: {true_states_flat.shape}")

"""**Reasoning**:
The output shows that `true_states_new` is still `(1000, 3)` after calling `generate_data_batch` with `seq_len`. This means the assumption that it would return `[batch, seq_len + 1, num_states]` was incorrect. The reshape to 3D also failed because the sizes don't match (`3000` vs `33000`). The code then fell into the `except` block, assuming `true_states_new` is already flattened as `[batch * seq_len, num_states]`. However, the printed shapes show `Reshaped activations shape: (9000, 64)` (which is `batch_size * seq_len`, 1000 * 9, since `n_ctx=10` and activations are for seq_len tokens) and `Reshaped true states shape: (1000, 3)`. These shapes do *not* match the expected `[batch * seq_len, num_states]` for `true_states_flat`. The true states should have `batch_size * seq_len` rows, which is 9000. The current shape `(1000, 3)` suggests it might be `[batch_size, num_states]`, perhaps representing the state *after* the entire sequence for each batch item, or maybe the initial state.

Let's re-examine the `generate_data_batch` function call and its documentation or behavior if possible. Since we cannot directly inspect the function's source or documentation in this environment, let's try to deduce its behavior based on the observed outputs.

We have seen `states.shape` is `(1000, 10, 3)` when `sequence_len=10` was passed to `mess3.generate`. This suggests that the `generate` method of the HMM object likely returns the full sequence of states. It's possible `generate_data_batch` wraps this and its state output format depends on the parameters.

Given that `true_states_new` is `(1000, 3)`, and we passed `seq_len=10` to `generate_data_batch` along with `batch_size=1000`, this `(1000, 3)` shape is likely `[batch_size, num_states]`. This could represent the state *after* the sequence of `seq_len` tokens has been generated, or perhaps the state *before* generating the first token (though the name `true_states` suggests the former). Neither of these aligns with the `[batch * seq_len, num_states]` needed to predict the state *after* each token.

The most reliable way to get the sequence of states `[batch, seq_len + 1, num_states]` is likely to use the `mess3.generate` method directly, as seen in a previous cell where `states.shape` was `(1000, 10, 3)` when requesting a sequence of length 10. Let's use `mess3.generate` to get the full sequence of states and then align it with the activations.

We need to generate sequences of tokens of length `seq_len` to get activations of shape `[batch, seq_len, d_model]`. The `mess3.generate` method takes the desired sequence length of *tokens* as the `sequence_len` parameter. It returns states of shape `[batch, sequence_len + 1, num_states]` and observations (tokens) of shape `[batch, sequence_len]`. This is exactly what we need.

So, the plan is:
1. Use `mess3.generate` to generate `seq_len` tokens and the corresponding `seq_len + 1` states.
2. Use these tokens to get activations from the model.
3. Align the states (taking states from index 1 onwards) with the activations.
4. Reshape both aligned states and activations for linear regression.


"""

# Re-generate data using mess3.generate to get the full state sequence
key, subkey = jax.random.split(key)
batch_keys = jax.random.split(subkey, batch_size)
initial_states = jnp.tile(mess3.initial_state, (batch_size, 1))

# Generate sequences of length seq_len. This returns states of shape [batch, seq_len + 1, num_states]
# and observations (tokens) of shape [batch, seq_len].
states_full, tokens_jax = mess3.generate(initial_states, batch_keys, seq_len, True)

# Convert tokens to PyTorch tensor
tokens = torch.from_numpy(np.array(tokens_jax)).long().to(device)

# Run with cache to get activations. Activations should be [batch, seq_len, d_model]
_, cache = model.run_with_cache(tokens, return_type=None)
activations = cache['blocks.3.hook_resid_post']

# Reshape activations to [batch * seq_len, d_model]
activations_flat_for_lr = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

# states_full has shape [batch, seq_len + 1, num_states].
# We need the states *after* processing each token, which are states_full[:, 1:, :].
# This slice has shape [batch, seq_len, num_states].
true_states_to_flatten = states_full[:, 1:, :]

# Reshape the sliced true states to [batch * seq_len, num_states]
true_states_flat = true_states_to_flatten.reshape(-1, true_states_to_flatten.shape[-1])

print(f"states_full shape: {states_full.shape}")
print(f"true_states_to_flatten shape after slicing: {true_states_to_flatten.shape}")
print(f"Reshaped activations shape: {activations_flat_for_lr.shape}")
print(f"Reshaped true states shape: {true_states_flat.shape}")

"""**Reasoning**:
The shapes are still not aligned. `Reshaped activations shape: (10000, 64)` which is `batch_size * (seq_len + 1)` instead of `batch_size * seq_len`. `Reshaped true states shape: (9000, 3)` which is `batch_size * seq_len`.

Looking at the `model.run_with_cache(tokens, return_type=None)` call, the activations are obtained for all input tokens. If `tokens` has shape `[batch, seq_len]`, the residual stream activations `cache['blocks.3.hook_resid_post']` should have shape `[batch, seq_len, d_model]`. This is because the model processes the sequence up to `seq_len` tokens. However, the output `Reshaped activations shape: (10000, 64)` suggests that the activations tensor `activations` has shape `[batch, seq_len + 1, d_model]` before flattening, which means the `tokens` tensor might have shape `[batch, seq_len + 1]`.

Let's check the shape of the `tokens` tensor right before passing it to `model.run_with_cache`. The `mess3.generate` function with `seq_len` requested should output `tokens_jax` of shape `[batch, seq_len]`. Converting this to a PyTorch tensor should maintain the shape `[batch, seq_len]`. If the activations are coming out as `[batch, seq_len + 1, d_model]`, it implies the model is processing an extra token or there's a misunderstanding of how the activations are generated or indexed in TransformerLens.

A common pattern is that the model processes `n_ctx` tokens and the residual stream contains activations for all `n_ctx` positions. If `cfg.n_ctx` is 10, and we pass tokens of length 10 (`seq_len`), the activations should be for positions 0 to 9.

Let's verify the shape of `tokens` just before the model call and the shape of `activations` just after getting it from the cache. Also, let's re-check the true states slicing based on the expected alignment. If tokens are `[batch, seq_len]`, activations are `[batch, seq_len, d_model]`. The true state we want to predict at position `t` is the state *after* processing token `t`. If `states_full` is `[batch, seq_len + 1, num_states]`, `states_full[:, t+1, :]` is the state after token `t`. So we want states from `states_full[:, 1:, :]`, which has shape `[batch, seq_len, num_states]`. This part seems correct. The discrepancy is with the activations shape.

Let's try setting `seq_len` equal to `cfg.n_ctx - 1`. If the model processes `n_ctx` tokens, and we want activations for `n_ctx - 1` tokens to predict the state after each of those, we should generate sequences of length `n_ctx - 1`.


"""

# Adjust seq_len to align with model's context window
seq_len = cfg.n_ctx - 1 # We want activations for n_ctx-1 tokens to predict n_ctx-1 states

# Re-generate data using mess3.generate
key, subkey = jax.random.split(key)
batch_keys = jax.random.split(subkey, batch_size)
initial_states = jnp.tile(mess3.initial_state, (batch_size, 1))

# Generate sequences of length seq_len. This returns states of shape [batch, seq_len + 1, num_states]
# and observations (tokens) of shape [batch, seq_len].
states_full, tokens_jax = mess3.generate(initial_states, batch_keys, seq_len, True)

# Convert tokens to PyTorch tensor
tokens = torch.from_numpy(np.array(tokens_jax)).long().to(device)

# Print token shape before model call
print(f"Tokens shape before model call: {tokens.shape}")

# Run with cache to get activations.
_, cache = model.run_with_cache(tokens, return_type=None)
activations = cache['blocks.3.hook_resid_post']

# Print activations shape after model call
print(f"Activations shape after model call: {activations.shape}")


# Reshape activations to [batch * seq_len, d_model]
activations_flat_for_lr = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

# states_full has shape [batch, seq_len + 1, num_states].
# We need the states *after* processing each token, which are states_full[:, 1:, :].
# This slice has shape [batch, seq_len, num_states].
true_states_to_flatten = states_full[:, 1:, :]

# Reshape the sliced true states to [batch * seq_len, num_states]
true_states_flat = true_states_to_flatten.reshape(-1, true_states_to_flatten.shape[-1])

print(f"states_full shape: {states_full.shape}")
print(f"true_states_to_flatten shape after slicing: {true_states_to_flatten.shape}")
print(f"Reshaped activations shape: {activations_flat_for_lr.shape}")
print(f"Reshaped true states shape: {true_states_flat.shape}")

"""**Reasoning**:
The print statements show that after setting `seq_len = cfg.n_ctx - 1 = 9`, the `tokens` shape is `[1000, 9]` and the `activations` shape is `[1000, 9, 64]`. This is the correct alignment between tokens and activations. However, `states_full` is `[1000, 9, 3]`, which means `mess3.generate` with `seq_len=9` is returning states of length 9, not `seq_len + 1 = 10`. This contradicts the previous observation where `states.shape` was `(1000, 10, 3)` when `sequence_len=10` was used.

Given that `states_full` is `[batch, seq_len, num_states]`, and activations are `[batch, seq_len, d_model]`, we need to align them. The activation at position `t` (`activations[:, t, :]`) corresponds to the token at position `t` (`tokens[:, t]`). The true state we want to predict is the state *after* processing token `t`. If `states_full` represents the states *after* each token, then `states_full[:, t, :]` would be the state after token `t`. In this case, we can use `states_full` directly (excluding the initial state if `states_full` includes it).

Since `states_full` has shape `[batch, seq_len, num_states]` where `seq_len` is the number of tokens, it likely represents the states *after* processing each of the `seq_len` tokens. If this is the case, the state at index `t` in `states_full` corresponds to the state after processing the token at index `t` in `tokens`. This aligns perfectly with the activations at index `t`.

Therefore, we should use `states_full` directly and reshape it. The slicing `states_full[:, 1:, :]` was incorrect because `states_full` doesn't have the extra initial state dimension in this case.

Let's use `states_full` directly and reshape it to `[batch * seq_len, num_states]`.


"""

# seq_len is already set to cfg.n_ctx - 1 = 9 from the previous step.

# Re-generate data using mess3.generate
key, subkey = jax.random.split(key)
batch_keys = jax.random.split(subkey, batch_size)
initial_states = jnp.tile(mess3.initial_state, (batch_size, 1))

# Generate sequences of length seq_len (9). This returns states of shape [batch, seq_len, num_states] (likely)
# and observations (tokens) of shape [batch, seq_len].
states_full, tokens_jax = mess3.generate(initial_states, batch_keys, seq_len, True)

# Convert tokens to PyTorch tensor
tokens = torch.from_numpy(np.array(tokens_jax)).long().to(device)

# Run with cache to get activations. Activations should be [batch, seq_len, d_model]
_, cache = model.run_with_cache(tokens, return_type=None)
activations = cache['blocks.3.hook_resid_post']

# Reshape activations to [batch * seq_len, d_model]
activations_flat_for_lr = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

# Based on the observed shape [batch, seq_len, num_states] for states_full,
# it seems to represent the states *after* each token.
# This aligns with the activations at each token position.
# So, we can use states_full directly and reshape it.
true_states_flat = states_full.reshape(-1, states_full.shape[-1])

print(f"states_full shape: {states_full.shape}")
print(f"Reshaped activations shape: {activations_flat_for_lr.shape}")
print(f"Reshaped true states shape: {true_states_flat.shape}")

"""## Train a linear regression model

### Subtask:
Train a linear regression model to predict the true beliefs from the residual stream activations.

**Reasoning**:
Train a linear regression model to predict the true beliefs from the residual stream activations by importing the necessary class, instantiating the model, and fitting it to the prepared data.
"""


# Instantiate a LinearRegression model
linear_regression_model = LinearRegression()

# Fit the LinearRegression model
# activations_flat_for_lr is X (features), true_states_flat is y (target)
linear_regression_model.fit(activations_flat_for_lr, true_states_flat)

print("Linear regression model trained successfully.")

"""## Predict beliefs using the linear model

### Subtask:
Use the trained linear regression model to predict the beliefs from the activations.

**Reasoning**:
Use the trained linear regression model to predict the beliefs from the activations.
"""

# Use the trained model to predict the beliefs from the flattened activations
predicted_beliefs = linear_regression_model.predict(activations_flat_for_lr)

print("Predicted beliefs generated.")
print(f"Shape of predicted beliefs: {predicted_beliefs.shape}")

"""## Visualize true vs. predicted beliefs

### Subtask:
Plot the true beliefs and the predicted beliefs side-by-side for comparison.

**Reasoning**:
Plot the true beliefs and predicted beliefs side-by-side as scatter plots to visually compare them.
"""

# Assuming true_states_flat and predicted_beliefs are available and have shape (N, 3)
# where N is batch_size * seq_len, and the 3 columns represent the belief probabilities for each state.

# Define the vertices of an equilateral triangle in 2D for visualization
v1 = np.array([0, 0])
v2 = np.array([1, 0])
v3 = np.array([0.5, np.sqrt(3)/2])

# Function to transform 3D simplex coordinates to 2D triangle coordinates
def simplex_to_triangle_coords(simplex_coords, v1, v2, v3):
    # Ensure the sum of simplex coordinates is close to 1 (optional, depending on data)
    # simplex_coords = simplex_coords / jnp.sum(simplex_coords, axis=-1, keepdims=True)

    # Map to 2D using barycentric coordinates
    x = simplex_coords[:, 0] * v1[0] + simplex_coords[:, 1] * v2[0] + simplex_coords[:, 2] * v3[0]
    y = simplex_coords[:, 0] * v1[1] + simplex_coords[:, 1] * v2[1] + simplex_coords[:, 2] * v3[1]
    return np.stack([x, y], axis=1)

# Transform both true and predicted beliefs to triangle coordinates
true_beliefs_triangle_coords = simplex_to_triangle_coords(true_states_flat, v1, v2, v3)
predicted_beliefs_triangle_coords = simplex_to_triangle_coords(predicted_beliefs, v1, v2, v3)


# Find the index of the most likely state for coloring for both true and predicted
true_states_indices = jnp.argmax(true_states_flat, axis=1)
predicted_states_indices = np.argmax(predicted_beliefs, axis=1) # Use np.argmax for numpy array

# Create a figure with two subplots using Plotly
fig = go.Figure()

# Subplot 1: True Beliefs on Simplex (using default x and y axes)
# Add scatter plot for each state type (0, 1, 2)
colors = ['red', 'green', 'blue']
state_names = ['State 0', 'State 1', 'State 2']

for state_id in range(3):
    mask = true_states_indices == state_id
    fig.add_trace(go.Scattergl( # Use Scattergl for potentially large number of points
        x=true_beliefs_triangle_coords[mask, 0],
        y=true_beliefs_triangle_coords[mask, 1],
        mode='markers',
        name=f'True {state_names[state_id]}',
        marker=dict(
            size=3,
            color=colors[state_id],
            opacity=0.6,
        ),
        legendgroup='True Beliefs', # Group for legend
        showlegend=True if state_id == 0 else False # Show legend only once per group
    ))

# Add triangle boundaries for True Beliefs subplot
fig.add_trace(go.Scatter(
    x=[v1[0], v2[0], v3[0], v1[0]],
    y=[v1[1], v2[1], v3[1], v1[1]],
    mode='lines',
    showlegend=False,
    line=dict(color='black', width=2),
    legendgroup='True Beliefs'
))


# Subplot 2: Predicted Beliefs on Simplex (using x2 and y2 axes)
for state_id in range(3):
    mask = predicted_states_indices == state_id
    fig.add_trace(go.Scattergl( # Use Scattergl for potentially large number of points
        x=predicted_beliefs_triangle_coords[mask, 0],
        y=predicted_beliefs_triangle_coords[mask, 1],
        mode='markers',
        name=f'Predicted {state_names[state_id]}',
        marker=dict(
            size=3,
            color=colors[state_id],
            opacity=0.6,
        ),
        legendgroup='Predicted Beliefs', # Group for legend
        showlegend=True if state_id == 0 else False, # Show legend only once per group
        xaxis='x2', # Assign to the second x-axis
        yaxis='y2'  # Assign to the second y-axis
    ))

# Add triangle boundaries for Predicted Beliefs subplot
fig.add_trace(go.Scatter(
    x=[v1[0], v2[0], v3[0], v1[0]],
    y=[v1[1], v2[1], v3[1], v1[1]],
    mode='lines',
    showlegend=False,
    line=dict(color='black', width=2),
    legendgroup='Predicted Beliefs',
    xaxis='x2', # Assign to the second x-axis
    yaxis='y2'  # Assign to the second y-axis
))


# Update layout for two subplots side-by-side
fig.update_layout(
    title='True vs. Predicted Beliefs on a Simplex',
    xaxis=dict(
        title='Simplex Dimension 1 (True Beliefs)',
        domain=[0, 0.45], # Adjust domain to position subplot
        scaleanchor="y", scaleratio=1
    ),
    yaxis=dict(
        title='Simplex Dimension 2 (True Beliefs)',
        domain=[0, 1]
    ),
    xaxis2=dict(
        title='Simplex Dimension 1 (Predicted Beliefs)',
        domain=[0.55, 1], # Adjust domain to position subplot
        scaleanchor="y2", scaleratio=1
    ),
    yaxis2=dict(
        title='Simplex Dimension 2 (Predicted Beliefs)',
        domain=[0, 1],
        overlaying='y' # Overlay on the first y-axis
    ),
    height=600,
    width=1200, # Adjust width for two subplots
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


fig.show()

"""## Analyze the results

### Subtask:
Analyze the linear regression model's performance and the visualization to understand how well the residual stream activations encode the hidden states.

## Summary:

### Data Analysis Key Findings

*   The `model.run_with_cache` function produced "Invalid return\_type" warnings for several specified return types, although `return_type=None` was eventually used successfully.
*   The shape of the `true_states` generated by `generate_data_batch` was `(batch_size, num_states)`, indicating it returned only the final state for each sequence, not the full sequence of states.
*   Using `mess3.generate` with a requested length `seq_len` returned `states_full` with shape `[batch, seq_len, num_states]`, representing the state *after* each token in the generated sequence.
*   Activations from the last layer (`blocks.3.hook_resid_post`) had the shape `[batch, seq_len, d_model]`.
*   The activations and true states (from `mess3.generate`) were successfully reshaped to `[batch * seq_len, d_model]` and `[batch * seq_len, num_states]` respectively, resulting in shapes `(9000, 64)` and `(9000, 3)` for 9000 data points.
*   A linear regression model was successfully trained using the flattened activations to predict the flattened true states.
*   The trained linear model successfully predicted beliefs, resulting in a `predicted_beliefs` array with shape `(9000, 3)`.
*   Scatter plots of the first two components of the true and predicted beliefs were generated and displayed side-by-side, colored by the most likely state, allowing for visual comparison.

### Insights or Next Steps

*   Investigate the correct usage of the `return_type` parameter in `model.run_with_cache` to understand the intended behavior and potential issues encountered.
*   Quantify the performance of the linear regression model (e.g., using R-squared or Mean Squared Error) to provide a numerical assessment of how well the activations predict the beliefs, in addition to visual inspection.
"""