import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Sequence, Optional, List
from aanet_pipeline.cluster_summary import AAnetDescriptor
from AAnet.AAnet_torch.models.AAnet_vanilla import AAnet_vanilla
from aanet_pipeline.training import TrainingConfig

class StreamingAAnetTrainer:
    """
    Manages a collection of AAnet models and trains them in parallel (sequentially in loop, but logically parallel)
    on streaming data.
    """
    def __init__(
        self,
        *,
        descriptors: Sequence[AAnetDescriptor],
        config: TrainingConfig,
        device: torch.device,
        input_dim: int,
        sae_decoder_weight: torch.Tensor, # (d_sae, d_model)
    ):
        self.descriptors = descriptors
        self.config = config
        self.device = device
        self.input_dim = input_dim
        self.sae_decoder_weight = sae_decoder_weight.to(device)
        
        self.models: Dict[int, AAnet_vanilla] = {}
        self.optimizers: Dict[int, optim.Optimizer] = {}
        self.schedulers: Dict[int, object] = {}
        
        # Map cluster_id to latent indices tensor for fast slicing
        self.cluster_indices: Dict[int, torch.Tensor] = {}
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        print(f"Initializing {len(self.descriptors)} AAnet models on {self.device}...")
        for desc in self.descriptors:
            # Create model
            model = AAnet_vanilla(
                input_shape=self.input_dim,
                n_archetypes=self.config.k, # Note: k is fixed for now, or we need separate trainers for different k
                noise=self.config.noise,
                layer_widths=self.config.layer_widths,
                activation_out="tanh", # Standard for AAnet
                simplex_scale=self.config.simplex_scale,
                device=self.device
            )
            self.models[desc.cluster_id] = model
            
            # Optimizer
            self.optimizers[desc.cluster_id] = optim.Adam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
            
            # Scheduler
            self.schedulers[desc.cluster_id] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[desc.cluster_id],
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                verbose=False
            )
            
            # Indices
            if desc.latent_indices:
                self.cluster_indices[desc.cluster_id] = torch.tensor(desc.latent_indices, device=self.device, dtype=torch.long)
            else:
                 self.cluster_indices[desc.cluster_id] = torch.empty((0,), device=self.device, dtype=torch.long)

    def initialize_extrema(self, warmup_data: Dict[int, torch.Tensor]):
        """
        Initialize extrema for models using warmup data.
        warmup_data: dict mapping cluster_id to a tensor of data points.
        """
        from aanet_pipeline.extrema import compute_diffusion_extrema, ExtremaConfig
        
        print("Initializing extrema from warmup buffer...")
        for cid, data in warmup_data.items():
            if cid not in self.models:
                continue
            if data.shape[0] < self.config.k:
                continue
                
            try:
                # Compute extrema
                # We need to move data to CPU for graph construction usually, unless we have GPU graph lib
                # compute_diffusion_extrema expects numpy
                data_np = data.cpu().numpy()
                extrema = compute_diffusion_extrema(
                    data_np,
                    max_k=self.config.k,
                    config=ExtremaConfig(knn=150) # Use default or config
                )
                
                # Set model extrema
                # AAnet_vanilla expects diffusion_extrema as tensor
                if extrema is not None:
                    self.models[cid].set_archetypes(extrema.to(self.device))
            except Exception as e:
                print(f"Failed to init extrema for cluster {cid}: {e}")

    def train_step(self, feature_acts: torch.Tensor) -> Dict[int, float]:
        """
        Perform one training step for all models using the global feature activations.
        feature_acts: (batch_size, d_sae) - sparse activations
        Returns: dict of losses for updated clusters
        """
        losses = {}
        
        # We iterate over clusters. 
        # Ideally we could vectorize this further, but models are separate objects.
        # Python loop over 768 items is fast enough (ms).
        
        for cid, model in self.models.items():
            indices = self.cluster_indices[cid]
            if indices.numel() == 0:
                continue
                
            # 1. Slice activations for this cluster: (batch, n_cluster_latents)
            # We only care about rows where at least one latent in this cluster is active?
            # Or do we train on zeros too?
            # AAnet trains on the manifold of *reconstructions*.
            # If the reconstruction is zero, it's the origin.
            # Should we train on the origin?
            # Usually we only train on "active" data.
            # Let's filter for active samples.
            
            acts_c = feature_acts[:, indices] # (batch, n_cluster_latents)
            
            # Check which samples are active (sum of abs > 0)
            # Or use a threshold
            active_mask = (acts_c.abs().sum(dim=1) > self.config.active_threshold)
            
            if not active_mask.any():
                continue
                
            acts_c_active = acts_c[active_mask] # (n_active, n_cluster_latents)
            
            # 2. Compute Reconstruction: acts_c @ W_dec_c
            # W_dec_c: (n_cluster_latents, d_model)
            W_c = self.sae_decoder_weight[indices, :]
            
            # (n_active, n_latents) @ (n_latents, d_model) -> (n_active, d_model)
            X_recon = acts_c_active @ W_c
            
            # 3. Train Step
            optimizer = self.optimizers[cid]
            optimizer.zero_grad()
            
            # Forward
            # AAnet forward returns (recon, etc.)
            # loss_function returns (total_loss, metrics)
            recon, _, _ = model(X_recon)
            loss, metrics = model.loss_function(
                X_recon, 
                recon, 
                model.get_archetypes(), 
                gamma_reconstruction=self.config.gamma_reconstruction,
                gamma_archetypal=self.config.gamma_archetypal,
                gamma_extrema=self.config.gamma_extrema
            )
            
            loss.backward()
            
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
            optimizer.step()
            
            losses[cid] = {
                "loss": loss.item(),
                "reconstruction_loss": metrics["reconstruction_loss"],
                "archetypal_loss": metrics["archetypal_loss"],
                "extrema_loss": metrics.get("extrema_loss", 0.0)
            }
            
        return losses

    def save_models(self, output_dir: str):
        for cid, model in self.models.items():
            path = os.path.join(output_dir, f"aanet_cluster_{cid}.pt")
            torch.save(model.state_dict(), path)
