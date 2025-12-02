import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from BatchTopK.sae import BaseAutoencoder


class BandedCovarianceSAE(BaseAutoencoder):
    """Sparse autoencoder with configurable sparsity penalties and banded covariance loss."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.lambda_sparse = float(cfg.get("lambda_sparse", 0.01))
        self.lambda_ar = float(cfg.get("lambda_ar", 0.01))
        self.sparsity_mode = cfg.get("sparsity_mode", "l0")
        self.delta = float(cfg.get("delta", 0.5))
        self.epsilon = float(cfg.get("epsilon", 1e-6))
        self.p = int(cfg.get("p", 3))
        self.beta_slope = float(cfg.get("beta_slope", 2.0))
        self.use_beta = bool(cfg.get("use_beta", True))
        self.use_alpha = bool(cfg.get("use_alpha", True))

        if self.use_beta and self.p > 0:
            self.beta = torch.nn.Parameter(torch.ones(self.p, device=self.W_enc.device, dtype=self.W_enc.dtype))
        else:
            beta_init = torch.ones(self.p, device=self.W_enc.device, dtype=self.W_enc.dtype)
            self.register_buffer("beta", beta_init)

        if self.use_alpha:
            self.alpha = torch.nn.Parameter(torch.zeros(self.cfg["dict_size"], device=self.W_enc.device, dtype=self.W_enc.dtype))
        else:
            self.register_buffer("alpha", torch.zeros(self.cfg["dict_size"], device=self.W_enc.device, dtype=self.W_enc.dtype))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec

        self.update_inactive_features(acts)

        recon_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        sparsity_loss, sparsity_stats = self._compute_sparsity_loss(acts)
        ar_loss = self._compute_ar_loss(acts)

        total_loss = recon_loss + self.lambda_sparse * sparsity_loss + self.lambda_ar * ar_loss
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum(),
            "loss": total_loss,
            "l2_loss": recon_loss,
            "l1_loss": self.lambda_sparse * sparsity_loss,
            "l1_norm": sparsity_stats["base_penalty"],
            "l0_norm": (acts > 0).float().sum(-1).mean(),
            "aux_loss": self.lambda_ar * ar_loss,
            "raw_sparsity_loss": sparsity_loss,
            "raw_ar_loss": ar_loss,
        }
        return output

    def _compute_sparsity_loss(self, acts: torch.Tensor) -> (torch.Tensor, Dict[str, torch.Tensor]):
        if self.lambda_sparse == 0:
            zero = torch.tensor(0.0, device=acts.device, dtype=acts.dtype)
            return zero, {"base_penalty": zero}

        if self.sparsity_mode.lower() == "l1":
            base_penalty = acts.abs().sum(-1).mean()
        else:
            scale = 2.0 / self.delta * math.log((1.0 - self.epsilon) / self.epsilon)
            shifted = acts - (self.delta / 2.0)
            penalty = torch.sigmoid(scale * shifted)
            base_penalty = penalty.sum(-1).mean()
        return base_penalty, {"base_penalty": base_penalty}

    def _compute_ar_loss(self, acts: torch.Tensor) -> torch.Tensor:
        if self.lambda_ar == 0 or self.p <= 0:
            return torch.tensor(0.0, device=acts.device, dtype=acts.dtype)

        pred = torch.zeros_like(acts)
        if self.beta_slope == 0:
            slope_denom = torch.tensor(1.0, device=acts.device, dtype=acts.dtype)
        else:
            slope_denom = torch.tensor(self.beta_slope, device=acts.device, dtype=acts.dtype)
        alpha_term = torch.tanh(self.alpha / slope_denom)

        for k in range(1, self.p + 1):
            shifted = torch.zeros_like(acts)
            shifted[:, k:] = acts[:, :-k]
            coeff = torch.pow(alpha_term, k)
            coeff = coeff * self.beta[k - 1]
            pred = pred + coeff * shifted

        diff = acts - pred
        return diff.pow(2).mean()
