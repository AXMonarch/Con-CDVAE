"""
sae_model.py
------------
Top-K Sparse Autoencoder (SAE) for probing Con-CDVAE GNN activations.

Architecture (2025/2026 SOTA):
  - Encoder:  h = ReLU(W_enc @ x + b_enc)    shape [batch, dict_size]
  - Top-K:    keep only the K largest activations, zero the rest
  - Decoder:  x̂ = W_dec @ h_sparse + b_dec   shape [batch, input_dim]

Key design choices:
  - Top-K sparsity is enforced by construction (no L1 tuning needed)
  - Decoder weights are L2-normalised after every gradient step
    (prevents the trivial solution of shrinking W_dec to make W_enc large)
  - "Dead feature" tracking: if a neuron fires < once per 10k tokens
    across a batch we consider it dead and can optionally re-initialise it
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class SAEConfig:
    input_dim: int = 256          # GNN hidden dimension (DimeNet++ default in CDVAE)
    expansion_factor: int = 16    # dict_size = expansion_factor * input_dim
    k: int = 32                   # how many features fire per token
    lr: float = 2e-4
    batch_size: int = 4096        # atom-level tokens
    num_epochs: int = 20
    normalise_decoder: bool = True
    # Logging / checkpointing
    log_every: int = 200          # steps
    save_every: int = 2000        # steps
    checkpoint_dir: str = "sae_checkpoints"

    @property
    def dict_size(self) -> int:
        return self.expansion_factor * self.input_dim


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder.

    Parameters
    ----------
    cfg : SAEConfig
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.input_dim
        d_sae = cfg.dict_size

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder  (columns are the "feature directions" in activation space)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Initialise
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W_dec, nonlinearity="relu")

        # Dead-feature counter: tracks tokens since last fire
        self.register_buffer(
            "steps_since_last_fire", torch.zeros(d_sae, dtype=torch.long)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [batch, d_in]
        returns: h_sparse [batch, d_sae]
        """
        pre_acts = F.relu(x @ self.W_enc + self.b_enc)  # [B, d_sae]
        h_sparse = self._top_k(pre_acts)
        return h_sparse

    def decode(self, h_sparse: torch.Tensor) -> torch.Tensor:
        """h_sparse : [batch, d_sae]  →  x̂ [batch, d_in]"""
        return h_sparse @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        x_hat  : reconstructed input
        h_sparse : sparse feature activations
        loss_dict: dict with 'recon_loss', 'l0_loss', 'total_loss'
        """
        h_sparse = self.encode(x)
        x_hat = self.decode(h_sparse)

        recon_loss = F.mse_loss(x_hat, x)
        # L0 as a diagnostic (not used for gradient)
        l0 = (h_sparse > 0).float().sum(dim=-1).mean()

        loss_dict = {
            "recon_loss": recon_loss,
            "l0": l0.detach(),
            "total_loss": recon_loss,   # Top-K has no L1 penalty
        }

        # Update dead-feature tracker
        with torch.no_grad():
            fired = (h_sparse > 0).any(dim=0)  # [d_sae]
            self.steps_since_last_fire += 1
            self.steps_since_last_fire[fired] = 0

        return x_hat, h_sparse, loss_dict

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _top_k(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Zero out all activations except the top-K per token."""
        k = self.cfg.k
        # topk over the SAE dimension
        topk_vals, topk_idx = pre_acts.topk(k, dim=-1)       # [B, K]
        mask = torch.zeros_like(pre_acts)
        mask.scatter_(-1, topk_idx, 1.0)
        return pre_acts * mask

    @torch.no_grad()
    def normalise_decoder_(self):
        """
        Project each decoder column (feature direction) onto the unit sphere.
        Call after every optimiser step.
        Prevents the model from hiding information in W_dec norms.
        """
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def dead_features(self, threshold: int = 10_000) -> torch.Tensor:
        """Return indices of features that haven't fired in `threshold` steps."""
        return (self.steps_since_last_fire > threshold).nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def reinit_dead_features_(self, activations: torch.Tensor, threshold: int = 10_000):
        """
        Heuristic re-initialisation for dead features.
        Reinitialises their encoder/decoder rows to random directions sampled
        from the residual stream data distribution.
        """
        dead = self.dead_features(threshold)
        if len(dead) == 0:
            return 0

        # sample random activation vectors as new directions
        idx = torch.randperm(activations.shape[0], device=activations.device)[: len(dead)]
        new_dirs = activations[idx]  # [n_dead, d_in]
        new_dirs = F.normalize(new_dirs, dim=-1)

        self.W_dec.data[dead] = new_dirs
        self.W_enc.data[:, dead] = new_dirs.T
        self.b_enc.data[dead] = 0.0
        self.steps_since_last_fire[dead] = 0

        return len(dead)
