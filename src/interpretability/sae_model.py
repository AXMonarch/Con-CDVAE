# sae_model.py — Top-K Sparse Autoencoder for Con-CDVAE representations
#
# A Sparse Autoencoder (SAE) decomposes a dense representation (e.g. z_con)
# into a much wider, sparse set of features. The sparsity is enforced by
# keeping only the top-K activations per input, setting all others to zero.
#
# Architecture:
#   Input  x       (batch, input_dim)       e.g. (batch, 256)
#   Pre-bias        x_centered = x - b_dec  (subtract decoder bias)
#   Encode          h = W_enc @ x_centered + b_enc    (batch, n_features)
#   Top-K           keep only K largest values in h, zero the rest
#   Decode          x_hat = W_dec @ h_sparse + b_dec  (batch, input_dim)
#
# Loss:
#   L = MSE(x, x_hat)   +   aux_loss (dead feature revival)
#
# Reference: Gao et al. "Scaling and evaluating sparse autoencoders" (2024)
#

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Configuration for a Top-K Sparse Autoencoder."""

    input_dim: int = 256        # Dimension of the representation (z_con = 256)
    n_features: int = 4096      # Dictionary size (expansion factor × input_dim)
    k: int = 32                 # Number of active features per input
    dead_threshold: int = 200   # Steps without activation before a feature is "dead"
    aux_loss_coeff: float = 1/32  # Weight for auxiliary dead-feature loss


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder.

    Instead of using an L1 penalty to encourage sparsity (which requires
    tuning a coefficient), Top-K simply keeps only the K largest hidden
    activations and zeros out the rest. This gives exact, predictable
    sparsity: every input activates exactly K out of n_features features.

    Parameters
    ----------
    config : SAEConfig
        Hyperparameters for the SAE.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d = config.input_dim
        n = config.n_features

        # Encoder: project input_dim -> n_features
        self.W_enc = nn.Parameter(torch.empty(n, d))
        self.b_enc = nn.Parameter(torch.zeros(n))

        # Decoder: project n_features -> input_dim (no bias in weight, separate b_dec)
        self.W_dec = nn.Parameter(torch.empty(d, n))
        self.b_dec = nn.Parameter(torch.zeros(d))

        # Initialise weights
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        # Normalise decoder columns to unit norm (each feature direction is a unit vector)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

        # Track dead features: count steps since each feature was last active
        self.register_buffer(
            "steps_since_active", torch.zeros(n, dtype=torch.long),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to pre-TopK hidden activations.

        Parameters
        ----------
        x : Tensor (batch, input_dim)

        Returns
        -------
        h : Tensor (batch, n_features) — dense, before top-k selection
        """
        x_centered = x - self.b_dec
        return x_centered @ self.W_enc.T + self.b_enc

    def topk_mask(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply Top-K selection: keep only the K largest values per row.

        Parameters
        ----------
        h : Tensor (batch, n_features)

        Returns
        -------
        h_sparse : Tensor (batch, n_features) — same shape, but only K
                   entries per row are nonzero.
        """
        topk_vals, topk_idx = torch.topk(h, self.config.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return h * mask

    def decode(self, h_sparse: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to input space.

        Parameters
        ----------
        h_sparse : Tensor (batch, n_features)

        Returns
        -------
        x_hat : Tensor (batch, input_dim)
        """
        return h_sparse @ self.W_dec.T + self.b_dec

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> top-k -> decode.

        Parameters
        ----------
        x : Tensor (batch, input_dim)

        Returns
        -------
        x_hat    : Tensor (batch, input_dim) — reconstruction
        h_sparse : Tensor (batch, n_features) — sparse activations
        h_pre    : Tensor (batch, n_features) — pre-topk activations
        """
        h_pre = self.encode(x)
        h_sparse = self.topk_mask(h_pre)
        x_hat = self.decode(h_sparse)
        return x_hat, h_sparse, h_pre

    # -- Dead feature tracking and auxiliary loss --------------------------------

    @torch.no_grad()
    def update_dead_tracker(self, h_sparse: torch.Tensor) -> None:
        """
        Update the dead feature tracker after a training step.

        A feature is "active" if any sample in the batch activates it.
        Dead features get revived via the auxiliary loss.
        """
        active_mask = (h_sparse.abs() > 0).any(dim=0)  # (n_features,)
        self.steps_since_active[active_mask] = 0
        self.steps_since_active[~active_mask] += 1

    def dead_feature_mask(self) -> torch.Tensor:
        """Return a boolean mask of features that haven't fired recently."""
        return self.steps_since_active >= self.config.dead_threshold

    def auxiliary_loss(
        self, x: torch.Tensor, h_pre: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary loss to revive dead features.

        For dead features, compute what their reconstruction would be if
        we used THEM as the top-k instead. This gives gradients that push
        dead features towards useful directions.

        If no features are dead, returns 0.

        Parameters
        ----------
        x : Tensor (batch, input_dim) — original input
        h_pre : Tensor (batch, n_features) — pre-topk activations

        Returns
        -------
        loss : scalar Tensor
        """
        dead_mask = self.dead_feature_mask()
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return torch.tensor(0.0, device=x.device)

        # Take top-k among dead features only
        h_dead = h_pre[:, dead_mask]
        k_dead = min(self.config.k, n_dead)
        topk_vals, topk_idx = torch.topk(h_dead, k_dead, dim=-1)

        # Build sparse activation vector for dead features
        h_dead_sparse = torch.zeros_like(h_dead)
        h_dead_sparse.scatter_(-1, topk_idx, topk_vals)

        # Decode using only dead feature columns
        W_dec_dead = self.W_dec[:, dead_mask]  # (input_dim, n_dead)
        x_hat_dead = h_dead_sparse @ W_dec_dead.T + self.b_dec

        return F.mse_loss(x_hat_dead, x)

    # -- Utility ----------------------------------------------------------------

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        """Normalise decoder weight columns to unit norm (call after each step)."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def n_dead_features(self) -> int:
        """Return the current count of dead features."""
        return self.dead_feature_mask().sum().item()
