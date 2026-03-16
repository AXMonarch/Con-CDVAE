import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sae_scorer import importance_scores, select_top_dims


# ── SAE Architecture ─────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    Single-layer sparse autoencoder.
    Learns an overcomplete sparse representation of a dense activation vector.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h     = F.relu(self.encoder(x))   # sparse feature activations
        x_hat = self.decoder(h)            # reconstruction
        return x_hat, h


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    h: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Reconstruction loss + L1 sparsity penalty.
    lam controls sparsity — higher = fewer active features.
    """
    recon_loss    = F.mse_loss(x_hat, x)
    sparsity_loss = lam * h.abs().mean()
    return recon_loss + sparsity_loss


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_sae(
    activations: torch.Tensor,
    input_dim: int,
    hidden_dim: int,
    lam: float = 1e-3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    val_split: float = 0.1,
    device: str = 'cpu',
    verbose: bool = True,
) -> SparseAutoencoder:
    """
    Train a single SAE on an activation tensor.

    Args:
        activations: (N, input_dim) tensor
        input_dim:   dimension of input activations (post-filtering)
        hidden_dim:  number of sparse features to learn
        lam:         L1 sparsity penalty weight
        lr:          learning rate
        epochs:      training epochs
        batch_size:  minibatch size
        val_split:   fraction of data held out for validation
        device:      'cuda' or 'cpu'
        verbose:     print loss per epoch

    Returns:
        trained SparseAutoencoder (best val loss checkpoint)
    """
    dataset  = TensorDataset(activations)
    val_size = int(len(dataset) * val_split)
    trn_size = len(dataset) - val_size
    trn_ds, val_ds = random_split(dataset, [trn_size, val_size])

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    sae       = SparseAutoencoder(input_dim, hidden_dim).to(device)
    optimiser = torch.optim.Adam(sae.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, epochs + 1):
        # train
        sae.train()
        trn_loss = 0.0
        for (x,) in trn_loader:
            x = x.to(device)
            x_hat, h = sae(x)
            loss = sae_loss(x, x_hat, h, lam)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            trn_loss += loss.item() * len(x)
        trn_loss /= trn_size

        # validate
        sae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                x_hat, h = sae(x)
                val_loss += sae_loss(x, x_hat, h, lam).item() * len(x)
        val_loss /= val_size

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in sae.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d}/{epochs}  trn: {trn_loss:.6f}  val: {val_loss:.6f}")

    sae.load_state_dict(best_state)
    return sae