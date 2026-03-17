# train_sae.py — Training loop for the Top-K Sparse Autoencoder
#
# Trains an SAE on pre-extracted activations from a specific hook point
# (typically z_con). Reuses the activations saved by the probe pipeline.
#
# Training details:
#   - Optimizer: Adam with weight decay
#   - Decoder columns are re-normalised to unit norm after each step
#   - Dead features are tracked and revived via an auxiliary loss
#   - Logs reconstruction MSE, dead feature count, and % variance explained

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from .sae_model import TopKSAE, SAEConfig


class SAETrainer:
    """
    Trains a Top-K SAE on pre-extracted activations.

    Parameters
    ----------
    config : SAEConfig
        SAE hyperparameters.
    lr : float
        Learning rate.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        config: SAEConfig,
        lr: float = 2e-4,
        num_epochs: int = 20,
        batch_size: int = 4096,
        device: str = "cpu",
    ):
        self.config = config
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.sae = TopKSAE(config).to(self.device)

    def train(
        self,
        X_train: torch.Tensor,
        X_val: torch.Tensor,
    ) -> dict[str, list[float]]:
        """
        Train the SAE and return training history.

        Parameters
        ----------
        X_train : Tensor (N_train, input_dim)
        X_val : Tensor (N_val, input_dim)

        Returns
        -------
        history : dict with keys "train_loss", "val_loss", "n_dead", "var_explained"
        """
        train_ds = TensorDataset(X_train)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
        )

        optimiser = torch.optim.Adam(
            self.sae.parameters(), lr=self.lr, weight_decay=1e-5,
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "n_dead": [],
            "var_explained": [],
        }

        for epoch in range(self.num_epochs):
            # -- Training ---------------------------------------------------------
            self.sae.train()
            epoch_loss = 0.0
            n_batches = 0

            for (x_batch,) in train_loader:
                x_batch = x_batch.to(self.device)

                x_hat, h_sparse, h_pre = self.sae(x_batch)

                # Reconstruction loss
                recon_loss = F.mse_loss(x_hat, x_batch)

                # Auxiliary loss for dead features
                aux_loss = self.sae.auxiliary_loss(x_batch, h_pre)
                loss = recon_loss + self.config.aux_loss_coeff * aux_loss

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Post-step: normalise decoder and update dead tracker
                self.sae.normalise_decoder()
                self.sae.update_dead_tracker(h_sparse)

                epoch_loss += recon_loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            # -- Validation -------------------------------------------------------
            val_metrics = self.evaluate(X_val)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_metrics["mse"])
            history["n_dead"].append(self.sae.n_dead_features())
            history["var_explained"].append(val_metrics["var_explained"])

            print(
                f"  Epoch {epoch+1:3d}/{self.num_epochs}  "
                f"train_MSE={avg_train_loss:.6f}  "
                f"val_MSE={val_metrics['mse']:.6f}  "
                f"var_expl={val_metrics['var_explained']:.4f}  "
                f"dead={self.sae.n_dead_features()}"
            )

        return history

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor) -> dict[str, float]:
        """
        Evaluate reconstruction quality on a dataset.

        Returns dict with "mse" and "var_explained".
        """
        self.sae.eval()

        preds = []
        for i in range(0, len(X), self.batch_size):
            x_batch = X[i : i + self.batch_size].to(self.device)
            x_hat, _, _ = self.sae(x_batch)
            preds.append(x_hat.cpu())

        x_hat_all = torch.cat(preds, dim=0)
        mse = F.mse_loss(x_hat_all, X).item()

        # Variance explained: 1 - Var(residual) / Var(input)
        residual = X - x_hat_all
        var_explained = 1.0 - residual.var().item() / X.var().item()

        return {"mse": mse, "var_explained": var_explained}
