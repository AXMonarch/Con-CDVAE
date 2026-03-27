# train_sae.py — Training loop for the Normalized Top-K Sparse Autoencoder
#
# Trains an SAE on pre-extracted activations from a specific hook point
# (typically z_con). Reuses the activations saved by the probe pipeline.
#
# Training details:
#   - Input normalization: per-dimension mean/std computed from training set,
#     stored as frozen buffers on the SAE model
#   - Loss is computed in normalized space so all dimensions contribute equally
#   - Optimizer: Adam with weight decay
#   - Decoder columns are re-normalised to unit norm after each step
#   - Dead features are tracked and revived via an auxiliary loss
#   - Logs reconstruction MSE (normalized), dead feature count, and
#     % variance explained (in both normalized and original space)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .sae_model import TopKSAE, SAEConfig


class SAETrainer:
    """
    Trains a Normalized Top-K SAE on pre-extracted activations.

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
        early_stop_patience: int | None = None,
    ):
        self.config = config
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.early_stop_patience = early_stop_patience

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
        X_train : Tensor (N_train, input_dim) — raw activations
        X_val : Tensor (N_val, input_dim) — raw activations

        Returns
        -------
        history : dict with keys "train_loss", "val_loss", "n_dead",
                  "var_explained", "var_explained_raw"
        """
        # Compute and freeze normalization stats from training data
        self.sae.set_norm_stats(X_train)
        print(f"  Input normalization: μ range [{self.sae.input_mean.min():.3f}, "
              f"{self.sae.input_mean.max():.3f}], "
              f"σ range [{self.sae.input_std.min():.3f}, "
              f"{self.sae.input_std.max():.3f}]")

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
            "var_explained_raw": [],
        }

        best_val_loss = None
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            self.sae.train()
            epoch_loss = 0.0
            n_batches = 0

            for (x_batch,) in train_loader:
                x_batch = x_batch.to(self.device)

                # forward() returns x_hat in normalized space
                x_hat_norm, h_sparse, h_pre = self.sae(x_batch)

                # Reconstruction loss in normalized space
                x_norm = self.sae.normalize(x_batch)
                recon_loss = F.mse_loss(x_hat_norm, x_norm)

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

            val_metrics = self.evaluate(X_val)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_metrics["mse"])
            history["n_dead"].append(self.sae.n_dead_features())
            history["var_explained"].append(val_metrics["var_explained"])
            history["var_explained_raw"].append(val_metrics["var_explained_raw"])

            print(
                f"  Epoch {epoch+1:3d}/{self.num_epochs}  "
                f"train_MSE={avg_train_loss:.6f}  "
                f"val_MSE={val_metrics['mse']:.6f}  "
                f"var_expl={val_metrics['var_explained']:.4f}  "
                f"var_expl_raw={val_metrics['var_explained_raw']:.4f}  "
                f"dead={self.sae.n_dead_features()}"
            )

            if best_val_loss is None or val_metrics["mse"] < best_val_loss:
                best_val_loss = val_metrics["mse"]
                best_state = {k: v.clone() for k, v in self.sae.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (self.early_stop_patience is not None
                    and epochs_without_improvement >= self.early_stop_patience):
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {self.early_stop_patience} epochs)")
                break

        # Restore best model if early stopping was enabled
        if self.early_stop_patience is not None and best_state is not None:
            self.sae.load_state_dict(best_state)

        return history

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor) -> dict[str, float]:
        """
        Evaluate reconstruction quality on a dataset.

        Returns dict with:
            "mse"               — MSE in normalized space (matches training loss)
            "var_explained"     — VE in normalized space
            "var_explained_raw" — VE in original (raw) space
        """
        self.sae.eval()

        norm_preds = []
        raw_preds = []
        for i in range(0, len(X), self.batch_size):
            x_batch = X[i : i + self.batch_size].to(self.device)
            x_hat_norm, _, _ = self.sae(x_batch)
            norm_preds.append(x_hat_norm.cpu())
            raw_preds.append(self.sae.denormalize(x_hat_norm).cpu())

        x_hat_norm_all = torch.cat(norm_preds, dim=0)
        x_hat_raw_all = torch.cat(raw_preds, dim=0)

        # Normalized-space metrics (consistent with training loss)
        X_norm = self.sae.normalize(X)
        mse = F.mse_loss(x_hat_norm_all, X_norm).item()

        # R² = 1 - SS_res / SS_tot  (per-dimension means for SS_tot)
        ss_tot_norm = X_norm.var(dim=0, correction=0).mean().item()
        var_explained = 1.0 - mse / ss_tot_norm

        # Raw-space metrics (for interpretability / comparison)
        mse_raw = F.mse_loss(x_hat_raw_all, X).item()
        ss_tot_raw = X.var(dim=0, correction=0).mean().item()
        var_explained_raw = 1.0 - mse_raw / ss_tot_raw

        return {
            "mse": mse,
            "var_explained": var_explained,
            "var_explained_raw": var_explained_raw,
        }
