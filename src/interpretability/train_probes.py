# train_probes.py — Training loop and evaluation for probes
#
# Handles:
#   - Splitting activations + labels into train/val
#   - Training a probe (regression or classification)
#   - Evaluating with R² (regression) or accuracy (classification)
#   - Returning a results dict for one (hook_point, property) pair

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, accuracy_score

from .probes import create_probe
from .hooks import HOOK_POINTS


# ---- Metrics ----------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute R² and MSE for regression probes."""
    r2 = r2_score(y_true, y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"r2": r2, "mse": mse}


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy for classification probes."""
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc}


# ---- Per-atom -> per-crystal pooling ----------------------------------------

def pool_atom_activations(
    activations: torch.Tensor,
    num_atoms_list: list[int],
) -> torch.Tensor:
    """
    Mean-pool per-atom activations into per-crystal activations.

    Parameters
    ----------
    activations : Tensor of shape (total_atoms, dim)
    num_atoms_list : list of int, atoms per crystal

    Returns
    -------
    Tensor of shape (n_crystals, dim)
    """
    pooled = []
    offset = 0
    for n in num_atoms_list:
        crystal_acts = activations[offset : offset + n]
        pooled.append(crystal_acts.mean(dim=0))
        offset += n
    return torch.stack(pooled)


# ---- Single probe trainer ---------------------------------------------------

class ProbeTrainer:
    """
    Trains and evaluates a single probe for one (hook_point, property) pair.

    Parameters
    ----------
    task : str
        "regression" or "classification".
    input_dim : int
        Dimension of the activation vector.
    num_classes : int
        Number of classes for classification. Ignored for regression.
    probe_type : str
        "linear" or "mlp".
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
        task: str,
        input_dim: int,
        num_classes: int = 1,
        probe_type: str = "linear",
        lr: float = 1e-3,
        num_epochs: int = 100,
        batch_size: int = 512,
        device: str = "cpu",
    ):
        self.task = task
        self.probe_type = probe_type
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        output_dim = 1 if task == "regression" else num_classes
        self.probe = create_probe(probe_type, input_dim, output_dim)
        self.probe.to(self.device)

        if task == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> dict[str, float]:
        """
        Train the probe and return evaluation metrics on the val set.

        Parameters
        ----------
        X_train : Tensor (N_train, input_dim)
        y_train : Tensor (N_train,) — float for regression, long for classification
        X_val   : Tensor (N_val, input_dim)
        y_val   : Tensor (N_val,)

        Returns
        -------
        dict with metric names -> values (r2/mse or accuracy)
        """
        # -- Build data loaders -----------------------------------------------
        if self.task == "regression":
            y_train_t = y_train.float().unsqueeze(-1)
            y_val_t = y_val.float()
        else:
            y_train_t = y_train.long()
            y_val_t = y_val.long()

        train_ds = TensorDataset(X_train, y_train_t)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
        )

        # -- Training loop ----------------------------------------------------
        optimiser = torch.optim.Adam(self.probe.parameters(), lr=self.lr)
        self.probe.train()

        for epoch in range(self.num_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimiser.zero_grad()
                pred = self.probe(x_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                optimiser.step()

        # -- Evaluation -------------------------------------------------------
        return self.evaluate(X_val, y_val_t)

    @torch.no_grad()
    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Run the trained probe on X and compute metrics against y."""
        self.probe.eval()

        # Process in chunks to avoid OOM on large datasets
        preds = []
        for i in range(0, len(X), self.batch_size):
            x_batch = X[i : i + self.batch_size].to(self.device)
            pred = self.probe(x_batch).cpu()
            preds.append(pred)
        pred_all = torch.cat(preds, dim=0)

        if self.task == "regression":
            y_pred = pred_all.squeeze(-1).numpy()
            y_true = y.numpy() if isinstance(y, torch.Tensor) else y
            return compute_regression_metrics(y_true, y_pred)
        else:
            y_pred = pred_all.argmax(dim=-1).numpy()
            y_true = y.numpy() if isinstance(y, torch.Tensor) else y
            return compute_classification_metrics(y_true, y_pred)
