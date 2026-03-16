"""
utils.py
--------
Shared helpers for the SAE pipeline.
"""

import torch
import numpy as np
from pathlib import Path


def load_sae_and_norm(checkpoint_dir: str):
    """
    Load a trained SAE and its normalisation stats.
    Returns (sae, mean, std).
    """
    from sae_model import TopKSAE
    ckpt_dir = Path(checkpoint_dir)

    ckpt = torch.load(ckpt_dir / "sae_best.pt", map_location="cpu", weights_only=False)
    sae  = TopKSAE(ckpt["cfg"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()

    norm = torch.load(ckpt_dir / "normalisation.pt", weights_only=True)
    return sae, norm["mean"], norm["std"]


def activation_to_feature_space(
    gnn_acts: torch.Tensor,   # [N, d_in]
    sae,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Normalise raw GNN activations and run them through a frozen SAE encoder.
    Returns feature activations [N, d_sae].
    """
    acts_norm = (gnn_acts - mean) / std
    sae = sae.to(device)
    acts_norm = acts_norm.to(device)
    with torch.no_grad():
        return sae.encode(acts_norm).cpu()


def most_activating_crystals(
    feature_idx: int,
    feature_acts: torch.Tensor,  # [N_atoms, d_sae]
    crystal_ids:  torch.Tensor,  # [N_atoms]  — which crystal each atom belongs to
    top_n: int = 10,
) -> list[dict]:
    """
    For a given feature, return the `top_n` crystals where the feature fires
    most strongly (max pooled over atoms within that crystal).
    """
    feat = feature_acts[:, feature_idx]   # [N_atoms]
    n_crystals = int(crystal_ids.max().item()) + 1

    crystal_max = torch.zeros(n_crystals)
    for c in range(n_crystals):
        mask = crystal_ids == c
        if mask.any():
            crystal_max[c] = feat[mask].max()

    topk = crystal_max.topk(min(top_n, n_crystals))
    return [
        {"crystal_id": int(idx.item()), "max_activation": float(val.item())}
        for val, idx in zip(topk.values, topk.indices)
    ]


def explained_variance(
    gnn_acts: torch.Tensor,   # [N, d_in]
    sae,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str = "cpu",
    chunk_size: int = 10_000,
) -> float:
    """
    Compute the fraction of variance in the (normalised) GNN activations
    that is explained by the SAE reconstruction.

    A well-trained SAE should achieve >0.95 on the training set.
    """
    acts_norm = (gnn_acts - mean) / std
    sae = sae.to(device)

    recons = []
    with torch.no_grad():
        for i in range(0, acts_norm.shape[0], chunk_size):
            chunk = acts_norm[i : i + chunk_size].to(device)
            x_hat, _, _ = sae(chunk)
            recons.append(x_hat.cpu())

    recon = torch.cat(recons, dim=0)
    residual_var = (acts_norm - recon).var().item()
    total_var    = acts_norm.var().item()
    return 1.0 - residual_var / (total_var + 1e-10)
