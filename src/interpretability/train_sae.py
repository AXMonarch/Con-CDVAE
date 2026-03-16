"""
2_train_sae.py
--------------
Train the Top-K SAE on the atom-level GNN activations collected in Step 1.

Usage
-----
    python 2_train_sae.py \
        --activations_dir ./sae_activations \
        --output_dir      ./sae_checkpoints \
        --input_dim       256   \
        --expansion       16    \
        --k               32    \
        --epochs          20    \
        --batch_size      4096  \
        --lr              2e-4  \
        --device          cuda

Training loop notes
-------------------
  1. We train on *atom-level* tokens (each atom in each crystal is one example).
     With MP20 (~45k crystals, avg ~8 atoms each) you get ~360k training tokens.
     With expansion 16 and dim 256 you get 4,096 SAE features.
  2. After every gradient step we L2-normalise the decoder columns.
  3. Every `reinit_every` steps we check for dead features and re-initialise them.
  4. We track: reconstruction loss, L0 (avg features fired), dead feature count.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sae_model import TopKSAE, SAEConfig


# -----------------------------------------------------------------------
# Normalisation
# -----------------------------------------------------------------------

def compute_normalisation(activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean and std over the training corpus.
    Returns mean [d_in] and std [d_in].
    """
    mean = activations.mean(dim=0)
    std  = activations.std(dim=0).clamp(min=1e-6)
    return mean, std


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_sae(
    sae: TopKSAE,
    train_loader: DataLoader,
    val_acts: torch.Tensor,
    cfg: SAEConfig,
    device: torch.device,
    out_dir: Path,
    reinit_every: int = 1000,
):
    sae.to(device)
    val_acts = val_acts.to(device)

    optimiser = Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg.num_epochs * len(train_loader))

    step = 0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        sae.train()
        epoch_recon = 0.0
        epoch_l0    = 0.0
        n_batches   = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)

            optimiser.zero_grad()
            _, _, loss_dict = sae(x_batch)
            loss = loss_dict["total_loss"]
            loss.backward()
            optimiser.step()
            scheduler.step()

            # Normalise decoder after every step
            if cfg.normalise_decoder:
                sae.normalise_decoder_()

            epoch_recon += loss_dict["recon_loss"].item()
            epoch_l0    += loss_dict["l0"].item()
            n_batches   += 1
            step        += 1

            # Dead-feature re-initialisation
            if step % reinit_every == 0:
                n_dead = sae.reinit_dead_features_(x_batch, threshold=reinit_every * 5)
                if n_dead > 0:
                    print(f"  [step {step}] Reinitialised {n_dead} dead features.")

            # Periodic logging
            if step % cfg.log_every == 0:
                dead_count = len(sae.dead_features(threshold=5000))
                print(
                    f"  step {step:6d} | "
                    f"recon {loss_dict['recon_loss'].item():.5f} | "
                    f"L0 {loss_dict['l0'].item():.1f} / {cfg.k} | "
                    f"dead {dead_count}"
                )

        # Epoch-level validation
        sae.eval()
        with torch.no_grad():
            _, _, val_loss_dict = sae(val_acts)

        avg_train_recon = epoch_recon / n_batches
        avg_train_l0    = epoch_l0    / n_batches
        val_recon       = val_loss_dict["recon_loss"].item()
        dead_count      = len(sae.dead_features(threshold=5000))

        print(
            f"Epoch {epoch:3d}/{cfg.num_epochs} | "
            f"train_recon {avg_train_recon:.5f} | "
            f"val_recon {val_recon:.5f} | "
            f"L0 {avg_train_l0:.1f} | "
            f"dead_features {dead_count}/{cfg.dict_size}"
        )

        # Save best checkpoint
        if val_recon < best_val_loss:
            best_val_loss = val_recon
            ckpt_path = out_dir / "sae_best.pt"
            torch.save(
                {"state_dict": sae.state_dict(), "cfg": cfg, "step": step},
                ckpt_path,
            )

        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save(
                {"state_dict": sae.state_dict(), "cfg": cfg, "step": step},
                out_dir / f"sae_epoch{epoch:03d}.pt",
            )

    # Final save
    torch.save(
        {"state_dict": sae.state_dict(), "cfg": cfg, "step": step},
        out_dir / "sae_final.pt",
    )
    print(f"\nTraining complete. Best val recon loss: {best_val_loss:.5f}")
    return sae


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Top-K SAE on Con-CDVAE activations")
    p.add_argument("--activations_dir", default="sae_activations")
    p.add_argument("--output_dir",      default="sae_checkpoints")
    p.add_argument("--input_dim",   type=int,   default=256,
                   help="GNN hidden dimension (256 for CDVAE default)")
    p.add_argument("--expansion",   type=int,   default=16,
                   help="SAE expansion factor (dict_size = expansion * input_dim)")
    p.add_argument("--k",           type=int,   default=32,
                   help="Top-K: number of features allowed to fire per token")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=4096)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--normalise_decoder", action="store_true", default=True)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load activations
    acts_dir = Path(args.activations_dir)
    print("Loading training activations …")
    train_acts = torch.load(acts_dir / "activations_train.pt", weights_only=True)
    val_acts   = torch.load(acts_dir / "activations_val.pt",   weights_only=True)

    input_dim = train_acts.shape[1]
    print(f"  train tokens : {train_acts.shape[0]:,}")
    print(f"  val tokens   : {val_acts.shape[0]:,}")
    print(f"  input_dim    : {input_dim}")

    if args.input_dim != input_dim:
        print(f"  Note: overriding --input_dim {args.input_dim} → {input_dim} (from data)")

    # 2. Normalise (zero-mean, unit-variance per feature)
    print("Computing normalisation statistics …")
    mean, std = compute_normalisation(train_acts)
    train_acts_norm = (train_acts - mean) / std
    val_acts_norm   = (val_acts   - mean) / std
    torch.save({"mean": mean, "std": std}, out_dir / "normalisation.pt")

    # 3. Build DataLoader (shuffle atoms during training)
    train_dataset = TensorDataset(train_acts_norm)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Build SAE
    cfg = SAEConfig(
        input_dim=input_dim,
        expansion_factor=args.expansion,
        k=args.k,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        normalise_decoder=args.normalise_decoder,
    )
    sae = TopKSAE(cfg)
    n_params = sum(p.numel() for p in sae.parameters()) / 1e6
    print(f"\nSAE: {cfg.dict_size} features, K={cfg.k}, {n_params:.2f}M parameters")

    # 5. Train
    print("\nStarting SAE training …\n")
    train_sae(sae, train_loader, val_acts_norm, cfg, device, out_dir)


if __name__ == "__main__":
    main()
