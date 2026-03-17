#!/usr/bin/env python3
# run_probes.py — CLI entry point for the full probe pipeline
#
# End-to-end pipeline:
#   1. Load the frozen Con-CDVAE model and dataset
#   2. Extract activations at all hook points (or load from cache)
#   3. Train probes for each (hook_point, property) pair
#   4. Print results table
#
# Usage
# -----
#   python -m src.interpretability.run_probes \
#       --model_dir  src/model/mp20_format \
#       --data_dir   data/mp_20 \
#       --output_dir src/interpretability/outputs \
#       --device     cuda \
#       --probe_type linear \
#       --num_epochs 100

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# ---- Ensure repo root is on path -------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from .extract import extract_activations
from .labels import PROBE_TASKS
from .hooks import HOOK_POINTS
from .train_probes import ProbeTrainer


# ---- Model loading ----------------------------------------------------------

def load_frozen_model(model_dir: str, device: str = "cpu"):
    """
    Load Con-CDVAE using the authors' eval_utils.load_model function.

    Returns (model, cfg) with all parameters frozen.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from eval_utils import load_model

    # eval_utils / Hydra requires an absolute path
    model_dir_abs = str(Path(model_dir).resolve())

    model, _, cfg = load_model(model_path=model_dir_abs)
    model.to(device)
    model.eval()

    # Load lattice scaler (required for forward pass)
    scaler_path = Path(model_dir_abs) / "lattice_scaler.pt"
    if scaler_path.exists():
        model.lattice_scaler = torch.load(
            scaler_path, map_location=device, weights_only=False,
        )

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg


def load_dataset(data_dir: str, cfg, split: str = "train"):
    """Load a CrystDataset split using the model's config."""
    from concdvae.pl_data.dataset import CrystDataset

    csv_path = Path(data_dir) / f"{split}.csv"
    save_path = Path(data_dir) / f"{split}_data.pt"

    dataset = CrystDataset(
        name=f"Probe extraction {split}",
        path=str(csv_path),
        prop=cfg.data.prop,
        niggli=cfg.data.niggli,
        primitive=cfg.data.primitive,
        graph_method=cfg.data.graph_method,
        lattice_scale_method=cfg.data.lattice_scale_method,
        preprocess_workers=4,
        save_path=str(save_path),
        tolerance=cfg.data.tolerance,
        use_space_group=cfg.data.use_space_group,
        use_pos_index=cfg.data.use_pos_index,
        load_old=True,
        prelo_prop=cfg.data.prelo_prop,
    )
    return dataset, csv_path


# ---- Main pipeline ----------------------------------------------------------

def run_probe_pipeline(args):
    """Full probe pipeline: extract -> train -> evaluate -> report."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Step 1: Load model ---------------------------------------------------
    print("=" * 70)
    print("Loading frozen Con-CDVAE model...")
    model, cfg = load_frozen_model(args.model_dir, args.device)

    # -- Step 2: Extract activations (or load from cache) ---------------------
    for split in ["train", "val"]:
        acts_path = output_dir / f"activations_{split}.pt"
        if acts_path.exists() and not args.force_extract:
            print(f"Found cached {acts_path.name}, skipping extraction.")
            continue

        print(f"\n{'=' * 70}")
        dataset, csv_path = load_dataset(args.data_dir, cfg, split)
        extract_activations(
            model=model,
            dataset=dataset,
            csv_path=csv_path,
            output_dir=output_dir,
            split=split,
            batch_size=args.batch_size,
            device=args.device,
        )

    # -- Step 3: Load extracted data ------------------------------------------
    print(f"\n{'=' * 70}")
    print("Loading extracted activations and labels...")

    train_acts = torch.load(output_dir / "activations_train.pt", weights_only=False)
    train_labels = torch.load(output_dir / "labels_train.pt", weights_only=False)
    val_acts = torch.load(output_dir / "activations_val.pt", weights_only=False)
    val_labels = torch.load(output_dir / "labels_val.pt", weights_only=False)

    n_train = len(next(iter(train_labels.values())))
    n_val = len(next(iter(val_labels.values())))
    print(f"  Train: {n_train} crystals, Val: {n_val} crystals")

    # -- Step 4: Train probes -------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training {args.probe_type} probes ({args.num_epochs} epochs)...\n")

    results = {}   # (property, hook) -> metrics dict

    for prop_name, task_cfg in PROBE_TASKS.items():
        task = task_cfg["task"]
        hooks = task_cfg["hooks"]
        num_classes = task_cfg.get("num_classes", 1)

        for hook_name in hooks:
            if hook_name not in train_acts:
                print(f"  SKIP {prop_name:25s} @ {hook_name:10s} (no activations)")
                continue

            X_train = train_acts[hook_name]
            X_val = val_acts[hook_name]
            y_train = torch.tensor(train_labels[prop_name])
            y_val = torch.tensor(val_labels[prop_name])

            input_dim = X_train.shape[1]

            trainer = ProbeTrainer(
                task=task,
                input_dim=input_dim,
                num_classes=num_classes,
                probe_type=args.probe_type,
                lr=args.lr,
                num_epochs=args.num_epochs,
                batch_size=args.probe_batch_size,
                device=args.device,
            )

            metrics = trainer.train(X_train, y_train, X_val, y_val)
            results[(prop_name, hook_name)] = metrics

            # Save individual probe model (for SAE probe alignment later)
            probe_dir = output_dir / "probes"
            probe_dir.mkdir(exist_ok=True)
            torch.save(
                trainer.probe.state_dict(),
                probe_dir / f"{prop_name}_{hook_name}_{args.probe_type}.pt",
            )

            # Print inline
            if task == "regression":
                print(f"  {prop_name:25s} @ {hook_name:10s}  R²={metrics['r2']:.4f}  MSE={metrics['mse']:.4f}")
            else:
                print(f"  {prop_name:25s} @ {hook_name:10s}  Acc={metrics['accuracy']:.4f}")

    # -- Step 5: Print summary table ------------------------------------------
    print_results_table(results)

    # -- Save results ---------------------------------------------------------
    torch.save(results, output_dir / "probe_results.pt")
    print(f"\nResults saved to {output_dir / 'probe_results.pt'}")


# ---- Pretty printing --------------------------------------------------------

def print_results_table(results: dict) -> None:
    """Print the probe matrix as a formatted table."""
    # Collect all hooks and properties that appear
    all_hooks = []
    all_props = []
    for (prop, hook) in results:
        if hook not in all_hooks:
            all_hooks.append(hook)
        if prop not in all_props:
            all_props.append(prop)

    print(f"\n{'=' * 70}")
    print("PROBE RESULTS SUMMARY")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Property':25s}"
    for hook in all_hooks:
        header += f" | {hook:>10s}"
    print(header)
    print("-" * len(header))

    # Rows
    for prop in all_props:
        row = f"{prop:25s}"
        for hook in all_hooks:
            key = (prop, hook)
            if key not in results:
                row += f" | {'--':>10s}"
            else:
                m = results[key]
                if "r2" in m:
                    row += f" | {m['r2']:>10.4f}"
                elif "accuracy" in m:
                    row += f" | {m['accuracy']:>10.4f}"
            row += ""
        print(row)

    print(f"{'=' * 70}")


# ---- CLI --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Con-CDVAE probe pipeline",
    )
    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to model checkpoint directory (e.g., src/model/mp20_format)")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to dataset directory (e.g., data/mp_20)")
    p.add_argument("--output_dir", type=str, default="src/interpretability/outputs",
                   help="Directory to save activations and results")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Crystals per batch for activation extraction")
    p.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"],
                   help="Probe architecture: 'linear' or 'mlp'")
    p.add_argument("--num_epochs", type=int, default=100,
                   help="Epochs for probe training")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for probe training")
    p.add_argument("--probe_batch_size", type=int, default=512,
                   help="Batch size for probe training")
    p.add_argument("--force_extract", action="store_true",
                   help="Re-extract activations even if cache exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_probe_pipeline(args)
