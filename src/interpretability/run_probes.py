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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from .extract import extract_activations
from .labels import PROBE_TASKS
from .hooks import HOOK_POINTS
from .train_probes import ProbeTrainer



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



def run_probe_pipeline(args):
    """Full probe pipeline: extract -> train -> evaluate -> report."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Loading frozen Con-CDVAE model...")
    model, cfg = load_frozen_model(args.model_dir, args.device)

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

    print(f"\n{'=' * 70}")
    print("Loading extracted activations and labels...")

    train_acts = torch.load(output_dir / "activations_train.pt", weights_only=False)
    train_labels = torch.load(output_dir / "labels_train.pt", weights_only=False)
    val_acts = torch.load(output_dir / "activations_val.pt", weights_only=False)
    val_labels = torch.load(output_dir / "labels_val.pt", weights_only=False)

    n_train = len(next(iter(train_labels.values())))
    n_val = len(next(iter(val_labels.values())))
    print(f"  Train: {n_train} crystals, Val: {n_val} crystals")

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
                early_stop_patience=args.early_stop if args.early_stop else None,
            )

            metrics = trainer.train(X_train, y_train, X_val, y_val)
            results[(prop_name, hook_name)] = metrics

            # Save individual probe model (for SAE probe alignment later)
            probe_dir = output_dir / "probes" / args.probe_type
            probe_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                trainer.probe.state_dict(),
                probe_dir / f"{prop_name}_{hook_name}.pt",
            )

            # Print inline
            if task == "regression":
                print(f"  {prop_name:25s} @ {hook_name:10s}  R²={metrics['r2']:.4f}  MSE={metrics['mse']:.4f}")
            else:
                print(f"  {prop_name:25s} @ {hook_name:10s}  Acc={metrics['accuracy']:.4f}")

    print_results_table(results)

    if args.visualise:
        fig_dir = Path(args.fig_dir) if args.fig_dir else output_dir / "figures"
        plot_probe_training_curves(results, fig_dir)

    results_path = output_dir / f"probe_results_{args.probe_type}.pt"
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")


def plot_probe_training_curves(results: dict, fig_dir: Path) -> None:
    """Plot per-epoch training curves for all probes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Group by property for subplot layout
    props_seen = {}
    for (prop, hook), m in results.items():
        if "history" not in m:
            continue
        props_seen.setdefault(prop, []).append((hook, m["history"]))

    if not props_seen:
        print("  No training history found, skipping visualisation.")
        return

    for prop, hook_histories in props_seen.items():
        n_hooks = len(hook_histories)
        fig, axes = plt.subplots(1, n_hooks, figsize=(5 * n_hooks, 4), squeeze=False)
        fig.suptitle(prop, fontsize=14)

        for col, (hook, hist) in enumerate(hook_histories):
            ax = axes[0, col]
            epochs = range(1, len(hist["train_loss"]) + 1)

            # Primary axis: loss
            ax.plot(epochs, hist["train_loss"], label="train loss", color="tab:blue")
            ax.plot(epochs, hist["val_loss"], label="val loss", color="tab:orange")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(hook)

            # Secondary axis: metric
            ax2 = ax.twinx()
            if "val_r2" in hist:
                ax2.plot(epochs, hist["val_r2"], label="val R²",
                         color="tab:green", linestyle="--")
                ax2.set_ylabel("R²")
            elif "val_accuracy" in hist:
                ax2.plot(epochs, hist["val_accuracy"], label="val acc",
                         color="tab:green", linestyle="--")
                ax2.set_ylabel("Accuracy")

            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      loc="center right", fontsize=8)

        plt.tight_layout()
        save_path = fig_dir / f"probe_{prop}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {save_path}")

    print(f"  Probe training curves saved to {fig_dir}/")



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
    p.add_argument("--early_stop", type=int, default=None,
                   help="Early stopping patience (epochs without improvement). "
                        "Disabled by default.")
    p.add_argument("--visualise", action="store_true",
                   help="Save per-epoch training curve plots")
    p.add_argument("--fig_dir", type=str, default=None,
                   help="Directory for figure output (default: <output_dir>/figures/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_probe_pipeline(args)
