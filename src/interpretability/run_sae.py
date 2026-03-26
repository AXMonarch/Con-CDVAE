#!/usr/bin/env python3
# run_sae.py — CLI entry point for Normalized SAE training and analysis
#
# End-to-end SAE pipeline:
#   1. Load pre-extracted activations (from probe pipeline)
#   2. Train a Normalized Top-K SAE on a chosen hook point (default: z_con)
#      (input standardization computed from training set, stored as buffers)
#   3. Run full analysis: stats, correlations, co-occurrence, decoder
#      similarity, element enrichment, probe alignment, variance explained
#   4. Print feature dashboard and save results
#
# Usage
# -----
#   python -m src.interpretability.run_sae \
#       --activations_dir src/interpretability/outputs \
#       --hook_point z_con \
#       --output_dir src/interpretability/outputs/sae \
#       --device cuda
#
# With probe alignment and element enrichment:
#   python -m src.interpretability.run_sae \
#       --activations_dir src/interpretability/outputs \
#       --probe_results src/interpretability/outputs/probe_results.pt \
#       --model_dir src/model/mp20_format \
#       --data_dir data/mp_20 \
#       --device cuda

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

from .sae_model import TopKSAE, SAEConfig
from .train_sae import SAETrainer
from .analyse_sae import SAEAnalyser
from .hooks import HOOK_POINTS


def load_probe_weights(probe_results_path: str, hook_point: str) -> dict[str, torch.Tensor]:
    """
    Extract linear probe weight vectors from saved probe results.

    The probe pipeline saves trained probes to probe_results.pt. Each
    linear probe's weight vector defines a direction in activation space
    that maximally predicts the target property.

    Parameters
    ----------
    probe_results_path : str
        Path to probe_results.pt (or directory containing linear probe .pt files).
    hook_point : str
        Which hook point's probes to load (e.g., "z_con").

    Returns
    -------
    weights : dict mapping property name -> Tensor (input_dim,)
    """
    results_dir = Path(probe_results_path).parent
    weights = {}

    # Only linear probes have a single weight direction usable for alignment.
    # Structure: probes/linear/{prop}_{hook}.pt
    linear_dir = results_dir / "probes" / "linear"
    if linear_dir.exists():
        for path in linear_dir.glob(f"*_{hook_point}.pt"):
            prop_name = path.stem.replace(f"_{hook_point}", "")
            state = torch.load(path, weights_only=False)
            if "linear.weight" in state:
                weights[prop_name] = state["linear.weight"].cpu()

    if not weights:
        print("  Note: No linear probe model files found. Skipping probe alignment.")
        print(f"  (Looked in {linear_dir} for *_{hook_point}.pt)")
        print("  Re-run probes with --probe_type linear to generate them.")

    return weights


def load_dataset_for_enrichment(data_dir: str, model_dir: str, split: str = "val"):
    """
    Load the dataset for element enrichment analysis.

    Reuses the same loading logic as run_probes.py.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from eval_utils import load_model

    model_dir_abs = str(Path(model_dir).resolve())
    _, _, cfg = load_model(model_path=model_dir_abs)

    from concdvae.pl_data.dataset import CrystDataset
    csv_path = Path(data_dir) / f"{split}.csv"
    save_path = Path(data_dir) / f"{split}_data.pt"

    dataset = CrystDataset(
        name=f"SAE analysis {split}",
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
    return dataset


def plot_sae_training_curves(history: dict, fig_dir: Path) -> None:
    """Plot SAE training curves: loss, variance explained, dead features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Loss
    axes[0].plot(epochs, history["train_loss"], label="train MSE")
    axes[0].plot(epochs, history["val_loss"], label="val MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (normalized)")
    axes[0].set_title("Reconstruction Loss")
    axes[0].legend()

    # Panel 2: Variance explained
    axes[1].plot(epochs, history["var_explained"], label="normalized")
    axes[1].plot(epochs, history["var_explained_raw"], label="raw")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Variance Explained")
    axes[1].set_title("Variance Explained")
    axes[1].legend()

    # Panel 3: Dead features
    axes[2].plot(epochs, history["n_dead"], color="tab:red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Dead Features")

    plt.tight_layout()
    save_path = fig_dir / "sae_training.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved SAE training curves to {save_path}")


def run_sae_pipeline(args):
    """Full SAE pipeline: load data -> train -> analyse -> report."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    acts_dir = Path(args.activations_dir)

    # -- Step 1: Load pre-extracted activations --------------------------------
    print("=" * 70)
    print(f"Loading activations from {acts_dir}...")

    train_acts = torch.load(acts_dir / "activations_train.pt", weights_only=False)
    val_acts = torch.load(acts_dir / "activations_val.pt", weights_only=False)
    train_labels = torch.load(acts_dir / "labels_train.pt", weights_only=False)
    val_labels = torch.load(acts_dir / "labels_val.pt", weights_only=False)

    hook = args.hook_point
    if hook not in train_acts:
        available = list(train_acts.keys())
        print(f"ERROR: hook '{hook}' not found. Available: {available}")
        sys.exit(1)

    X_train = train_acts[hook].float()
    X_val = val_acts[hook].float()
    input_dim = X_train.shape[1]

    print(f"  Hook: {hook}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # -- Step 2: Train Normalized SAE ------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Training Normalized Top-K SAE (K={args.k}, {args.n_features} features)...")

    config = SAEConfig(
        input_dim=input_dim,
        n_features=args.n_features,
        k=args.k,
    )
    trainer = SAETrainer(
        config=config,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=args.device,
        early_stop_patience=args.early_stop if args.early_stop else None,
    )
    history = trainer.train(X_train, X_val)

    # Save the trained SAE
    sae_path = output_dir / f"sae_{hook}.pt"
    torch.save({
        "config": config,
        "state_dict": trainer.sae.state_dict(),
        "history": history,
    }, sae_path)
    print(f"\nSaved SAE to {sae_path}")

    # Visualise training curves
    if args.visualise:
        fig_dir = Path(args.fig_dir) if args.fig_dir else output_dir / "figures"
        plot_sae_training_curves(history, fig_dir)

    # -- Step 3: Full analysis -------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Running full SAE analysis...")

    analyser = SAEAnalyser(
        sae=trainer.sae,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Load optional probe weights for alignment
    probe_weights = None
    if args.probe_results:
        print("Loading probe weights for alignment...")
        probe_weights = load_probe_weights(args.probe_results, hook)

    # Load optional dataset for element enrichment
    dataset = None
    if args.data_dir and args.model_dir:
        print("Loading dataset for element enrichment...")
        dataset = load_dataset_for_enrichment(args.data_dir, args.model_dir)

    # Run the full analysis pipeline
    results = analyser.full_analysis(
        X=X_val,
        labels=val_labels,
        probe_weights=probe_weights if probe_weights else None,
        dataset=dataset,
        n_top_features=100,
        top_p=args.top_p,
    )

    # Print full report
    analyser.print_full_report(results)
    print(f"Final variance explained (normalized): {history['var_explained'][-1]:.4f}")
    print(f"Final variance explained (raw):        {history['var_explained_raw'][-1]:.4f}")

    # -- Save all results ------------------------------------------------------
    # Remove large tensors from saved results to keep file size manageable
    save_results = {k: v for k, v in results.items() if k != "H"}
    save_results["H_val"] = results["H"]
    save_results["history"] = history

    analysis_path = output_dir / f"analysis_{hook}.pt"
    torch.save(save_results, analysis_path)
    print(f"\nSaved full analysis to {analysis_path}")


# ---- CLI --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train and analyse a Top-K SAE on Con-CDVAE activations",
    )
    # Required / main args
    p.add_argument("--activations_dir", type=str,
                   default="src/interpretability/outputs",
                   help="Directory containing activations_train.pt etc.")
    p.add_argument("--hook_point", type=str, default="z_con",
                   help="Which hook point to train SAE on (default: z_con)")
    p.add_argument("--output_dir", type=str,
                   default="src/interpretability/outputs/sae",
                   help="Directory to save SAE model and analysis")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # SAE hyperparameters
    p.add_argument("--n_features", type=int, default=4096,
                   help="SAE dictionary size (default: 4096 = 16x expansion)")
    p.add_argument("--k", type=int, default=32,
                   help="Number of active features per input (default: 32)")
    p.add_argument("--num_epochs", type=int, default=20,
                   help="Training epochs")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate")
    p.add_argument("--batch_size", type=int, default=4096,
                   help="Batch size for SAE training")

    # Dashboard feature selection
    p.add_argument("--top_p", type=float, default=None,
                   help="Include top p%% of features in dashboard (e.g. 0.10 for 10%%). "
                        "Overrides the default n_top_features=100.")

    # Optional: for probe alignment
    p.add_argument("--probe_results", type=str, default=None,
                   help="Path to probe_results.pt for probe direction alignment")

    # Optional: for element enrichment (requires loading the dataset)
    p.add_argument("--model_dir", type=str, default=None,
                   help="Path to model dir (for loading dataset config)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to data dir (for element enrichment)")

    # Training control
    p.add_argument("--early_stop", type=int, default=None,
                   help="Early stopping patience (epochs without val loss improvement). "
                        "Disabled by default.")
    p.add_argument("--visualise", action="store_true",
                   help="Save training curve plots")
    p.add_argument("--fig_dir", type=str, default=None,
                   help="Directory for figure output (default: <output_dir>/figures/)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sae_pipeline(args)
