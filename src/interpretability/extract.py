# extract.py — Activation extraction from frozen Con-CDVAE
#
# Runs forward passes over the full dataset with hooks registered,
# collecting activations at all probe points. Saves results to disk
# so probes can be trained without re-running the model.
#
# Output structure (saved to output_dir/):
#   activations_{split}.pt  — dict[hook_name -> Tensor]
#   labels_{split}.pt       — dict[label_name -> ndarray]
#   num_atoms_{split}.pt    — list[int] (for atom->crystal pooling)
# =============================================================================

from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .hooks import HookManager, HOOK_POINTS
from .labels import LabelExtractor
from .train_probes import pool_atom_activations


def extract_activations(
    model: torch.nn.Module,
    dataset,
    csv_path: str | Path,
    output_dir: str | Path,
    split: str = "train",
    batch_size: int = 64,
    device: str = "cpu",
) -> None:
    """
    Run the frozen model on the full dataset, collecting activations
    at all hook points. Save to disk.

    Parameters
    ----------
    model : CDVAE
        Loaded and frozen Con-CDVAE model.
    dataset : CrystDataset
        The dataset split to extract from.
    csv_path : str or Path
        Path to the CSV for this split (for spacegroup labels).
    output_dir : str or Path
        Directory to save extracted activations and labels.
    split : str
        Name of the split ("train", "val", "test").
    batch_size : int
        Crystals per batch for forward passes.
    device : str
        "cuda" or "cpu".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    model.to(device_obj)
    model.eval()

    # -- Register hooks -------------------------------------------------------
    hook_mgr = HookManager(model)
    hook_mgr.register_hooks()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # -- Accumulators ---------------------------------------------------------
    # Per-crystal hooks: accumulate directly
    # Per-atom hooks: accumulate raw, pool later
    all_activations = {name: [] for name in HOOK_POINTS}
    all_num_atoms = []

    print(f"Extracting {split} activations ({len(dataset)} crystals)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device_obj)
            hook_mgr.clear()

            # Full forward pass to trigger all hooks
            model(batch, teacher_forcing=False, training=False)

            acts = hook_mgr.get_activations()
            num_atoms_batch = batch.num_atoms.cpu().tolist()
            all_num_atoms.extend(num_atoms_batch)

            for name in HOOK_POINTS:
                if name not in acts:
                    print(f"  Warning: hook '{name}' did not capture. Skipping.")
                    continue

                act = acts[name]
                level = HOOK_POINTS[name]["level"]

                if level == "atom":
                    # Pool per-atom -> per-crystal now to save memory
                    act = pool_atom_activations(act, num_atoms_batch)

                all_activations[name].append(act)

            if (batch_idx + 1) % 20 == 0:
                n_done = min((batch_idx + 1) * batch_size, len(dataset))
                print(f"  {n_done}/{len(dataset)} crystals processed")

    # -- Concatenate and save -------------------------------------------------
    final_activations = {}
    for name in HOOK_POINTS:
        if all_activations[name]:
            final_activations[name] = torch.cat(all_activations[name], dim=0)
            print(f"  {name}: {final_activations[name].shape}")

    # -- Extract labels -------------------------------------------------------
    print(f"Extracting labels from dataset + CSV...")
    label_extractor = LabelExtractor(dataset, csv_path)
    labels = label_extractor.get_all_labels()

    # -- Save -----------------------------------------------------------------
    acts_path = output_dir / f"activations_{split}.pt"
    labels_path = output_dir / f"labels_{split}.pt"
    num_atoms_path = output_dir / f"num_atoms_{split}.pt"

    torch.save(final_activations, acts_path)
    torch.save(labels, labels_path)
    torch.save(all_num_atoms, num_atoms_path)

    print(f"Saved to {output_dir}/:")
    print(f"  {acts_path.name}  ({sum(v.nelement() * v.element_size() for v in final_activations.values()) / 1e6:.1f} MB)")
    print(f"  {labels_path.name}")
    print(f"  {num_atoms_path.name}")
