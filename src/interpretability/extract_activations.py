"""
1_extract_activations.py
------------------------
Hook into the FROZEN Con-CDVAE encoder (DimeNet++) and dump every atom's
hidden-state vector to disk.

Usage
-----
    python 1_extract_activations.py \
        --checkpoint /path/to/epoch=xxx-step=xxx.ckpt \
        --data_path  /path/to/con-cdvae/data/mp_20 \
        --output_dir ./sae_activations \
        --layer_idx  -1          # -1 = last interaction block (recommended)
        --batch_size 32          # crystals per batch (not atoms)
        --device     cuda

Output
------
    sae_activations/
        activations_train.pt    ← torch.Tensor [N_atoms_total, hidden_dim]
        meta_train.pt           ← dict with atom-to-crystal mapping, element types
        activations_val.pt
        meta_val.pt
"""

import argparse
import os
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
# -----------------------------------------------------------------------
# Con-CDVAE imports (assumes you are running from the repo root)
# -----------------------------------------------------------------------
# The Con-CDVAE repo uses Hydra + pytorch-lightning.
# We load the checkpoint directly via pytorch-lightning's load_from_checkpoint.
# -----------------------------------------------------------------------
try:
    from concdvae.pl_modules.model import CDVAE  # v2.x module path
except ImportError:
    try:
        from model.cdvae.pl_modules.model import CDVAE as ConCDVAE  # fallback
    except ImportError:
        raise ImportError(
            "Could not import Con-CDVAE model. Make sure you are running from "
            "the repository root with the conda environment activated."
        )

try:
    from concdvae.pl_data.dataset import CrystDataset  # v2.x
except ImportError:
    from cdvae.pl_data.dataset import CrystDataset


# -----------------------------------------------------------------------
# Hook management
# -----------------------------------------------------------------------

class ActivationHook:
    """
    Registers a forward hook on a PyTorch module and collects all outputs.

    For DimeNet++ interaction blocks the output is a node-feature tensor of
    shape [N_atoms_in_batch, hidden_dim].
    """

    def __init__(self):
        self.activations: list[torch.Tensor] = []
        self._handle = None

    def hook_fn(self, module, input, output):
        # Some DimeNet++ blocks return (node_feats, edge_feats) tuples;
        # we always want the node feature tensor (first element or direct tensor).
        if isinstance(output, (tuple, list)):
            node_feats = output[0]
        else:
            node_feats = output
        self.activations.append(node_feats.detach().cpu())

    def register(self, module: torch.nn.Module):
        self._handle = module.register_forward_hook(self.hook_fn)

    def clear(self):
        self.activations = []

    def remove(self):
        if self._handle is not None:
            self._handle.remove()

    def collect(self) -> torch.Tensor:
        """Concatenate all collected activations along the atom dimension."""
        return torch.cat(self.activations, dim=0)


def get_interaction_blocks(encoder: torch.nn.Module) -> list:
    """
    Return the list of DimeNet++ interaction blocks from the encoder.

    Con-CDVAE's encoder is a DimeNetPlusPlusWrap; the interaction blocks are
    stored as self.interaction_blocks (a ModuleList).
    """
    # Try standard attribute names used by CDVAE / Con-CDVAE
    for attr in ("interaction_blocks", "int_blocks", "layers", "blocks"):
        if hasattr(encoder, attr):
            blocks = getattr(encoder, attr)
            if isinstance(blocks, torch.nn.ModuleList) and len(blocks) > 0:
                return list(blocks)

    # Fallback: collect all modules whose class name contains 'Interaction' or 'Block'
    blocks = [
        m for m in encoder.modules()
        if any(k in type(m).__name__ for k in ("Interaction", "Block", "MessagePassing"))
    ]
    if not blocks:
        raise RuntimeError(
            "Could not find DimeNet++ interaction blocks in encoder. "
            "Inspect encoder.named_modules() and set the correct attribute name."
        )
    return blocks


# -----------------------------------------------------------------------
# Main extraction function
# -----------------------------------------------------------------------

@torch.no_grad()
def extract_activations(
    model: torch.nn.Module,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Pass every batch through the frozen encoder and collect per-atom activations
    from the specified interaction block.

    Returns
    -------
    activations : Tensor [N_atoms_total, hidden_dim]
    meta        : dict with:
        'atom_types'     : Tensor [N_atoms_total]          (atomic number)
        'crystal_ids'    : Tensor [N_atoms_total]          (which crystal)
        'batch_offsets'  : list[int]                       (first atom idx per crystal)
    """
    model.eval()
    model.to(device)

    encoder = model.encoder  # DimeNetPlusPlusWrap
    blocks = get_interaction_blocks(encoder)
    target_block = blocks[layer_idx]

    hook = ActivationHook()
    hook.register(target_block)

    atom_types_all = []
    crystal_ids_all = []
    crystal_counter = 0

    for batch in loader:
        batch = batch.to(device)
        hook.clear()

        # Forward through encoder only (we don't need the full VAE pass)
        # Con-CDVAE encoder takes a Batch object from torch-geometric
        try:
            _ = encoder(
                batch.atom_types,
                batch.frac_coords,
                batch.lengths,
                batch.angles,
                batch.edge_index,
                batch.to_jimages,
                batch.num_atoms,
            )
        except TypeError:
            # Fallback: some versions accept the batch object directly
            _ = encoder(batch)

        # Collect metadata
        n_atoms_per_crystal = batch.num_atoms.tolist()
        for n in n_atoms_per_crystal:
            crystal_ids_all.append(
                torch.full((n,), crystal_counter, dtype=torch.long)
            )
            crystal_counter += 1

        atom_types_all.append(batch.atom_types.cpu())

    hook.remove()

    activations = hook.collect()          # [N_total, hidden_dim]
    atom_types  = torch.cat(atom_types_all, dim=0)
    crystal_ids = torch.cat(crystal_ids_all, dim=0)

    meta = {
        "atom_types":   atom_types,
        "crystal_ids":  crystal_ids,
        "n_crystals":   crystal_counter,
        "hidden_dim":   activations.shape[-1],
    }

    return activations, meta


# -----------------------------------------------------------------------
# Dataset loading helper
# -----------------------------------------------------------------------


def load_mp20_splits(data_path: str, batch_size: int) -> dict[str, DataLoader]:
    """
    Load train/val/test splits from the MP20 dataset folder.
    Assumes the same format as the Con-CDVAE data/ directory.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = CrystDataset(
            name=split,
            path=data_path,
            prop="formation_energy_per_atom",
            niggli=True,
            primitive=False,
            graph_method="crystalnn",
            preprocess_workers=8,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
    return loaders


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract Con-CDVAE GNN activations for SAE training")
    p.add_argument("--checkpoint",  required=True,  help="Path to Con-CDVAE .ckpt file")
    p.add_argument("--data_path",   required=True,  help="Path to mp_20 data directory")
    p.add_argument("--output_dir",  default="sae_activations")
    p.add_argument("--layer_idx",   type=int, default=-1,
                   help="Which DimeNet++ block to hook (-1 = last, recommended)")
    p.add_argument("--batch_size",  type=int, default=32,
                   help="Number of crystals per batch")
    p.add_argument("--splits",      nargs="+", default=["train", "val"],
                   help="Which splits to extract")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load frozen model
    print(f"Loading Con-CDVAE checkpoint: {args.checkpoint}")
    model = CDVAE.load_from_checkpoint(args.checkpoint)
    for p in model.parameters():
        p.requires_grad_(False)  # freeze everything
    print(f"  Encoder hidden_dim inferred from model: will be shown after first batch")

    # 2. Load data
    print(f"Loading MP20 dataset from: {args.data_path}")
    loaders = load_mp20_splits(args.data_path, args.batch_size)

    # 3. Extract per split
    for split in args.splits:
        if split not in loaders:
            print(f"  Warning: split '{split}' not found, skipping.")
            continue

        print(f"\n=== Extracting {split} split ===")
        acts, meta = extract_activations(
            model, loaders[split], args.layer_idx, device
        )
        print(f"  activations shape : {acts.shape}")
        print(f"  n_crystals        : {meta['n_crystals']}")
        print(f"  hidden_dim        : {meta['hidden_dim']}")

        # Save
        acts_path = out_dir / f"activations_{split}.pt"
        meta_path = out_dir / f"meta_{split}.pt"
        torch.save(acts, acts_path)
        torch.save(meta, meta_path)
        print(f"  Saved → {acts_path}  ({acts.element_size() * acts.nelement() / 1e6:.1f} MB)")

    print("\nDone. Ready for SAE training (run 2_train_sae.py).")


if __name__ == "__main__":
    main()
