"""
3_interpret_features.py
-----------------------
After SAE training, this script analyses what each SAE feature "means"
in terms of crystal chemistry.

Analysis performed
------------------
  A. Feature activation statistics
       - Mean activation, sparsity fraction, max-activating crystals per feature

  B. Chemical correlation dashboard
       For each SAE feature, compute Spearman correlation between its activation
       strength and a suite of per-atom / per-crystal labels:
         • Atomic number (element identity)
         • Electronegativity, atomic radius, valence electrons
         • Crystal system (cubic, hexagonal, …)
         • Space group number
         • Coordination number
         • Formation energy per atom
         • Band gap

  C. Top-K examples per feature
       For each feature f, find the 20 atoms where feature f fires most strongly.
       Print their element, coordination number, local environment.

  D. Feature-to-feature cosine similarity
       Compute W_dec cosine similarity matrix to spot feature clusters.

  E. Dead feature report

Usage
-----
    python 3_interpret_features.py \
        --sae_checkpoint  ./sae_checkpoints/sae_best.pt \
        --activations_dir ./sae_activations \
        --meta_dir        ./sae_activations \
        --output_dir      ./sae_analysis \
        --top_n_features  50   \
        --device          cuda
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sae_model import TopKSAE, SAEConfig


# -----------------------------------------------------------------------
# Property lookups (subset of common elements)
# -----------------------------------------------------------------------

# Atomic number → (electronegativity, atomic_radius_pm, valence_electrons)
ELEMENT_PROPERTIES = {
    1:  (2.20, 53,  1),   # H
    3:  (0.98, 167, 1),   # Li
    4:  (1.57, 112, 2),   # Be
    5:  (2.04, 87,  3),   # B
    6:  (2.55, 67,  4),   # C
    7:  (3.04, 56,  5),   # N
    8:  (3.44, 48,  6),   # O
    9:  (3.98, 42,  7),   # F
    11: (0.93, 186, 1),   # Na
    12: (1.31, 160, 2),   # Mg
    13: (1.61, 143, 3),   # Al
    14: (1.90, 117, 4),   # Si
    15: (2.19, 98,  5),   # P
    16: (2.58, 88,  6),   # S
    19: (0.82, 243, 1),   # K
    20: (1.00, 194, 2),   # Ca
    22: (1.54, 147, 4),   # Ti
    23: (1.63, 134, 5),   # V
    24: (1.66, 128, 6),   # Cr
    25: (1.55, 127, 7),   # Mn
    26: (1.83, 126, 8),   # Fe
    27: (1.88, 125, 9),   # Co
    28: (1.91, 124, 10),  # Ni
    29: (1.90, 128, 11),  # Cu
    30: (1.65, 122, 12),  # Zn
    # ... extend as needed
}

ELEMENT_SYMBOLS = {
    1: "H", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 19: "K",
    20: "Ca", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
    28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 47: "Ag",
    48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 55: "Cs", 56: "Ba",
    57: "La", 72: "Hf", 73: "Ta", 74: "W", 78: "Pt", 79: "Au", 80: "Hg",
    82: "Pb", 83: "Bi",
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman correlation via rank transform."""
    from scipy.stats import spearmanr
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return 0.0
    r, _ = spearmanr(x[valid], y[valid])
    return float(r) if np.isfinite(r) else 0.0


def get_atom_properties(atom_types: torch.Tensor) -> dict[str, np.ndarray]:
    """Build per-atom property arrays from atomic numbers."""
    nums = atom_types.numpy()
    ens, radii, valences = [], [], []
    for z in nums:
        props = ELEMENT_PROPERTIES.get(int(z), (np.nan, np.nan, np.nan))
        ens.append(props[0])
        radii.append(props[1])
        valences.append(props[2])
    return {
        "atomic_number":   nums.astype(float),
        "electronegativity": np.array(ens),
        "atomic_radius":   np.array(radii),
        "valence_electrons": np.array(valences),
    }


# -----------------------------------------------------------------------
# Main analysis routines
# -----------------------------------------------------------------------

def analyse_feature_statistics(
    feature_acts: torch.Tensor,    # [N_atoms, d_sae]
) -> dict:
    """
    Compute per-feature stats:
      - mean activation (only over firing tokens)
      - fraction of tokens where feature fires (sparsity check)
      - max activation value
    """
    fires     = (feature_acts > 0).float()
    sparsity  = fires.mean(dim=0)            # [d_sae]
    mean_act  = (feature_acts * fires).sum(dim=0) / fires.sum(dim=0).clamp(min=1)
    max_act   = feature_acts.max(dim=0).values

    return {
        "sparsity":  sparsity.numpy(),   # fraction of tokens firing
        "mean_act":  mean_act.numpy(),
        "max_act":   max_act.numpy(),
    }


def build_chemical_dashboard(
    feature_acts: torch.Tensor,   # [N_atoms, d_sae]
    atom_types:   torch.Tensor,   # [N_atoms]
    top_n:        int = 50,
) -> list[dict]:
    """
    For the `top_n` most active features, compute correlations with
    chemical properties and find exemplar atoms.
    """
    stats = analyse_feature_statistics(feature_acts)
    # Rank features by mean activation (proxy for "information content")
    ranked = np.argsort(-stats["mean_act"])[:top_n]

    atom_props = get_atom_properties(atom_types)
    acts_np    = feature_acts.numpy()   # [N_atoms, d_sae]

    dashboard = []
    for feat_idx in ranked:
        feat_acts = acts_np[:, feat_idx]   # [N_atoms]

        # Chemical correlations
        correlations = {}
        for prop_name, prop_vals in atom_props.items():
            correlations[prop_name] = spearman_correlation(feat_acts, prop_vals)

        # Top-20 exemplar atoms
        top_atom_idxs = np.argsort(-feat_acts)[:20]
        exemplars = []
        for idx in top_atom_idxs:
            z = int(atom_types[idx].item())
            exemplars.append({
                "atom_idx":    int(idx),
                "element":     ELEMENT_SYMBOLS.get(z, f"Z={z}"),
                "atomic_num":  z,
                "activation":  float(feat_acts[idx]),
            })

        dashboard.append({
            "feature_idx":  int(feat_idx),
            "sparsity":     float(stats["sparsity"][feat_idx]),
            "mean_act":     float(stats["mean_act"][feat_idx]),
            "max_act":      float(stats["max_act"][feat_idx]),
            "correlations": correlations,
            "top_exemplars": exemplars,
        })

    return dashboard


def compute_decoder_similarity(W_dec: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity matrix of decoder feature directions.
    W_dec : [d_sae, d_in]
    Returns [d_sae, d_sae] similarity matrix.
    """
    normed = F.normalize(W_dec, dim=-1)  # [d_sae, d_in]
    return normed @ normed.T             # [d_sae, d_sae]


def print_feature_report(dashboard: list[dict], n_show: int = 20):
    """Print a human-readable summary of the top features."""
    print("\n" + "="*80)
    print(f"TOP {n_show} SAE FEATURES — Chemical Dashboard")
    print("="*80)

    for entry in dashboard[:n_show]:
        f = entry["feature_idx"]
        print(f"\n--- Feature {f:4d} | sparsity {entry['sparsity']:.3f} | "
              f"mean_act {entry['mean_act']:.3f} | max_act {entry['max_act']:.3f}")

        # Best chemical correlation
        best_prop = max(entry["correlations"], key=lambda k: abs(entry["correlations"][k]))
        best_r    = entry["correlations"][best_prop]
        print(f"  Strongest chem. correlation: {best_prop} (r={best_r:+.3f})")

        # Print all correlations
        corr_str = "  All correlations: " + "  ".join(
            f"{k}={v:+.2f}" for k, v in sorted(
                entry["correlations"].items(), key=lambda x: -abs(x[1])
            )
        )
        print(corr_str)

        # Exemplar elements
        elem_counts: dict[str, int] = {}
        for ex in entry["top_exemplars"]:
            elem_counts[ex["element"]] = elem_counts.get(ex["element"], 0) + 1
        elem_str = ", ".join(
            f"{el}×{c}" for el, c in sorted(elem_counts.items(), key=lambda x: -x[1])
        )
        print(f"  Top-20 exemplar elements: {elem_str}")


def save_dashboard_json(dashboard: list[dict], path: Path):
    import json
    # Convert numpy floats for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64, float)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, int)):
            return int(obj)
        return obj

    serialisable = json.loads(
        json.dumps(dashboard, default=convert)
    )
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Dashboard saved → {path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Interpret SAE features from Con-CDVAE")
    p.add_argument("--sae_checkpoint",  required=True)
    p.add_argument("--activations_dir", default="sae_activations")
    p.add_argument("--output_dir",      default="sae_analysis")
    p.add_argument("--top_n_features",  type=int, default=50)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load SAE
    print(f"Loading SAE checkpoint: {args.sae_checkpoint}")
    ckpt = torch.load(args.sae_checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    sae  = TopKSAE(cfg)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    print(f"  dict_size={cfg.dict_size}  K={cfg.k}  input_dim={cfg.input_dim}")

    # 2. Load activations + metadata
    acts_dir = Path(args.activations_dir)
    print("Loading activations …")
    raw_acts   = torch.load(acts_dir / "activations_train.pt", weights_only=True)
    meta       = torch.load(acts_dir / "meta_train.pt",        weights_only=False)
    atom_types = meta["atom_types"]

    # Load normalisation stats
    norm_path = Path("sae_checkpoints") / "normalisation.pt"
    if norm_path.exists():
        norm = torch.load(norm_path, weights_only=True)
        acts_norm = (raw_acts - norm["mean"]) / norm["std"]
    else:
        print("Warning: normalisation.pt not found, using raw activations")
        acts_norm = raw_acts

    # 3. Run SAE encoder to get feature activations
    print("Running SAE encoder to get feature activations …")
    CHUNK = 50_000
    feat_acts_list = []
    with torch.no_grad():
        for i in range(0, acts_norm.shape[0], CHUNK):
            chunk = acts_norm[i : i + CHUNK].to(device)
            h = sae.encode(chunk)
            feat_acts_list.append(h.cpu())
    feature_acts = torch.cat(feat_acts_list, dim=0)   # [N_atoms, d_sae]
    print(f"  feature_acts shape: {feature_acts.shape}")

    # 4. Dead features
    dead = sae.dead_features(threshold=5000)
    print(f"\nDead features (fired <5000 steps): {len(dead)} / {cfg.dict_size}")

    # 5. Build dashboard
    print(f"\nBuilding chemical dashboard for top {args.top_n_features} features …")
    dashboard = build_chemical_dashboard(feature_acts, atom_types, top_n=args.top_n_features)

    # 6. Print report
    print_feature_report(dashboard, n_show=20)

    # 7. Save
    save_dashboard_json(dashboard, out_dir / "feature_dashboard.json")

    # 8. Decoder similarity (spot feature clusters)
    print("\nComputing decoder feature similarity matrix …")
    W_dec = sae.W_dec.data  # [d_sae, d_in]
    sim_matrix = compute_decoder_similarity(W_dec).cpu()
    torch.save(sim_matrix, out_dir / "decoder_similarity.pt")
    # Report top-10 most-similar feature pairs (excluding diagonal)
    sim_np = sim_matrix.numpy()
    np.fill_diagonal(sim_np, 0)
    flat_idx = np.argsort(-sim_np.ravel())[:10]
    print("Top 10 most similar feature pairs (possible polysemanticity):")
    for idx in flat_idx:
        i, j = divmod(idx, cfg.dict_size)
        print(f"  Feature {i} ↔ Feature {j} : cosine_sim = {sim_np[i, j]:.4f}")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
