# eval_steer.py — Evaluate SAE feature steering on Con-CDVAE generation
#
# Runs generation with and without steering, then compares the two sets
# of generated crystals across validity, element distribution, property
# shifts, and composition metrics.
#
# Three evaluation modes:
#
#   single_feature  Zero all features except one, generate, check whether
#                   output chemistry matches the feature's label.
#
#   amplify         Baseline generation vs steered generation.  Compares
#                   element distribution shift, validity delta, and
#                   property predictions.
#
#   sweep           Sweep amplification scale from 0.5x to 5x, collect
#                   metrics at each scale to map the steering response.
#
# Usage
# -----
#   python -m src.interpretability.eval_steer \
#       --gen_config conf/gen/default.yaml \
#       --sae_path src/interpretability/outputs/sae/sae_z_con.pt \
#       --mode amplify --feature 3479 --scale 2.0

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
import pandas as pd
import re

# Add scripts/ to path for eval_utils imports
_script_dir = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_script_dir))

from eval_utils import (
    load_model,
    generation,
    get_crystals_list,
    smact_validity,
    structure_validity,
)
from concdvae.common.data_utils import chemical_symbols

from .sae_model import TopKSAE, SAEConfig
from .steer import (
    SteeringManager,
    SteeringConfig,
    SteerDirective,
    SteerOp,
    CLUSTER_TO_FEATURES,
)
from .mapping_exported import feature_to_cluster


# ── SAE loading ──────────────────────────────────────────────────────────────

def load_sae(sae_path: str | Path, device: str = "cpu") -> TopKSAE:
    """Load a trained SAE from checkpoint.

    Parameters
    ----------
    sae_path : str or Path
        Path to the saved SAE checkpoint file.
    device : str
        Device to load the SAE onto.

    Returns
    -------
    TopKSAE
        The loaded SAE in eval mode.
    """
    ckpt = torch.load(sae_path, map_location=device, weights_only=False)
    config: SAEConfig = ckpt["config"]
    sae = TopKSAE(config)
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def safe_get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms):
    num_atoms = num_atoms.squeeze()   # guard against (1,N) shape
    expected = num_atoms.sum().item()
    if frac_coords.size(0) == expected and atom_types.size(0) == expected:
        return get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    starts = torch.cat([torch.tensor([0]), torch.cumsum(num_atoms[:-1], dim=0)])
    valid = [
        i for i, (s, n) in enumerate(zip(starts, num_atoms))
        if (s + n) <= frac_coords.size(0) and (s + n) <= atom_types.size(0)
    ]
    print(f"  WARNING: dropping {len(num_atoms) - len(valid)} malformed samples")
    valid_t = torch.tensor(valid)
    new_num_atoms = num_atoms[valid_t]
    new_fc = torch.cat([frac_coords[starts[i]:starts[i]+num_atoms[i]] for i in valid])
    new_at = torch.cat([atom_types[starts[i]:starts[i]+num_atoms[i]] for i in valid])
    return get_crystals_list(new_fc, new_at, lengths[valid_t], angles[valid_t], new_num_atoms)

# ── Generation wrapper ───────────────────────────────────────────────────────

def run_generation(model, prior, prop_dict, gen_cfg, ld_kwargs):
    frac_coords, num_atoms, atom_types, lengths, angles, _, _ = generation(
        model, prior, prop_dict,
        batch_size=gen_cfg.batch_size,
        down_sample=gen_cfg.down_sample,
        num_batches_to_sample=gen_cfg.num_batches_to_samples,
        ld_kwargs=ld_kwargs,
    )
    # generation() stacks into (num_samples_per_z, total, ...) — take first and flatten
    return (
        frac_coords[0],
        atom_types[0],
        lengths[0],
        angles[0],
        num_atoms[0].squeeze(),   # was (1, N), now (N,)
    )
# ── Metric computation ───────────────────────────────────────────────────────

def compute_element_distribution(crystal_list: list[dict]) -> Counter:
    """Count element occurrences across all generated crystals.

    Parameters
    ----------
    crystal_list : list of dict
        Per-crystal dicts from ``get_crystals_list``, each with an
        ``"atom_types"`` key containing an integer array.

    Returns
    -------
    Counter
        Mapping element symbol to total count across all crystals.
    """
    counts: Counter = Counter()
    for cryst in crystal_list:
        for z in cryst["atom_types"]:
            z_int = int(z)
            if 0 < z_int < len(chemical_symbols):
                counts[chemical_symbols[z_int]] += 1
    return counts


def compute_validity(crystal_list: list[dict]) -> dict:
    """Compute SMACT and structural validity rates.

    Parameters
    ----------
    crystal_list : list of dict
        Per-crystal dicts from ``get_crystals_list``.

    Returns
    -------
    dict
        Keys: ``n_crystals``, ``smact_rate``, ``struct_rate``,
        ``both_rate``.
    """
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice

    n = len(crystal_list)
    smact_valid = 0
    struct_valid = 0
    both_valid = 0

    for cryst in crystal_list:
        atom_types = cryst["atom_types"]
        comp, count = np.unique(atom_types, return_counts=True)

        try:
            s_valid = smact_validity(comp, count)
        except Exception:
            s_valid = False

        st_valid = False
        try:
            lengths_arr = cryst["lengths"]
            angles_arr = cryst["angles"]
            lattice = Lattice.from_parameters(
                *(lengths_arr.tolist()), *(angles_arr.tolist()),
            )
            species = [chemical_symbols[int(z)] for z in atom_types]
            structure = Structure(
                lattice, species, cryst["frac_coords"],
                coords_are_cartesian=False,
            )
            st_valid = structure_validity(structure)
        except Exception:
            st_valid = False

        smact_valid += s_valid
        struct_valid += st_valid
        both_valid += (s_valid and st_valid)

    return {
        "n_crystals": n,
        "smact_rate": smact_valid / max(n, 1),
        "struct_rate": struct_valid / max(n, 1),
        "both_rate": both_valid / max(n, 1),
    }


def compute_num_atoms_stats(crystal_list: list[dict]) -> dict:
    """Statistics on number of atoms per crystal.

    Parameters
    ----------
    crystal_list : list of dict
        Per-crystal dicts from ``get_crystals_list``.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``min``, ``max``.
    """
    counts = [len(c["atom_types"]) for c in crystal_list]
    if not counts:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    arr = np.array(counts, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def compare_element_distributions(
    baseline: Counter,
    steered: Counter,
    top_n: int = 10,
) -> dict:
    """Compare two element frequency distributions.

    Parameters
    ----------
    baseline : Counter
        Element counts from baseline generation.
    steered : Counter
        Element counts from steered generation.
    top_n : int
        Number of top shifted elements to report.

    Returns
    -------
    dict
        Keys: ``kl_divergence``, ``top_enriched``, ``top_depleted``.
    """
    all_elements = sorted(set(baseline.keys()) | set(steered.keys()))
    if not all_elements:
        return {"kl_divergence": 0.0, "top_enriched": [], "top_depleted": []}

    # Normalize to probability distributions
    b_total = max(sum(baseline.values()), 1)
    s_total = max(sum(steered.values()), 1)
    eps = 1e-10

    # Compute per-element ratio (steered / baseline)
    ratios = {}
    for elem in all_elements:
        b_frac = baseline.get(elem, 0) / b_total
        s_frac = steered.get(elem, 0) / s_total
        if b_frac > eps:
            ratios[elem] = s_frac / b_frac
        elif s_frac > eps:
            ratios[elem] = float("inf")

    # KL divergence: D_KL(steered || baseline)
    kl = 0.0
    for elem in all_elements:
        p = steered.get(elem, 0) / s_total + eps
        q = baseline.get(elem, 0) / b_total + eps
        kl += p * np.log(p / q)

    # Top enriched / depleted
    sorted_ratios = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    top_enriched = sorted_ratios[:top_n]
    top_depleted = sorted_ratios[-top_n:][::-1]

    return {
        "kl_divergence": float(kl),
        "top_enriched": [(e, round(r, 2)) for e, r in top_enriched],
        "top_depleted": [(e, round(r, 2)) for e, r in top_depleted],
    }


def run_property_probes(
    z_con_tensors: list[torch.Tensor],
    probe_dir: str,
    hook_point: str = "z_con",
) -> dict[str, float]:
    """Run trained linear probes on captured z_con to predict properties.

    Parameters
    ----------
    z_con_tensors : list of torch.Tensor
        Captured z_con feature tensors from steering (one per batch).
    probe_dir : str
        Directory containing trained linear probe ``.pt`` files.
    hook_point : str
        Hook point name used in probe filenames.

    Returns
    -------
    dict of str -> float
        Mean predicted value for each property.
    """
    from .probes import LinearProbe

    probe_path = Path(probe_dir) / "probes" / "linear"
    if not probe_path.exists():
        return {}

    # Reconstruct z_con from SAE features would require decode —
    # instead we accept pre-decoded z_con tensors
    z_con = torch.cat(z_con_tensors, dim=0)

    results = {}
    for path in probe_path.glob(f"*_{hook_point}.pt"):
        prop_name = path.stem.replace(f"_{hook_point}", "")
        state = torch.load(path, map_location="cpu", weights_only=False)

        if "linear.weight" not in state:
            continue

        weight = state["linear.weight"]  # (out_dim, input_dim)
        bias = state.get("linear.bias", torch.zeros(weight.shape[0]))

        with torch.no_grad():
            preds = z_con.cpu() @ weight.T + bias
            if preds.shape[1] == 1:
                results[prop_name] = float(preds.mean())
            else:
                results[prop_name] = float(preds.argmax(dim=1).float().mean())

    return results


# ── Printing ─────────────────────────────────────────────────────────────────

def print_comparison_table(
    baseline_metrics: dict,
    steered_metrics: dict,
    config_desc: str,
) -> None:
    """Pretty-print side-by-side comparison of baseline vs steered.

    Parameters
    ----------
    baseline_metrics : dict
        Metrics dict for baseline generation.
    steered_metrics : dict
        Metrics dict for steered generation.
    config_desc : str
        Human-readable description of the steering configuration.
    """
    print(f"\n{'=' * 70}")
    print(f"Steering Evaluation: {config_desc}")
    print(f"{'=' * 70}")

    # Validity
    bv = baseline_metrics["validity"]
    sv = steered_metrics["validity"]
    print(f"\n{'Metric':<30s} {'Baseline':>12s} {'Steered':>12s} {'Delta':>12s}")
    print("-" * 70)
    print(f"{'N crystals':<30s} {bv['n_crystals']:>12d} {sv['n_crystals']:>12d}")
    for key, label in [
        ("smact_rate", "SMACT validity"),
        ("struct_rate", "Structure validity"),
        ("both_rate", "Both valid"),
    ]:
        b, s = bv[key], sv[key]
        print(f"{label:<30s} {b:>11.1%} {s:>12.1%} {s - b:>+11.1%}")

    # Num atoms
    bn = baseline_metrics["num_atoms"]
    sn = steered_metrics["num_atoms"]
    print(f"\n{'Num atoms (mean ± std)':<30s} "
          f"{bn['mean']:>5.1f}±{bn['std']:<5.1f} "
          f"{sn['mean']:>6.1f}±{sn['std']:<5.1f}")

    # Element distribution comparison
    comp = steered_metrics.get("comparison")
    if comp:
        print(f"\n{'KL divergence':<30s} {comp['kl_divergence']:>12.4f}")
        if comp["top_enriched"]:
            print(f"\nTop enriched elements (steered/baseline ratio):")
            for elem, ratio in comp["top_enriched"][:5]:
                ratio_str = f"{ratio:.2f}x" if ratio != float("inf") else "new"
                print(f"  {elem:<4s} {ratio_str}")
        if comp["top_depleted"]:
            print(f"\nTop depleted elements:")
            for elem, ratio in comp["top_depleted"][:5]:
                print(f"  {elem:<4s} {ratio:.2f}x")

    # Property predictions
    bp = baseline_metrics.get("property_preds", {})
    sp = steered_metrics.get("property_preds", {})
    if bp or sp:
        print(f"\n{'Property predictions (mean)':}")
        print(f"  {'Property':<25s} {'Baseline':>12s} {'Steered':>12s}")
        print(f"  {'-' * 55}")
        for prop in sorted(set(bp.keys()) | set(sp.keys())):
            b_val = f"{bp[prop]:>12.4f}" if prop in bp else f"{'—':>12s}"
            s_val = f"{sp[prop]:>12.4f}" if prop in sp else f"{'—':>12s}"
            print(f"  {prop:<25s} {b_val} {s_val}")

    print(f"\n{'=' * 70}\n")


def print_element_table(elem_dist: Counter, title: str, top_n: int = 15) -> None:
    """Print the top elements in a distribution.

    Parameters
    ----------
    elem_dist : Counter
        Element symbol to count mapping.
    title : str
        Title to print above the table.
    top_n : int
        Number of top elements to show.
    """
    total = sum(elem_dist.values())
    print(f"\n{title} (total atoms: {total})")
    print(f"  {'Element':<8s} {'Count':>8s} {'Fraction':>10s}")
    print(f"  {'-' * 30}")
    for elem, count in elem_dist.most_common(top_n):
        print(f"  {elem:<8s} {count:>8d} {count / max(total, 1):>10.1%}")


# ── Fast evaluation (output heads only, no generation) ───────────────────────

@torch.no_grad()
def fast_evaluate(model, z_con: torch.Tensor) -> dict:
    """Run z_con through model output heads without Langevin dynamics.

    Returns predicted composition distribution, num_atoms distribution,
    and lattice parameters — all from a single forward pass through the
    MLP heads.  Orders of magnitude faster than full generation.

    Parameters
    ----------
    model : nn.Module
        The Con-CDVAE model (frozen, eval mode).
    z_con : torch.Tensor
        Steered (or baseline) z_con of shape ``(batch, 256)``.

    Returns
    -------
    dict with keys:
        ``composition`` : Counter of element symbols
        ``num_atoms_mean`` : float
        ``num_atoms_std`` : float
        ``lattice_means`` : dict with a, b, c, alpha, beta, gamma
    """
    device = z_con.device

    # Predict num_atoms: (batch, max_atoms+1) logits → argmax
    num_atoms_logits = model.predict_num_atoms(z_con)
    num_atoms = num_atoms_logits.argmax(dim=-1)  # (batch,)
    num_atoms = num_atoms.clamp(min=1)  # avoid zero-atom crystals

    # Predict composition: z_con repeated per atom → (total_atoms, 100) logits
    z_per_atom = z_con.repeat_interleave(num_atoms, dim=0)
    comp_logits = model.fc_composition(z_per_atom)  # (total_atoms, MAX_ATOMIC_NUM)
    comp_probs = torch.softmax(comp_logits, dim=-1)

    # Convert to element distribution (argmax per atom, then count)
    pred_elements = comp_logits.argmax(dim=-1)  # (total_atoms,)
    elem_counts = Counter()
    for z_int in pred_elements.cpu().tolist():
        if 0 < z_int < len(chemical_symbols):
            elem_counts[chemical_symbols[z_int]] += 1

    # Also compute soft composition: mean probability per element
    mean_probs = comp_probs.mean(dim=0).cpu()  # (MAX_ATOMIC_NUM,)

    # Lattice parameters
    try:
        model.lattice_scaler.match_device(z_con)
        lat_pred = model.fc_lattice(z_con)
        lat_scaled = model.lattice_scaler.inverse_transform(lat_pred)
        lat_means = lat_scaled.mean(dim=0).cpu()
        lattice = {
            "a": float(lat_means[0]), "b": float(lat_means[1]),
            "c": float(lat_means[2]), "alpha": float(lat_means[3]),
            "beta": float(lat_means[4]), "gamma": float(lat_means[5]),
        }
    except Exception:
        lattice = {}

    na = num_atoms.float()
    return {
        "composition": elem_counts,
        "composition_probs": mean_probs,
        "num_atoms_mean": float(na.mean()),
        "num_atoms_std": float(na.std()),
        "lattice": lattice,
    }


def get_baseline_z_con(
    model, prior, prop_dict, gen_cfg, ld_kwargs, sae, device: str,
) -> torch.Tensor:
    """Get a batch of baseline z_con by running generation with a capture hook.

    Registers a passive hook on model.z_condition that captures z_con
    without modifying it. Returns the concatenated captured tensors.
    """
    captured = []

    def _capture_hook(module, input, output):
        captured.append(output.detach())
        return output

    handle = model.z_condition.register_forward_hook(_capture_hook)
    torch.manual_seed(42)
    run_generation(model, prior, prop_dict, gen_cfg, ld_kwargs)
    handle.remove()

    if captured:
        return torch.cat(captured, dim=0).to(device)
    raise RuntimeError("No z_con captured — generation may have failed")


# ── Feature activation diagnostics ────────────────────────────────────────────

@torch.no_grad()
def diagnose_feature_activations(
    model, sae: TopKSAE, z_con: torch.Tensor,
    clusters: list[str] | None = None,
) -> dict:
    """Check which labeled features actually fire for the given z_con.

    For each cluster, reports:
      - Per-feature activation rate (fraction of samples where feature is in top-k)
      - Per-feature mean activation (when active)
      - Element enrichment when cluster features are CLAMPed on
      - Whether enriched elements match the cluster name

    Parameters
    ----------
    model : nn.Module
        Con-CDVAE model (for fast_evaluate).
    sae : TopKSAE
        Trained SAE.
    z_con : torch.Tensor
        Baseline z_con of shape (batch, 256).
    clusters : list of str, optional
        Clusters to diagnose. Defaults to all clusters.

    Returns
    -------
    dict
        Per-cluster diagnostics.
    """
    device = z_con.device
    sae = sae.to(device)
    k = sae.config.k

    # Encode and get top-k mask
    h_pre = sae.encode(z_con)
    _, topk_idx = torch.topk(h_pre, k, dim=-1)  # (batch, k)

    # Set of all labeled feature indices
    all_labeled = set(feature_to_cluster.keys())

    # Global stats
    topk_set_per_sample = [set(row.tolist()) for row in topk_idx]
    labeled_in_topk = [len(s & all_labeled) for s in topk_set_per_sample]

    print(f"\n{'=' * 70}")
    print(f"FEATURE ACTIVATION DIAGNOSTICS (k={k}, {z_con.shape[0]} samples)")
    print(f"{'=' * 70}")
    print(f"  Total labeled features: {len(all_labeled)} / {sae.config.n_features}")
    print(f"  Labeled features in top-{k} per sample: "
          f"{np.mean(labeled_in_topk):.1f} ± {np.std(labeled_in_topk):.1f} "
          f"(range {min(labeled_in_topk)}-{max(labeled_in_topk)})")

    # Baseline composition for comparison
    baseline_fast = fast_evaluate(model, z_con)
    baseline_comp = baseline_fast["composition"]

    if clusters is None:
        clusters = list(CLUSTER_TO_FEATURES.keys())

    results = {}
    for cluster in clusters:
        feature_ids = CLUSTER_TO_FEATURES[cluster]

        # Per-feature activation rate and mean value
        feat_stats = []
        for fid in feature_ids:
            active_mask = torch.zeros(z_con.shape[0], dtype=torch.bool, device=device)
            for i, s in enumerate(topk_set_per_sample):
                if fid in s:
                    active_mask[i] = True
            rate = active_mask.float().mean().item()
            if active_mask.any():
                mean_val = h_pre[active_mask, fid].mean().item()
            else:
                mean_val = 0.0
            feat_stats.append((fid, rate, mean_val))

        # Sort by activation rate descending
        feat_stats.sort(key=lambda x: -x[1])
        cluster_rate = np.mean([s[1] for s in feat_stats])

        # CLAMP test: force all cluster features to median active value
        # to see what elements this cluster *would* produce
        median_val = np.median([s[2] for s in feat_stats if s[2] > 0]) if any(s[2] > 0 for s in feat_stats) else 1.0
        clamp_directives = [
            SteerDirective(fid, SteerOp.CLAMP, float(median_val))
            for fid in feature_ids
        ]
        clamp_manager = SteeringManager(
            sae, SteeringConfig(directives=clamp_directives), verbose=False,
        )
        z_clamped = clamp_manager.steer_z_con(z_con)
        clamped_fast = fast_evaluate(model, z_clamped)
        clamped_comp = clamped_fast["composition"]
        comp_diff = compare_element_distributions(baseline_comp, clamped_comp)

        # Check if enriched elements match cluster name
        label_elements = set()
        for sym in chemical_symbols[1:]:
            if sym in cluster.split("/") or sym in cluster:
                label_elements.add(sym)
        top_enriched = [e for e, _ in comp_diff.get("top_enriched", [])[:5]]
        selectivity = (
            sum(1 for e in top_enriched if e in label_elements) / max(len(top_enriched), 1)
        ) if label_elements else float("nan")

        # Print
        print(f"\n  Cluster: {cluster} ({len(feature_ids)} features)")
        print(f"    Avg activation rate: {cluster_rate:.1%}")
        print(f"    Per-feature breakdown:")
        for fid, rate, mean_val in feat_stats:
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            print(f"      {fid:5d}  {bar} {rate:5.1%}  (mean={mean_val:.4f})")
        print(f"    CLAMP test (force features to {median_val:.4f}):")
        print(f"      KL from baseline: {comp_diff['kl_divergence']:.4f}")
        print(f"      Top enriched: {top_enriched}")
        if label_elements:
            print(f"      Expected elements (from name): {label_elements}")
            print(f"      Selectivity: {selectivity:.0%}")
        else:
            print(f"      (no element symbols found in cluster name)")

        results[cluster] = {
            "feature_stats": feat_stats,
            "cluster_activation_rate": cluster_rate,
            "clamp_kl": comp_diff["kl_divergence"],
            "top_enriched": top_enriched,
            "selectivity": selectivity,
        }

    print(f"\n{'=' * 70}")
    return results


# ── Verification metrics ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_verification_metrics(
    model, sae, z_con_baseline: torch.Tensor,
    cluster_name: str, scales: list[float], k_values: list[int],
) -> dict:
    """Compute steering verification metrics for one cluster.

    Returns metrics that answer: "is steering this cluster actually working?"

    Checks
    ------
    1. Reconstruction fidelity: cosine sim(original, SAE-roundtripped z_con)
    2. Baseline identity: at scale=1.0 the composition should match unsteered
    3. Dose-response monotonicity: does effect grow with scale?
    4. Selectivity: does steering this cluster primarily shift its own elements?
    """
    device = z_con_baseline.device
    sae = sae.to(device)
    feature_ids = CLUSTER_TO_FEATURES[cluster_name]

    # 1. Reconstruction fidelity (no steering, just SAE roundtrip)
    noop_manager = SteeringManager(sae, SteeringConfig())
    z_con_recon = noop_manager.reconstruct_z_con(z_con_baseline)
    recon_cos = torch.nn.functional.cosine_similarity(
        z_con_baseline, z_con_recon, dim=-1,
    ).mean().item()

    # Baseline composition (no steering at all)
    baseline_fast = fast_evaluate(model, z_con_baseline)
    baseline_comp = baseline_fast["composition"]

    # 2 & 3. Sweep scale at training k, measure composition shift
    training_k = sae.config.k
    composition_by_scale = {}
    kl_by_scale = {}

    for scale in sorted(scales):
        manager = SteeringManager.from_cluster(
            sae, cluster_name, scale, k_override=training_k,
        )
        z_steered = manager.steer_z_con(z_con_baseline)
        result = fast_evaluate(model, z_steered)
        composition_by_scale[scale] = result["composition"]
        comp = compare_element_distributions(baseline_comp, result["composition"])
        kl_by_scale[scale] = comp["kl_divergence"]

    # Monotonicity: KL should increase with |scale - 1|
    sorted_scales = sorted(s for s in scales if s >= 1.0)
    kl_vals = [kl_by_scale.get(s, 0.0) for s in sorted_scales]
    monotonic = all(b >= a - 1e-6 for a, b in zip(kl_vals, kl_vals[1:]))

    # Baseline identity: KL at scale=1.0 should be near zero
    identity_kl = kl_by_scale.get(1.0, float("nan"))

    # 4. Selectivity: what fraction of the top-5 enriched elements at max scale
    #    are elements that appear in the cluster name?
    label_elements = set()
    for sym in chemical_symbols[1:]:  # skip empty string at index 0
        if sym in cluster_name.split("/") or sym in cluster_name:
            label_elements.add(sym)

    max_scale = max(scales)
    max_comp = compare_element_distributions(
        baseline_comp, composition_by_scale.get(max_scale, Counter()),
    )
    top_enriched = [e for e, _ in max_comp.get("top_enriched", [])[:5]]
    selectivity = (
        sum(1 for e in top_enriched if e in label_elements) / max(len(top_enriched), 1)
    )

    # k-sensitivity: does lowering k amplify the steering effect?
    kl_by_k = {}
    for k in k_values:
        manager = SteeringManager.from_cluster(
            sae, cluster_name, max_scale, k_override=k,
        )
        z_steered = manager.steer_z_con(z_con_baseline)
        result = fast_evaluate(model, z_steered)
        comp = compare_element_distributions(baseline_comp, result["composition"])
        kl_by_k[k] = comp["kl_divergence"]

    return {
        "reconstruction_cosine_sim": recon_cos,
        "identity_kl": identity_kl,
        "dose_response_monotonic": monotonic,
        "kl_by_scale": kl_by_scale,
        "kl_by_k": kl_by_k,
        "selectivity": selectivity,
        "top_enriched_at_max_scale": top_enriched,
        "label_elements": sorted(label_elements),
        "n_cluster_features": len(feature_ids),
    }


# ── Mode runners ─────────────────────────────────────────────────────────────

def run_single_feature_mode(args, model, prior, sae, prop_dict, gen_cfg, ld_kwargs):
    """Ablate all features except one, generate, check chemistry.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments (must have ``feature``).
    model, prior : nn.Module
        Con-CDVAE model and prior.
    sae : TopKSAE
        Trained SAE.
    prop_dict : dict
        Condition properties.
    gen_cfg : OmegaConf
        Generation config.
    ld_kwargs : SimpleNamespace
        Langevin dynamics kwargs.

    Returns
    -------
    dict
        Results including element distribution and validity.
    """
    feature_idx = args.feature
    cluster = feature_to_cluster.get(feature_idx, "unknown")

    print(f"\n{'=' * 70}")
    print(f"Single-feature test: feature {feature_idx}")
    print(f"  Cluster: {cluster}")
    print(f"{'=' * 70}")

    config = SteeringConfig(ablate_all_except=[feature_idx])
    manager = SteeringManager(sae, config)
    manager.enable_capture()
    manager.register(model)

    results = run_generation(model, prior, prop_dict, gen_cfg, ld_kwargs)
    manager.remove()

    crystal_list = safe_get_crystals_list(*results)
    elem_dist = compute_element_distribution(crystal_list)
    validity = compute_validity(crystal_list)
    num_atoms = compute_num_atoms_stats(crystal_list)

    print_element_table(elem_dist, f"Feature {feature_idx}: {cluster}")
    print(f"\nValidity:")
    print(f"  SMACT:     {validity['smact_rate']:.1%}")
    print(f"  Structure: {validity['struct_rate']:.1%}")
    print(f"  Both:      {validity['both_rate']:.1%}")
    print(f"  Num atoms: {num_atoms['mean']:.1f} ± {num_atoms['std']:.1f}")

    return {
        "mode": "single_feature",
        "feature_idx": feature_idx,
        "cluster": cluster,
        "element_distribution": dict(elem_dist),
        "validity": validity,
        "num_atoms": num_atoms,
        "n_generated": len(crystal_list),
    }


def run_amplify_mode(args, model, prior, sae, prop_dict, gen_cfg, ld_kwargs):
    """Baseline vs steered generation comparison.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments (one of ``feature``, ``cluster``, or ``property``
        must be set, plus ``scale``).
    model, prior : nn.Module
        Con-CDVAE model and prior.
    sae : TopKSAE
        Trained SAE.
    prop_dict : dict
        Condition properties.
    gen_cfg : OmegaConf
        Generation config.
    ld_kwargs : SimpleNamespace
        Langevin dynamics kwargs.

    Returns
    -------
    dict
        Baseline and steered metrics plus comparison.
    """
    # ── Baseline (no steering) ───────────────────────────────────────────
    print("\nRunning baseline generation (no steering)...")
    torch.manual_seed(args.seed)
    baseline_results = run_generation(model, prior, prop_dict, gen_cfg, ld_kwargs)
    baseline_crystals = safe_get_crystals_list(*baseline_results)

    # ── Build steering manager ───────────────────────────────────────────
    if args.cluster:
        config_desc = f"cluster '{args.cluster}' × {args.scale}"
        manager = SteeringManager.from_cluster(sae, args.cluster, args.scale)
    elif args.feature is not None:
        cluster = feature_to_cluster.get(args.feature, "unknown")
        config_desc = f"feature {args.feature} ({cluster}) × {args.scale}"
        manager = SteeringManager.from_features(sae, {args.feature: args.scale})
    elif args.property:
        config_desc = (
            f"property '{args.property}' {args.direction} "
            f"top-{args.top_n} × {args.scale}"
        )
        manager = SteeringManager.from_property(
            sae, args.property, args.direction,
            top_n=args.top_n, scale=args.scale,
            analysis_path=args.analysis_path,
        )
    else:
        raise ValueError("Must specify --feature, --cluster, or --property")

    # ── Steered generation ───────────────────────────────────────────────
    print(f"Running steered generation: {config_desc}...")
    manager.enable_capture()
    manager.register(model)
    torch.manual_seed(args.seed)  # same seed for fair comparison
    steered_results = run_generation(model, prior, prop_dict, gen_cfg, ld_kwargs)
    manager.remove()

    steered_crystals = safe_get_crystals_list(*steered_results)

    # ── Compute metrics ──────────────────────────────────────────────────
    baseline_elem = compute_element_distribution(baseline_crystals)
    steered_elem = compute_element_distribution(steered_crystals)
    comparison = compare_element_distributions(baseline_elem, steered_elem)

    baseline_metrics = {
        "validity": compute_validity(baseline_crystals),
        "num_atoms": compute_num_atoms_stats(baseline_crystals),
        "element_distribution": dict(baseline_elem),
    }
    steered_metrics = {
        "validity": compute_validity(steered_crystals),
        "num_atoms": compute_num_atoms_stats(steered_crystals),
        "element_distribution": dict(steered_elem),
        "comparison": comparison,
    }

    # Optional property probes
    if args.probe_dir and manager.get_captured_features():
        captured_z = manager.get_captured_features()
        # Decode captured features back to z_con space for probes
        with torch.no_grad():
            z_con_list = [sae.decode_denorm(h.to(sae.W_dec.device))
                          for h in captured_z]
        steered_metrics["property_preds"] = run_property_probes(
            z_con_list, args.probe_dir,
        )

    print_comparison_table(baseline_metrics, steered_metrics, config_desc)

    return {
        "mode": "amplify",
        "config_desc": config_desc,
        "baseline": baseline_metrics,
        "steered": steered_metrics,
    }


def run_sweep_mode(args, model, prior, sae, prop_dict, gen_cfg, ld_kwargs):
    scales = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    if args.scales:
        scales = [float(s) for s in args.scales.split(",")]

    if args.feature is not None:
        target_desc = f"feature {args.feature}"
        label = feature_to_cluster.get(args.feature, "unknown")
    elif args.cluster:
        target_desc = f"cluster '{args.cluster}'"
        label = args.cluster
    else:
        raise ValueError("Sweep mode requires --feature or --cluster")

    # Set up output dir here so per-scale saves work
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_name = args.feature or args.cluster or "unknown"

    print(f"\n{'=' * 70}")
    print(f"Sweep: {target_desc} ({label})")
    print(f"Scales: {scales}")
    print(f"{'=' * 70}")

    results_per_scale = {}

    for scale in scales:
        print(f"\n--- Scale {scale:.1f}x ---")

        if args.feature is not None:
            config = SteeringConfig(directives=[
                SteerDirective(args.feature, SteerOp.CLAMP, scale)
            ])
            manager = SteeringManager(sae, config)
        else:
            # Use AMPLIFY for clusters: scales relative to natural activation,
            # so inactive features stay at 0 and active ones scale proportionally.
            manager = SteeringManager.from_cluster(sae, args.cluster, scale)

        manager.register(model)
        torch.manual_seed(args.seed)
        gen_results = run_generation(model, prior, prop_dict, gen_cfg, ld_kwargs)
        manager.remove()

        crystal_list = safe_get_crystals_list(*gen_results)
        validity = compute_validity(crystal_list)
        elem_dist = compute_element_distribution(crystal_list)
        num_atoms = compute_num_atoms_stats(crystal_list)

        results_per_scale[scale] = {
            "validity": validity,
            "element_distribution": dict(elem_dist),
            "num_atoms": num_atoms,
            "n_crystals": len(crystal_list),
        }

        print(f"  Validity: SMACT={validity['smact_rate']:.1%}  "
              f"Struct={validity['struct_rate']:.1%}  "
              f"Both={validity['both_rate']:.1%}")
        print(f"  Num atoms: {num_atoms['mean']:.1f} ± {num_atoms['std']:.1f}")
        top_elems = elem_dist.most_common(5)
        top_str = ", ".join(f"{e}({c})" for e, c in top_elems)
        print(f"  Top elements: {top_str}")

        # Save this scale immediately
        # replace the filename block inside the for loop:
        scale_str = f"{scale:.1f}".replace(".", "_")
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(target_name))
        out_path = output_dir / f"steer_sweep_{safe_name}_scale{scale_str}.pt"

        torch.save({
            "mode": "sweep",
            "target": target_desc,
            "label": label,
            "scale": scale,
            "results": results_per_scale[scale],
        }, out_path)
        print(f"  Saved: {out_path.name}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"Sweep summary: {target_desc} ({label})")
    print(f"{'Scale':>8s} {'SMACT':>8s} {'Struct':>8s} {'Both':>8s} "
          f"{'N_atoms':>8s} {'Top element':>12s}")
    print("-" * 60)
    for scale in scales:
        r = results_per_scale[scale]
        v = r["validity"]
        top_elem = max(r["element_distribution"], key=r["element_distribution"].get,
                       default="—")
        print(f"{scale:>8.1f} {v['smact_rate']:>7.1%} {v['struct_rate']:>8.1%} "
              f"{v['both_rate']:>8.1%} {r['num_atoms']['mean']:>7.1f} "
              f"{top_elem:>12s}")
    print(f"{'=' * 70}\n")

    return {
        "mode": "sweep",
        "target": target_desc,
        "label": label,
        "scales": scales,
        "results": results_per_scale,
    }


# ── Grid mode ────────────────────────────────────────────────────────────────

def run_grid_mode(args, model, prior, sae, prop_dict, gen_cfg, ld_kwargs):
    """3D ablation grid: cluster × scale × k.

    Uses fast evaluation (output heads) for the full grid. Optionally runs
    full generation on selected points.
    """
    # Parse grid axes
    if args.clusters:
        clusters = [c.strip() for c in args.clusters.split(",")]
    else:
        clusters = list(CLUSTER_TO_FEATURES.keys())

    scales = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    if args.scales:
        scales = [float(s) for s in args.scales.split(",")]

    k_values = [8, 16, 32, 64, 128]
    if args.k_values:
        k_values = [int(k) for k in args.k_values.split(",")]

    # Validate clusters
    for c in clusters:
        if c not in CLUSTER_TO_FEATURES:
            available = list(CLUSTER_TO_FEATURES.keys())
            raise ValueError(f"Unknown cluster '{c}'. Available: {available}")

    n_cells = len(clusters) * len(scales) * len(k_values)
    print(f"\n{'=' * 70}")
    print(f"Grid ablation: {len(clusters)} clusters × {len(scales)} scales × {len(k_values)} k values = {n_cells} cells")
    print(f"Clusters: {clusters}")
    print(f"Scales:   {scales}")
    print(f"k values: {k_values}")
    print(f"{'=' * 70}")

    # Step 1: get baseline z_con
    print("\nCapturing baseline z_con...")
    z_con_baseline = get_baseline_z_con(
        model, prior, prop_dict, gen_cfg, ld_kwargs, sae, args.device,
    )
    print(f"  Captured z_con: {z_con_baseline.shape}")

    # Step 2: baseline fast evaluation (no steering)
    baseline_fast = fast_evaluate(model, z_con_baseline)
    baseline_comp = baseline_fast["composition"]
    print(f"  Baseline top elements: {baseline_comp.most_common(5)}")
    print(f"  Baseline num_atoms: {baseline_fast['num_atoms_mean']:.1f} ± {baseline_fast['num_atoms_std']:.1f}")

    # Step 2.5: feature activation diagnostics
    diag = diagnose_feature_activations(model, sae, z_con_baseline, clusters)

    # Step 3: verification metrics per cluster
    print("\nRunning verification checks...")
    verification = {}
    for cluster in clusters:
        print(f"  Verifying: {cluster}")
        v = compute_verification_metrics(
            model, sae, z_con_baseline, cluster, scales, k_values,
        )
        verification[cluster] = v
        print(f"    Recon cosine sim:  {v['reconstruction_cosine_sim']:.4f}")
        print(f"    Identity KL:      {v['identity_kl']:.6f}")
        print(f"    Dose-response OK: {v['dose_response_monotonic']}")
        print(f"    Selectivity:      {v['selectivity']:.2f}")

    # Step 4: full grid (fast path)
    print("\nRunning full grid (fast evaluation)...")
    grid = {}
    for ci, cluster in enumerate(clusters):
        grid[cluster] = {}
        feature_ids = CLUSTER_TO_FEATURES[cluster]
        for scale in scales:
            grid[cluster][scale] = {}
            for k in k_values:
                manager = SteeringManager.from_cluster(
                    sae, cluster, scale, k_override=k,
                )
                z_steered = manager.steer_z_con(z_con_baseline)
                result = fast_evaluate(model, z_steered)
                comp = compare_element_distributions(baseline_comp, result["composition"])
                grid[cluster][scale][k] = {
                    "composition": dict(result["composition"]),
                    "num_atoms_mean": result["num_atoms_mean"],
                    "num_atoms_std": result["num_atoms_std"],
                    "lattice": result["lattice"],
                    "kl_divergence": comp["kl_divergence"],
                    "top_enriched": comp["top_enriched"][:5],
                    "top_depleted": comp["top_depleted"][:5],
                }

        # Print per-cluster heatmap
        print(f"\n  Cluster: {cluster} ({len(feature_ids)} features)")
        print(f"  {'':>8s}", end="")
        for k in k_values:
            print(f"  k={k:<4d}", end="")
        print()
        for scale in scales:
            print(f"  s={scale:<5.1f}", end="")
            for k in k_values:
                kl = grid[cluster][scale][k]["kl_divergence"]
                print(f"  {kl:>6.4f}", end="")
            print()

    # Step 5: optional full generation on interesting points
    full_gen_results = {}
    if not args.fast_only:
        # Auto-select: highest KL point per cluster (most effect)
        print("\nRunning full generation on high-impact points...")
        for cluster in clusters:
            best_kl = 0.0
            best_point = (scales[-1], k_values[0])
            for scale in scales:
                for k in k_values:
                    kl = grid[cluster][scale][k]["kl_divergence"]
                    if kl > best_kl:
                        best_kl = kl
                        best_point = (scale, k)

            scale, k = best_point
            print(f"  {cluster}: scale={scale}, k={k} (KL={best_kl:.4f})")

            manager = SteeringManager.from_cluster(
                sae, cluster, scale, k_override=k, verbose=False,
            )
            manager.register(model)
            torch.manual_seed(args.seed)
            gen_results = run_generation(model, prior, prop_dict, gen_cfg, ld_kwargs)
            manager.remove()

            crystal_list = safe_get_crystals_list(*gen_results)
            validity = compute_validity(crystal_list)
            elem_dist = compute_element_distribution(crystal_list)
            num_atoms_stats = compute_num_atoms_stats(crystal_list)

            full_gen_results[cluster] = {
                "scale": scale, "k": k,
                "validity": validity,
                "element_distribution": dict(elem_dist),
                "num_atoms": num_atoms_stats,
                "n_crystals": len(crystal_list),
            }

            print(f"    Validity: SMACT={validity['smact_rate']:.1%}  "
                  f"Struct={validity['struct_rate']:.1%}  "
                  f"Both={validity['both_rate']:.1%}")
            top_str = ", ".join(f"{e}({c})" for e, c in elem_dist.most_common(5))
            print(f"    Top elements: {top_str}")

    # Summary
    print(f"\n{'=' * 70}")
    print("GRID ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Cluster':<40s} {'Recon':>6s} {'Ident':>7s} {'Mono':>5s} {'Select':>7s}")
    print("-" * 70)
    for cluster in clusters:
        v = verification[cluster]
        print(f"{cluster:<40s} {v['reconstruction_cosine_sim']:>6.3f} "
              f"{v['identity_kl']:>7.5f} "
              f"{'Y' if v['dose_response_monotonic'] else 'N':>5s} "
              f"{v['selectivity']:>6.2f}")
    print(f"{'=' * 70}\n")

    return {
        "mode": "grid",
        "clusters": clusters,
        "scales": scales,
        "k_values": k_values,
        "baseline": {
            "composition": dict(baseline_comp),
            "num_atoms_mean": baseline_fast["num_atoms_mean"],
            "num_atoms_std": baseline_fast["num_atoms_std"],
        },
        "grid": {
            c: {
                str(s): {
                    str(k): grid[c][s][k] for k in k_values
                } for s in scales
            } for c in clusters
        },
        "verification": verification,
        "full_generation": full_gen_results,
    }


# ── Model + generation setup ────────────────────────────────────────────────

def setup_generation(gen_cfg, device: str):
    """Load model, prior, conditions, and build ld_kwargs.

    Parameters
    ----------
    gen_cfg : OmegaConf
        Generation config loaded from YAML.
    device : str
        Device to place models on.

    Returns
    -------
    model : nn.Module
        The Con-CDVAE model in eval mode.
    prior : nn.Module
        The prior model in eval mode.
    prop_dict : dict
        Condition properties for the first input row.
    ld_kwargs : SimpleNamespace
        Langevin dynamics keyword arguments.
    """
    if str(gen_cfg.prior_path).lower() == "none":
        gen_cfg.prior_path = gen_cfg.model_path
    if str(gen_cfg.prior_label).lower() == "none":
        gen_cfg.prior_label = gen_cfg.prior_file.split("-epoch")[0]

    model, _, _ = load_model(gen_cfg.model_path, gen_cfg.model_file)
    prior, _, cfg_train = load_model(
        gen_cfg.prior_path, gen_cfg.prior_file,
        prior_label=gen_cfg.prior_label,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in prior.parameters():
        p.requires_grad = False
    model.eval()
    prior.eval()
    model.to(device)
    prior.to(device)

    # Build condition dict from input CSV
    need_props = [
        x.condition_name
        for x in cfg_train.prior.prior_model.conditionmodel.condition_embeddings
    ]
    input_path = os.path.join(gen_cfg.model_path, gen_cfg.input_path)
    input_data = pd.read_csv(input_path)

    # Use first row (or --input_row if we add it later)
    row_idx = 0
    prop_dict = {
        k: torch.tensor([input_data[k][row_idx]]).float().to(device)
        for k in input_data.columns
        if k in need_props
    }
    print(f"Conditions: {{{', '.join(f'{k}: {v.item():.4f}' for k, v in prop_dict.items())}}}")

    ld_kwargs = SimpleNamespace(
        num_samples_per_z=gen_cfg.num_samples_per_z,
        save_traj=False,
        down_sample_traj_step=gen_cfg.down_sample_traj_step,
        use_one=gen_cfg.use_one,
        disable_bar=gen_cfg.disable_bar,
        min_sigma=gen_cfg.min_sigma,
        step_lr=gen_cfg.step_lr,
        n_step_each=gen_cfg.n_step_each,
    )

    return model, prior, prop_dict, ld_kwargs


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    p = argparse.ArgumentParser(
        description="Evaluate SAE feature steering on crystal generation",
    )

    # Model / generation config
    p.add_argument(
        "--gen_config", type=str, required=True,
        help="Path to generation YAML config (e.g. conf/gen/default.yaml)",
    )
    p.add_argument(
        "--sae_path", type=str,
        default="src/interpretability/outputs/sae/sae_z_con.pt",
        help="Path to trained SAE checkpoint",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="src/interpretability/outputs/steering",
        help="Directory to save evaluation results",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=42)

    # Steering mode
    p.add_argument(
        "--mode", type=str, required=True,
        choices=["single_feature", "amplify", "sweep", "grid", "diagnose"],
        help="Steering evaluation mode",
    )

    # Target specification
    p.add_argument(
        "--feature", type=int, default=None,
        help="Feature index to steer",
    )
    p.add_argument(
        "--cluster", type=str, default=None,
        help="Cluster name to steer (from mapping_exported)",
    )
    p.add_argument(
        "--property", type=str, default=None,
        help="Property name for correlation-based steering",
    )
    p.add_argument(
        "--direction", type=str, default="increase",
        choices=["increase", "decrease"],
    )
    p.add_argument(
        "--top_n", type=int, default=10,
        help="Number of top-correlated features for property steering",
    )

    # Steering parameters
    p.add_argument(
        "--scale", type=float, default=2.0,
        help="Amplification scale (for amplify mode)",
    )
    p.add_argument(
        "--scales", type=str, default=None,
        help="Comma-separated scales for sweep mode (default: 0.5,1,1.5,2,3,5)",
    )

    # Grid mode parameters
    p.add_argument(
        "--clusters", type=str, default=None,
        help="Comma-separated cluster names for grid mode (default: all clusters)",
    )
    p.add_argument(
        "--k_values", type=str, default=None,
        help="Comma-separated k values for grid mode (default: 8,16,32,64,128)",
    )
    p.add_argument(
        "--fast_only", action="store_true",
        help="Grid mode: skip full generation, only run fast output-head evaluation",
    )

    # Optional
    p.add_argument(
        "--analysis_path", type=str, default=None,
        help="Path to analysis_z_con.pt (for property-based steering)",
    )
    p.add_argument(
        "--probe_dir", type=str, default=None,
        help="Directory with trained probes for property prediction",
    )
    p.add_argument(
        "--save_crystals", action="store_true",
        help="Include generated crystal tensors in saved output",
    )

    return p.parse_args()


def main():
    """Entry point for steering evaluation."""
    args = parse_args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load generation config
    gen_cfg = OmegaConf.load(args.gen_config)
    OmegaConf.resolve(gen_cfg)

    # Setup model, prior, conditions
    model, prior, prop_dict, ld_kwargs = setup_generation(gen_cfg, args.device)

    # Load SAE
    sae = load_sae(args.sae_path, args.device)
    print(f"Loaded SAE: {sae.config.n_features} features, k={sae.config.k}")

    # Dispatch to mode
    if args.mode == "single_feature":
        if args.feature is None:
            raise ValueError("--feature is required for single_feature mode")
        results = run_single_feature_mode(
            args, model, prior, sae, prop_dict, gen_cfg, ld_kwargs,
        )
    elif args.mode == "amplify":
        results = run_amplify_mode(
            args, model, prior, sae, prop_dict, gen_cfg, ld_kwargs,
        )
    elif args.mode == "sweep":
        results = run_sweep_mode(
            args, model, prior, sae, prop_dict, gen_cfg, ld_kwargs,
        )
    elif args.mode == "grid":
        results = run_grid_mode(
            args, model, prior, sae, prop_dict, gen_cfg, ld_kwargs,
        )
    elif args.mode == "diagnose":
        # Capture baseline z_con, then run diagnostics
        print("\nCapturing baseline z_con...")
        z_con_baseline = get_baseline_z_con(
            model, prior, prop_dict, gen_cfg, ld_kwargs, sae, args.device,
        )
        print(f"  Captured z_con: {z_con_baseline.shape}")
        clusters = None
        if args.clusters:
            clusters = [c.strip() for c in args.clusters.split(",")]
        results = diagnose_feature_activations(
            model, sae, z_con_baseline, clusters,
        )

    # Save results
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("grid", "diagnose"):
        safe_target = args.mode
    else:
        safe_target = re.sub(r'[^a-zA-Z0-9_]', '_',
            str(args.feature or args.cluster or args.property or "unknown"))
    out_path = output_dir / f"steer_{args.mode}_{safe_target}.pt"
    torch.save(results, out_path)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
