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
from .mapping_exported import feature_labels, feature_to_cluster


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
    label = feature_labels.get(feature_idx, "unlabeled")
    cluster = feature_to_cluster.get(feature_idx, "unknown")

    print(f"\n{'=' * 70}")
    print(f"Single-feature test: feature {feature_idx}")
    print(f"  Label:   {label}")
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

    print_element_table(elem_dist, f"Feature {feature_idx}: {label}")
    print(f"\nValidity:")
    print(f"  SMACT:     {validity['smact_rate']:.1%}")
    print(f"  Structure: {validity['struct_rate']:.1%}")
    print(f"  Both:      {validity['both_rate']:.1%}")
    print(f"  Num atoms: {num_atoms['mean']:.1f} ± {num_atoms['std']:.1f}")

    return {
        "mode": "single_feature",
        "feature_idx": feature_idx,
        "label": label,
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
        label = feature_labels.get(args.feature, "unlabeled")
        config_desc = f"feature {args.feature} ({label}) × {args.scale}"
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
        label = feature_labels.get(args.feature, "unlabeled")
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
            feature_ids = CLUSTER_TO_FEATURES[args.cluster]
            config = SteeringConfig(directives=[
                SteerDirective(fid, SteerOp.CLAMP, scale)
                for fid in feature_ids
            ])
            manager = SteeringManager(sae, config)

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
        choices=["single_feature", "amplify", "sweep"],
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

    # Save results
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)


    safe_target = re.sub(r'[^a-zA-Z0-9_]', '_', 
        str(args.feature or args.cluster or args.property or "unknown"))
    out_path = output_dir / f"steer_{args.mode}_{safe_target}.pt"
    torch.save(results, out_path)
    print(f"Saved results to {out_path}")

    target_name = args.feature or args.cluster or args.property or "unknown"
    out_path = output_dir / f"steer_{args.mode}_{target_name}.pt"
    torch.save(results, out_path)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
