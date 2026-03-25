# steer.py — SAE-based model steering for Con-CDVAE generation
#
# Intercepts z_con (the fused structure + condition bottleneck) during
# generation, decomposes it through the trained SAE, modifies specific
# feature activations, and reconstructs a steered z_con.
#
# Usage
# -----
#   from src.interpretability.steer import SteeringManager, SteeringConfig, SteerDirective, SteerOp
#
#   # Amplify a single feature
#   manager = SteeringManager.from_features(sae, {3479: 3.0})
#   manager.register(model)
#   samples = model.langevin_dynamics(z_con, ld_kwargs)  # z_con is steered
#   manager.remove()
#
#   # Amplify an entire cluster
#   manager = SteeringManager.from_cluster(sae, "Chalcogenides", scale=2.0)
#
#   # Ablate all features except one (single-feature test)
#   config = SteeringConfig(ablate_all_except=[3479])
#   manager = SteeringManager(sae, config)
#
#   # Context manager
#   with SteeringManager.from_features(sae, {3479: 2.0}) as mgr:
#       mgr.register(model)
#       ...

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .sae_model import TopKSAE, SAEConfig
from .mapping_exported import feature_to_cluster


# ── Steering primitives ──────────────────────────────────────────────────────

class SteerOp(Enum):
    """Operation to apply to a single SAE feature.

    Attributes
    ----------
    AMPLIFY : str
        Multiply the feature activation by a scalar value.
    SUPPRESS : str
        Zero out the feature activation entirely.
    CLAMP : str
        Force the feature activation to a fixed value.
    """
    AMPLIFY = "amplify"
    SUPPRESS = "suppress"
    CLAMP = "clamp"


@dataclass
class SteerDirective:
    """A single steering instruction for one feature.

    Parameters
    ----------
    feature_idx : int
        Index of the SAE feature to modify (0 to n_features-1).
    op : SteerOp
        Which steering operation to apply.
    value : float
        Scale factor for AMPLIFY, target value for CLAMP.
        Unused for SUPPRESS.
    """
    feature_idx: int
    op: SteerOp
    value: float = 1.0


@dataclass
class SteeringConfig:
    """Full steering specification.

    If ``ablate_all_except`` is set, ALL features not in the list are
    zeroed before any per-feature directives are applied.  This enables
    single-feature isolation tests.

    Parameters
    ----------
    directives : list of SteerDirective
        Per-feature steering instructions.
    ablate_all_except : list of int or None
        If set, zero all features except these indices before applying
        directives.
    k_override : int or None
        If set, override the SAE's trained top-k value at inference time.
        Lower k = fewer active features = each steered feature has
        proportionally more influence.  None uses the SAE's training k.
    """
    directives: list[SteerDirective] = field(default_factory=list)
    ablate_all_except: list[int] | None = None
    k_override: int | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_cluster_to_features() -> dict[str, list[int]]:
    """Build reverse mapping from cluster name to feature indices.

    Returns
    -------
    dict of str -> list of int
        Cluster name to list of feature indices belonging to that cluster.
    """
    result: dict[str, list[int]] = defaultdict(list)
    for fid, cluster in feature_to_cluster.items():
        result[cluster].append(fid)
    return dict(result)


CLUSTER_TO_FEATURES = _build_cluster_to_features()


# ── SteeringManager ─────────────────────────────────────────────────────────

class SteeringManager:
    """Attach a modifying forward hook to ``model.z_condition`` that steers
    generation by intervening on SAE feature activations.

    The hook is stateless per-call: it encodes z_con through the SAE,
    applies steering directives, decodes back, and returns the steered
    z_con.  No mutable state accumulates between forward passes.

    Parameters
    ----------
    sae : TopKSAE
        Trained SAE model (must have normalization buffers set).
    config : SteeringConfig
        Specification of which features to modify and how.
    """

    def __init__(self, sae: TopKSAE, config: SteeringConfig, verbose: bool = True):
        self.sae = sae
        self.config = config
        self.verbose = verbose
        self._handle: torch.utils.hooks.RemovableHook | None = None
        self._captured_features: list[torch.Tensor] = []
        self._capture = False

    # ── Factory constructors ─────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        sae_path: str | Path,
        config: SteeringConfig,
        device: str = "cpu",
    ) -> SteeringManager:
        """Load SAE from checkpoint and wrap with a steering config.

        Parameters
        ----------
        sae_path : str or Path
            Path to the saved SAE checkpoint (``sae_z_con.pt``).
        config : SteeringConfig
            Steering specification to apply.
        device : str
            Device to load the SAE onto.

        Returns
        -------
        SteeringManager
        """
        ckpt = torch.load(sae_path, map_location=device, weights_only=False)
        sae_config: SAEConfig = ckpt["config"]
        sae = TopKSAE(sae_config)
        sae.load_state_dict(ckpt["state_dict"])
        sae.to(device)
        sae.eval()
        return cls(sae, config)

    @classmethod
    def from_features(
        cls,
        sae: TopKSAE,
        feature_multipliers: dict[int, float],
        k_override: int | None = None,
        verbose: bool = True,
    ) -> SteeringManager:
        """Create a manager from a dict of feature multipliers.

        Parameters
        ----------
        sae : TopKSAE
            Trained SAE model.
        feature_multipliers : dict of int -> float
            Feature index to scale factor.  Values > 1 amplify, 0 suppresses,
            values between 0 and 1 dampen.
        k_override : int or None
            Override SAE top-k at inference time.
        verbose : bool
            Print per-hook-call diagnostics.

        Returns
        -------
        SteeringManager
        """
        directives = []
        for idx, scale in feature_multipliers.items():
            if scale == 0.0:
                directives.append(SteerDirective(idx, SteerOp.SUPPRESS))
            else:
                directives.append(SteerDirective(idx, SteerOp.AMPLIFY, scale))
        return cls(sae, SteeringConfig(directives=directives, k_override=k_override), verbose=verbose)

    @classmethod
    def from_cluster(
        cls,
        sae: TopKSAE,
        cluster_name: str,
        scale: float,
        k_override: int | None = None,
        verbose: bool = True,
    ) -> SteeringManager:
        """Apply uniform scaling to all labeled features in a cluster.

        Parameters
        ----------
        sae : TopKSAE
            Trained SAE model.
        cluster_name : str
            Name of the feature cluster (from ``mapping_exported``).
        scale : float
            Multiplier to apply to every feature in the cluster.

        Returns
        -------
        SteeringManager

        Raises
        ------
        ValueError
            If ``cluster_name`` is not found in the cluster mapping.
        """
        if cluster_name not in CLUSTER_TO_FEATURES:
            available = list(CLUSTER_TO_FEATURES.keys())
            raise ValueError(
                f"Unknown cluster '{cluster_name}'. Available: {available}"
            )
        feature_ids = CLUSTER_TO_FEATURES[cluster_name]
        multipliers = {fid: scale for fid in feature_ids}
        return cls.from_features(sae, multipliers, k_override=k_override, verbose=verbose)

    @classmethod
    def from_property(
        cls,
        sae: TopKSAE,
        property_name: str,
        direction: str = "increase",
        top_n: int = 10,
        scale: float = 2.0,
        correlations: dict[str, np.ndarray] | None = None,
        analysis_path: str | Path | None = None,
    ) -> SteeringManager:
        """Steer using pre-computed property correlations.

        Selects the ``top_n`` SAE features most correlated (or
        anti-correlated) with the target property and amplifies them.

        Parameters
        ----------
        sae : TopKSAE
            Trained SAE model.
        property_name : str
            Property to steer toward/away from (e.g. ``"band_gap"``).
        direction : {"increase", "decrease"}
            ``"increase"`` selects features with highest positive
            correlation; ``"decrease"`` selects most negative.
        top_n : int
            Number of top-correlated features to steer.
        scale : float
            Amplification factor to apply.
        correlations : dict of str -> ndarray, optional
            Pre-loaded correlations mapping property name to an array of
            shape ``(n_features,)``.  Loaded from ``analysis_path`` if
            not provided.
        analysis_path : str or Path, optional
            Path to ``analysis_z_con.pt`` containing saved correlations.

        Returns
        -------
        SteeringManager

        Raises
        ------
        ValueError
            If neither ``correlations`` nor ``analysis_path`` is given,
            or if ``property_name`` is not in the correlations dict.
        """
        if correlations is None:
            if analysis_path is None:
                raise ValueError(
                    "Provide either correlations dict or analysis_path"
                )
            results = torch.load(analysis_path, map_location="cpu", weights_only=False)
            correlations = results["correlations"]

        if property_name not in correlations:
            available = list(correlations.keys())
            raise ValueError(
                f"Property '{property_name}' not in correlations. "
                f"Available: {available}"
            )

        corr = correlations[property_name]
        if isinstance(corr, torch.Tensor):
            corr = corr.numpy()

        if direction == "increase":
            top_indices = np.argsort(corr)[-top_n:][::-1]
        elif direction == "decrease":
            top_indices = np.argsort(corr)[:top_n]
        else:
            raise ValueError(
                f"direction must be 'increase' or 'decrease', got '{direction}'"
            )

        multipliers = {int(idx): scale for idx in top_indices}
        return cls.from_features(sae, multipliers)

    # ── Hook lifecycle ───────────────────────────────────────────────────

    def register(self, model: nn.Module) -> None:
        """Attach the modifying hook to ``model.z_condition``.

        Parameters
        ----------
        model : nn.Module
            The Con-CDVAE model.  Must have a ``z_condition`` sub-module.

        Raises
        ------
        RuntimeError
            If a hook is already registered.
        """
        if self._handle is not None:
            raise RuntimeError(
                "Steering hook already registered. Call remove() first."
            )

        # Move SAE to same device as model
        device = next(model.parameters()).device
        self.sae = self.sae.to(device)

        self._handle = model.z_condition.register_forward_hook(
            self._make_steering_hook()
        )

    def remove(self) -> None:
        """Remove the hook from the model."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ── Core hook ────────────────────────────────────────────────────────

    @staticmethod
    def _topk_with_k(h: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k masking with an explicit k value."""
        topk_vals, topk_idx = torch.topk(h, k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return h * mask

    def _make_steering_hook(self):
        sae = self.sae
        manager = self
        call_count = [0]

        def hook_fn(module, input, output):
            call_count[0] += 1
            z_con = output

            h_pre = sae.encode(z_con)
            k = manager.config.k_override or sae.config.k
            h_sparse = manager._topk_with_k(h_pre, k)

            if manager.verbose and manager.config.directives:
                d = manager.config.directives[0]
                active = (h_sparse[:, d.feature_idx] > 0).float().mean()
                val_before = h_sparse[:, d.feature_idx].mean()
                h_temp = manager._apply_steering(h_sparse)
                val_after = h_temp[:, d.feature_idx].mean()
                print(
                    f"  [hook call {call_count[0]}] "
                    f"feat {d.feature_idx}: "
                    f"active={active:.2f} "
                    f"before={val_before:.4f} "
                    f"after={val_after:.4f}"
                    f"  (k={k})"
                )

            h_steered = manager._apply_steering(h_sparse)
            z_con_steered = sae.decode_denorm(h_steered)

            if manager._capture:
                manager._captured_features.append(h_steered.detach().cpu())

            return z_con_steered

        return hook_fn

    def _apply_steering(self, h_sparse: torch.Tensor) -> torch.Tensor:
        """Apply all steering directives to sparse feature activations.

        If ``ablate_all_except`` is set, first zeros all features not in
        the whitelist.  Then applies individual per-feature directives.

        Parameters
        ----------
        h_sparse : torch.Tensor
            Sparse feature activations of shape ``(batch, n_features)``
            after top-k masking.

        Returns
        -------
        torch.Tensor
            Modified feature activations, same shape as input.
        """
        h = h_sparse.clone()

        # Ablation mask: keep only whitelisted features
        if self.config.ablate_all_except is not None:
            mask = torch.zeros(h.shape[1], device=h.device)
            for idx in self.config.ablate_all_except:
                mask[idx] = 1.0
            h = h * mask.unsqueeze(0)

        # Per-feature directives
        for d in self.config.directives:
            if d.op == SteerOp.AMPLIFY:
                h[:, d.feature_idx] *= d.value
            elif d.op == SteerOp.SUPPRESS:
                h[:, d.feature_idx] = 0.0
            elif d.op == SteerOp.CLAMP:
                h[:, d.feature_idx] = d.value

        return h

    # ── Offline steering (no hook, for fast evaluation) ─────────────────

    @torch.no_grad()
    def steer_z_con(self, z_con: torch.Tensor) -> torch.Tensor:
        """Apply steering to a z_con tensor offline (no hook needed).

        Encodes through SAE, applies top-k (with optional k_override),
        applies steering directives, decodes back to z_con space.

        Parameters
        ----------
        z_con : torch.Tensor
            Raw z_con tensor of shape ``(batch, 256)``.

        Returns
        -------
        torch.Tensor
            Steered z_con, same shape as input.
        """
        device = z_con.device
        sae = self.sae.to(device)
        h_pre = sae.encode(z_con)
        k = self.config.k_override or sae.config.k
        h_sparse = self._topk_with_k(h_pre, k)
        h_steered = self._apply_steering(h_sparse)
        return sae.decode_denorm(h_steered)

    @torch.no_grad()
    def reconstruct_z_con(self, z_con: torch.Tensor) -> torch.Tensor:
        """Encode and decode z_con through the SAE without steering.

        Useful for measuring reconstruction fidelity (the no-op baseline).
        """
        device = z_con.device
        sae = self.sae.to(device)
        h_pre = sae.encode(z_con)
        k = self.config.k_override or sae.config.k
        h_sparse = self._topk_with_k(h_pre, k)
        return sae.decode_denorm(h_sparse)

    # ── Diagnostics ──────────────────────────────────────────────────────

    def enable_capture(self) -> None:
        """Start capturing steered feature vectors for later analysis."""
        self._capture = True

    def disable_capture(self) -> None:
        """Stop capturing steered feature vectors."""
        self._capture = False

    def get_captured_features(self) -> list[torch.Tensor]:
        """Return list of captured feature tensors.

        Returns
        -------
        list of torch.Tensor
            One tensor of shape ``(batch, n_features)`` per hook call.
        """
        return list(self._captured_features)

    def clear_captured(self) -> None:
        """Clear accumulated captured features."""
        self._captured_features.clear()

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> SteeringManager:
        return self

    def __exit__(self, *args) -> None:
        self.remove()

    def __repr__(self) -> str:
        n_directives = len(self.config.directives)
        ablate = self.config.ablate_all_except
        parts = [f"SteeringManager({n_directives} directives"]
        if ablate is not None:
            parts.append(f", ablate_all_except={len(ablate)} features")
        parts.append(")")
        return "".join(parts)
