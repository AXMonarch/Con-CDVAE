# hooks.py — Hook registration and activation extraction
#
# Registers forward hooks on specific modules of the frozen Con-CDVAE model
# and collects activations during forward passes.
#
# Hook points and justifications:
#
#   1. fc_mu  (per-crystal, 256d)
#      The encoder's final answer after pooling + tanh + linear.
#      Purest encoder representation — no condition info mixed in.
#      If a property is readable here, the encoder learned it from
#      the crystal graph alone.
#
#   2. z_condition (per-crystal, 256d)  [output = z_con]
#      The fused representation of structure + condition.
#      Central bottleneck driving ALL downstream predictions.
#      Comparing probe accuracy here vs fc_mu tells us whether the
#      condition embedding helps predict non-conditioned properties.
#
#   3. condition_model (per-crystal, 128d)  [output = condition_emb]
#      Pure encoding of formation energy input.
#      NEGATIVE CONTROL: should predict formation_energy perfectly
#      and nothing else. If it predicts band_gap, that is correlation
#      in the data, not learned representation.
#
#   4. encoder.output_blocks[0] (per-atom, 256d)
#      First output block — reads initial atom embeddings before any
#      message passing. Baseline for what is knowable from element
#      identity + immediate bond distances alone.
#
#   5. encoder.output_blocks[4] (per-atom, 256d)
#      Last output block — reads atom representations after all 4
#      rounds of message passing. Comparing with output_blocks[0]
#      directly measures the value added by message passing.
#
# NOT hooked (and why):
#   - interaction_blocks: output (N_edges, 128), edge-level not atom-level
#   - decoder blocks: contaminated by random diffusion noise each pass
#   - z_var: encodes uncertainty, not crystal identity
#   - fusion intermediates: not interpretatively meaningful
#   - output heads: low-dim final predictions, probing them is circular

import torch
import torch.nn as nn
from dataclasses import dataclass, field


# ---- Hook point identifiers ------------------------------------------------

HOOK_POINTS = {
    "fc_mu":       {"level": "crystal", "dim": 256},
    "z_con":       {"level": "crystal", "dim": 256},
    "cond_emb":    {"level": "crystal", "dim": 128},
    "out_blk_0":   {"level": "atom",    "dim": 256},
    "out_blk_4":   {"level": "atom",    "dim": 256},
}


# ---- Hook manager -----------------------------------------------------------

class HookManager:
    """
    Registers forward hooks on a frozen Con-CDVAE model and collects
    activations into a dictionary keyed by hook point name.

    Usage
    -----
        manager = HookManager(model)
        manager.register_hooks()

        with torch.no_grad():
            model(batch, teacher_forcing=False, training=False)

        activations = manager.get_activations()   # dict[str, Tensor]
        manager.clear()                            # reset for next batch
        manager.remove_hooks()                     # cleanup when done
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[str, torch.Tensor] = {}

    # -- Registration ---------------------------------------------------------

    def register_hooks(self) -> None:
        """Attach forward hooks to all 5 probe points."""
        hook_targets = self._resolve_modules()
        for name, module in hook_targets.items():
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _resolve_modules(self) -> dict[str, nn.Module]:
        """Map hook point names to actual model sub-modules."""
        model = self.model
        return {
            "fc_mu":     model.fc_mu,
            "z_con":     model.z_condition,
            "cond_emb":  model.condition_model,
            "out_blk_0": model.encoder.output_blocks[0],
            "out_blk_4": model.encoder.output_blocks[4],
        }

    def _make_hook(self, name: str):
        """Create a closure that stores the module output in _activations."""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self._activations[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                # Some modules return tuples — take the first tensor
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self._activations[name] = item.detach().cpu()
                        break
        return hook_fn

    # -- Access ---------------------------------------------------------------

    def get_activations(self) -> dict[str, torch.Tensor]:
        """Return a copy of all captured activations."""
        return dict(self._activations)

    def clear(self) -> None:
        """Clear stored activations (call between batches)."""
        self._activations.clear()

    # -- Cleanup --------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __del__(self):
        self.remove_hooks()
