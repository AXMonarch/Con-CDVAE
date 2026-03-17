#
# Architecture reference (shapes from batch of 8 crystals, 74 atoms, 502 edges):
#
#   CONDITIONING PATHWAY (per-crystal)
#     condition_model                          -> (N_cryst, 128)   condition_emb
#
#   ENCODER - DimeNet++ (per-atom -> per-crystal)
#     encoder.output_blocks[0]                 -> (N_atoms, 256)   before message passing
#     encoder.interaction_blocks[0..3]         -> (N_edges, 128)   edge-level 
#     encoder.output_blocks[4]                 -> (N_atoms, 256)   after all message passing
#     scatter(P, batch, reduce='mean')         -> (N_cryst, 256)   hidden (inside encoder)
#     tanh(hidden)                             -> (N_cryst, 256)   smoothed
#
#   VAE BOTTLENECK (per-crystal)
#     fc_mu                                    -> (N_cryst, 256)   z_mu
#     fc_var                                   -> (N_cryst, 256)   z_var
#     z = mu + sigma * eps                     -> (N_cryst, 256)   z (sampled)
#
#   FUSION (per-crystal)
#     z_condition  (cat(z, cond) -> MLP)       -> (N_cryst, 256)   z_con
#
#   OUTPUT HEADS
#     fc_num_atoms                             -> (N_cryst, 21)
#     fc_lattice                               -> (N_cryst, 6)
#     fc_composition                           -> (N_atoms, 100)
#
#   DECODER - GemNet-T (per-atom, from noisy input)
#
# Probe matrix:
# -----------------------------------------------------------------------
#  Property               | fc_mu | z_con | cond_emb | out_blk_0 | out_blk_4
# -----------------------------------------------------------------------
#  formation_energy       |   R   |   R   |    R*    |    R+     |    R+
#  band_gap               |   R   |   R   |    R*    |    R+     |    R+
#  e_above_hull           |   R   |   R   |    R*    |    R+     |    R+
#  spacegroup             |   C   |   C   |    --    |    --     |    --
#  crystal_system         |   C   |   C   |    --    |    --     |    --
#  is_metal               |   C   |   C   |    C*    |    --     |    --
#  num_atoms              |   R   |   R   |    --    |    --     |    --
# -----------------------------------------------------------------------
#  R = regression probe, C = classification probe
#  R* / C* = negative control (should only predict formation_energy)
#  R+ = per-atom vectors mean-pooled to per-crystal before probing
#  -- = not applicable or not informative
#
# Modules:
#   hooks.py        - Hook registration and activation extraction
#   labels.py       - Label extraction from dataset + CSV
#   probes.py       - Probe model definitions (linear / MLP)
#   train_probes.py - Training loop and evaluation for probes
#   run_probes.py   - CLI entry point: extract activations, train, report
# =============================================================================

from .hooks import HookManager
from .labels import LabelExtractor
from .probes import LinearProbe, MLPProbe
