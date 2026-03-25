# src/interpretability/ — MechInterp + XAI for Con-CDVAE crystal generation
#
# Makes the latent space interpretable and steerable so users can
# understand what the model encodes and directly influence generation.
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
#     z_condition  (cat(z, cond) -> MLP)       -> (N_cryst, 256)   z_con  ← SAE trained here
#
#   OUTPUT HEADS (from z_con)
#     fc_num_atoms                             -> (N_cryst, 21)
#     fc_lattice                               -> (N_cryst, 6)
#     fc_composition                           -> (N_atoms, 100)
#
#   DECODER - GemNet-T (per-atom, from noisy input via Langevin dynamics)
#
# Pipeline:
#   1. hooks.py + extract.py   — Hook model, extract activations at 5 probe points
#   2. labels.py               — Extract ground-truth labels from dataset + CSV
#   3. probes.py + train_probes.py + run_probes.py
#                              — Train linear/MLP probes per (hook, property) pair
#   4. sae_model.py + train_sae.py + run_sae.py
#                              — Train Normalized Top-K SAE on z_con
#   5. analyse_sae.py          — Post-hoc feature analysis: stats, correlations,
#                                co-occurrence, decoder similarity, element enrichment
#   6. mapping_exported.py     — Feature labels and cluster assignments
#   7. steer.py                — SAE-based steering: modify feature activations
#                                during generation via hooks or offline
#   8. eval_steer.py           — Evaluate steering: single-feature, amplify, sweep,
#                                and grid (cluster × scale × k) ablation modes
# =============================================================================

from .hooks import HookManager
from .labels import LabelExtractor
from .probes import LinearProbe, MLPProbe
from .sae_model import TopKSAE, SAEConfig
from .steer import SteeringConfig, SteeringManager, SteerDirective, SteerOp
