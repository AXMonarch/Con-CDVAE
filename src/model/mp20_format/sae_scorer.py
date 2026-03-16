import torch
import numpy as np


def importance_scores(activations: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Compute composite per-dimension importance scores for an activation matrix.

    Args:
        activations: (N, D) tensor of activations across N crystals
        threshold:   activation magnitude threshold for occurrence scoring

    Returns:
        composite: (D,) tensor of importance scores, higher = more important
    """
    # 1. Variance score — dims doing global discriminative work
    var_score = activations.var(dim=0)                              # (D,)

    # 2. Occurrence score — how often each dim fires above threshold
    occ_score = (activations.abs() > threshold).float().mean(dim=0) # (D,)

    # 3. Max-over-dataset score — catches rare but high-impact dims
    max_score = activations.abs().max(dim=0).values                 # (D,)

    def norm(x: torch.Tensor) -> torch.Tensor:
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    composite = norm(var_score) + norm(occ_score) + norm(max_score)
    return composite


def select_top_dims(
    data: dict,
    K: int = 150,
    threshold: float = 0.1,
    save_path_filtered: str = None,
    save_path_dims: str = None,
) -> tuple[dict, dict]:
    """
    Run importance scoring across all five activation vectors,
    select top-K dims for each, and optionally save filtered dataset.

    Args:
        data:                dict with keys z_mu, z_var, z, cond_emb, z_cond,
                             formation_energy (as returned by activation collection)
        K:                   number of top dimensions to keep per vector
        threshold:           occurrence threshold passed to importance_scores
        save_path_filtered:  if provided, saves filtered dataset to this path
        save_path_dims:      if provided, saves top_dims dict to this path

    Returns:
        filtered:  dict of filtered activation tensors + formation_energy
        top_dims:  dict mapping vector name -> selected dimension indices
    """
    scores   = {}
    top_dims = {}

    for key in ['z_mu', 'z_var', 'z', 'z_cond']:
        scores[key]   = importance_scores(data[key], threshold=threshold)
        top_dims[key] = scores[key].topk(K).indices
        print(
            f"{key:10s}  top-{K} dims selected  "
            f"min score: {scores[key][top_dims[key]].min():.4f}  "
            f"max score: {scores[key][top_dims[key]].max():.4f}"
        )

    # cond_emb is only 128d — use all dims
    top_dims['cond_emb'] = torch.arange(data['cond_emb'].shape[1])
    print(f"{'cond_emb':10s}  all {data['cond_emb'].shape[1]} dims selected (no filtering)")

    # build filtered dataset
    filtered = {k: data[k][:, top_dims[k]] for k in top_dims}
    filtered['formation_energy'] = data['formation_energy']

    if save_path_filtered:
        torch.save(filtered, save_path_filtered)
        print(f"Filtered dataset saved → {save_path_filtered}")

    if save_path_dims:
        torch.save(top_dims, save_path_dims)
        print(f"Top dims saved        → {save_path_dims}")

    return filtered, top_dims