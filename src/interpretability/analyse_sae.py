# analyse_sae.py — Post-hoc analysis of trained SAE features
#
# After training, each of the n_features learned directions in the SAE
# dictionary needs to be understood. This module provides:
#
#   1. Activation statistics: fire rate, mean/max magnitude, full histograms
#   2. Property correlations: Pearson r between features and ground-truth labels
#   3. Top exemplars: which crystals maximally activate each feature
#   4. Feature co-occurrence: which features fire together
#   5. Decoder similarity: cosine similarity between decoder weight columns
#   6. Element enrichment: which elements are over-represented in top exemplars
#   7. Probe alignment: cosine similarity between SAE directions and probe weights
#   8. Feature dashboard: unified summary table for manual inspection
#   9. Variance explained: how much reconstruction variance a feature subset captures
#
# These analyses let us label features post-hoc and identify which ones
# correspond to physically meaningful properties.
#
# Note: the SAE normalizes inputs internally. Analysis methods that compare
# reconstructions against raw inputs use decode_denorm() to undo the
# normalization. Feature activations (H) are unaffected by normalization
# — they live in the SAE's own latent space.

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

from .sae_model import TopKSAE


# ---- Periodic table lookup (atomic number -> symbol) -------------------------
# Only need the first 100 elements (matches Con-CDVAE's 100-class atom types)

ELEMENT_SYMBOLS = [
    "",  # 0 = padding
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
]


class SAEAnalyser:
    """
    Analyses a trained SAE's learned features against crystal properties.

    Parameters
    ----------
    sae : TopKSAE
        A trained SAE model (with normalization stats already set).
    device : str
        "cuda" or "cpu".
    batch_size : int
        Batch size for encoding the dataset through the SAE.
    """

    def __init__(
        self,
        sae: TopKSAE,
        device: str = "cpu",
        batch_size: int = 4096,
    ):
        self.sae = sae
        self.device = torch.device(device)
        self.batch_size = batch_size

    # -- Encode full dataset through SAE ----------------------------------------

    @torch.no_grad()
    def encode_dataset(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode a full dataset through the SAE, returning sparse activations.

        Parameters
        ----------
        X : Tensor (N, input_dim) — raw (unnormalized) activations

        Returns
        -------
        H : Tensor (N, n_features) — sparse activations (mostly zeros)
        """
        self.sae.eval()
        parts = []
        for i in range(0, len(X), self.batch_size):
            x_batch = X[i : i + self.batch_size].to(self.device)
            _, h_sparse, _ = self.sae(x_batch)
            parts.append(h_sparse.cpu())
        return torch.cat(parts, dim=0)

    # ------------------------------------ #
    # 1. Per-feature activation statistics #
    # ------------------------------------ #

    def activation_stats(self, H: torch.Tensor) -> dict[str, np.ndarray]:
        """
        Compute per-feature activation statistics.

        Parameters
        ----------
        H : Tensor (N, n_features) — sparse activations

        Returns
        -------
        dict with:
            "fire_rate"   : ndarray (n_features,) — fraction of inputs that activate
            "mean_act"    : ndarray (n_features,) — mean activation when active
            "max_act"     : ndarray (n_features,) — max activation across dataset
            "std_act"     : ndarray (n_features,) — std of activation when active
        """
        is_active = (H.abs() > 0).float()
        fire_rate = is_active.mean(dim=0).numpy()

        # Mean and std only counting nonzero entries
        # Use abs() consistently — activations can be negative
        H_abs = H.abs()
        sum_act = (H_abs * is_active).sum(dim=0)
        count_act = is_active.sum(dim=0).clamp(min=1)
        mean_act = (sum_act / count_act).numpy()

        # Std when active: sqrt(E[|x|^2] - E[|x|]^2)
        sum_sq = (H_abs ** 2 * is_active).sum(dim=0)
        mean_sq = (sum_sq / count_act).numpy()
        std_act = np.sqrt(np.maximum(mean_sq - mean_act ** 2, 0.0))

        max_act = H.abs().max(dim=0).values.numpy()

        return {
            "fire_rate": fire_rate,
            "mean_act": mean_act,
            "std_act": std_act,
            "max_act": max_act,
        }

    def activation_histograms(
        self,
        H: torch.Tensor,
        feature_indices: list[int] | None = None,
        n_bins: int = 50,
    ) -> dict[int, dict[str, np.ndarray]]:
        """
        Compute activation histograms for selected features.

        Includes all samples (zeros and nonzeros). Bimodal distributions
        reveal features that partition the dataset into two groups.

        Parameters
        ----------
        H : Tensor (N, n_features)
        feature_indices : list of int, or None for all features
        n_bins : int

        Returns
        -------
        histograms : dict mapping feature_idx -> {"bin_edges": ..., "counts": ...}
        """
        if feature_indices is None:
            feature_indices = list(range(H.shape[1]))

        histograms = {}
        for j in feature_indices:
            vals = H[:, j].numpy()
            counts, bin_edges = np.histogram(vals, bins=n_bins)
            histograms[j] = {
                "bin_edges": bin_edges,
                "counts": counts,
                "n_zero": int((vals == 0).sum()),
                "n_nonzero": int((vals != 0).sum()),
            }
        return histograms

    # ------------------------ #
    # 2. Property correlations #
    # ------------------------ #

    def property_correlations(
        self,
        H: torch.Tensor,
        labels: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Compute Pearson correlation between each feature and each property.

        Uses vectorised computation for efficiency.

        Parameters
        ----------
        H : Tensor (N, n_features)
        labels : dict mapping property name -> ndarray (N,)

        Returns
        -------
        correlations : dict mapping property name -> ndarray (n_features,)
        """
        H_np = H.numpy().astype(np.float64)
        n_samples, n_features = H_np.shape

        # Centre H columns once
        H_mean = H_np.mean(axis=0, keepdims=True)
        H_centered = H_np - H_mean
        H_std = H_np.std(axis=0)
        H_std[H_std < 1e-10] = 1.0  # avoid division by zero

        correlations = {}
        for prop_name, y in labels.items():
            y = np.asarray(y, dtype=np.float64)
            y_std = y.std()
            if y_std < 1e-10:
                correlations[prop_name] = np.zeros(n_features)
                continue

            y_centered = y - y.mean()
            # Vectorised Pearson: r = (H_centered^T @ y_centered) / (N * std_H * std_y)
            corrs = (H_centered.T @ y_centered) / (n_samples * H_std * y_std)
            correlations[prop_name] = corrs

        return correlations

    # ------------------------------------ #
    # 3. Top exemplar crystals per feature #
    # ------------------------------------ #

    def top_exemplars(
        self,
        H: torch.Tensor,
        n_top: int = 10,
    ) -> dict[int, list[int]]:
        """
        For each feature, find the crystal indices that activate it most.

        Parameters
        ----------
        H : Tensor (N, n_features)
        n_top : int

        Returns
        -------
        exemplars : dict mapping feature_idx -> list of crystal indices
                    (sorted by activation, highest first)
        """
        n_features = H.shape[1]
        exemplars = {}

        for j in range(n_features):
            vals, idx = torch.topk(H[:, j], min(n_top, len(H)))
            active = vals > 0
            exemplars[j] = idx[active].tolist()

        return exemplars

    # -------------------------------- #
    # 4. Feature co-occurrence matrix
    # -------------------------------- #

    def cooccurrence_matrix(self, H: torch.Tensor) -> np.ndarray:
        """
        Compute the feature co-occurrence matrix.

        Entry (i, j) = fraction of samples where both feature i AND feature j
        are active. High co-occurrence means the features represent related
        or jointly-occurring concepts. Zero co-occurrence means they are
        mutually exclusive (e.g., "metal" vs "insulator").

        Parameters
        ----------
        H : Tensor (N, n_features)

        Returns
        -------
        cooc : ndarray (n_features, n_features) — values in [0, 1]
        """
        binary = (H.abs() > 0).float()  # (N, n_features)
        n_samples = binary.shape[0]

        # Co-occurrence = (binary^T @ binary) / N
        # Process in chunks to avoid OOM on large feature counts
        n_features = binary.shape[1]
        cooc = np.zeros((n_features, n_features), dtype=np.float32)

        chunk_size = 512
        for i in range(0, n_features, chunk_size):
            i_end = min(i + chunk_size, n_features)
            chunk_i = binary[:, i:i_end]  # (N, chunk)
            block = (chunk_i.T @ binary).numpy() / n_samples  # (chunk, n_features)
            cooc[i:i_end, :] = block

        return cooc

    def top_cooccurring_pairs(
        self,
        cooc: np.ndarray,
        n_top: int = 20,
    ) -> list[tuple[int, int, float]]:
        """
        Return the top co-occurring feature pairs (excluding self-pairs).

        Parameters
        ----------
        cooc : ndarray (n_features, n_features)
        n_top : int

        Returns
        -------
        pairs : list of (feature_i, feature_j, co-occurrence rate)
        """
        # Zero out diagonal and lower triangle to avoid duplicates
        mask = np.triu(np.ones_like(cooc, dtype=bool), k=1)
        vals = cooc[mask]
        n_top = min(n_top, len(vals))
        if n_top == 0:
            return []
        indices = np.argpartition(vals, -n_top)[-n_top:]
        indices = indices[np.argsort(vals[indices])[::-1]]

        rows, cols = np.where(mask)
        pairs = []
        for idx in indices:
            pairs.append((int(rows[idx]), int(cols[idx]), float(vals[idx])))
        return pairs

    # ------------------------------------ #
    # 5. Decoder weight cosine similarity  #
    # ------------------------------------ #

    @torch.no_grad()
    def decoder_similarity(self) -> np.ndarray:
        """
        Cosine similarity between all pairs of decoder weight columns.

        Each column of W_dec is the "direction" of one SAE feature in
        normalized z_con space. High similarity means two features point
        in nearly the same direction — possibly redundant or fine-grained
        variants of the same concept.

        Returns
        -------
        sim : ndarray (n_features, n_features) — values in [-1, 1]
        """
        W = self.sae.W_dec.data  # (input_dim, n_features)
        W_norm = F.normalize(W, dim=0)  # unit-norm columns
        sim = (W_norm.T @ W_norm).cpu().numpy()  # (n_features, n_features)
        return sim

    def decoder_similarity_clusters(
        self,
        sim: np.ndarray,
        threshold: float = 0.8,
    ) -> list[list[int]]:
        """
        Find clusters of features with high decoder similarity.

        Simple greedy clustering: two features are in the same cluster
        if their cosine similarity exceeds the threshold.

        Parameters
        ----------
        sim : ndarray (n_features, n_features)
        threshold : float

        Returns
        -------
        clusters : list of lists, each containing feature indices
                   (only clusters with 2+ members are returned)
        """
        n = sim.shape[0]
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            neighbors = np.where(sim[i] > threshold)[0]
            neighbors = [j for j in neighbors if j != i and j not in visited]
            if neighbors:
                cluster = [i] + neighbors
                clusters.append(cluster)
                visited.update(cluster)

        return clusters

    # ---------------------- #
    # 6. Element enrichment  #
    # ---------------------- #

    def element_enrichment(
        self,
        H: torch.Tensor,
        dataset,
        n_top_crystals: int = 50,
        n_top_elements: int = 5,
    ) -> dict[int, list[tuple[str, float]]]:
        """
        For each feature, find which elements are over-represented in its
        top-activating crystals compared to the full dataset.

        Enrichment ratio = (fraction of element in feature's top crystals)
                         / (fraction of element in full dataset)

        A ratio > 1 means the element is more common in this feature's
        top crystals than in the dataset overall.

        Parameters
        ----------
        H : Tensor (N, n_features)
        dataset : CrystDataset
            The dataset with cached_data containing graph_arrays.
        n_top_crystals : int
            Number of top-activating crystals to consider per feature.
        n_top_elements : int
            Number of enriched elements to return per feature.

        Returns
        -------
        enrichment : dict mapping feature_idx -> list of (element_symbol, ratio)
                     sorted by enrichment ratio, descending.
        """
        # Build global element frequency from full dataset
        global_counts = Counter()
        crystal_elements = []  # list of sets, one per crystal
        for entry in dataset.cached_data:
            _, atom_types, _, _, _, _, _ = entry["graph_arrays"]
            elements = set(int(a) for a in atom_types)
            crystal_elements.append(elements)
            global_counts.update(elements)

        n_total = len(crystal_elements)
        global_freq = {
            elem: count / n_total for elem, count in global_counts.items()
        }

        # Per-feature enrichment (only for features that actually fire)
        n_features = H.shape[1]
        enrichment = {}
        fire_mask = (H.abs() > 0).any(dim=0)  # skip dead features

        for j in range(n_features):
            if not fire_mask[j]:
                enrichment[j] = []
                continue
            vals, idx = torch.topk(H[:, j], min(n_top_crystals, len(H)))
            active = vals > 0
            top_idx = idx[active].tolist()

            if len(top_idx) < 3:
                enrichment[j] = []
                continue

            # Element frequency in this feature's top crystals
            feat_counts = Counter()
            for ci in top_idx:
                feat_counts.update(crystal_elements[ci])

            n_feat = len(top_idx)
            ratios = []
            for elem, count in feat_counts.items():
                feat_freq = count / n_feat
                base_freq = global_freq.get(elem, 1e-6)
                ratio = feat_freq / base_freq
                symbol = ELEMENT_SYMBOLS[elem] if elem < len(ELEMENT_SYMBOLS) else f"Z={elem}"
                ratios.append((symbol, ratio))

            ratios.sort(key=lambda x: x[1], reverse=True)
            enrichment[j] = ratios[:n_top_elements]

        return enrichment

    # ----------------------------- #
    # 7. Probe direction alignment  #
    # ----------------------------- #

    @torch.no_grad()
    def probe_alignment(
        self,
        probe_weights: dict[str, torch.Tensor],
    ) -> dict[str, np.ndarray]:
        """
        Cosine similarity between each SAE decoder column and each
        trained linear probe's weight vector.

        If SAE feature j's decoder column aligns with the band_gap probe
        direction, that feature likely encodes band gap.

        Parameters
        ----------
        probe_weights : dict mapping property name -> Tensor (input_dim,)
            or (1, input_dim). The weight vector from a trained linear probe
            for each property. For regression probes with output_dim=1,
            this is probe.linear.weight.squeeze(0).

        Returns
        -------
        alignment : dict mapping property name -> ndarray (n_features,)
            Cosine similarity between each feature direction and the
            probe direction. Values in [-1, 1].
        """
        W_dec = self.sae.W_dec.data  # (input_dim, n_features)
        W_dec_norm = F.normalize(W_dec, dim=0)  # unit-norm columns

        alignment = {}
        for prop_name, w_probe in probe_weights.items():
            # Classification probes have shape (num_classes, input_dim) with
            # num_classes > 1 — they don't define a single direction, so skip
            if w_probe.dim() > 1 and w_probe.shape[0] != 1:
                continue
            if w_probe.dim() > 1:
                w_probe = w_probe.squeeze(0)
            w_probe = w_probe.to(W_dec.device)
            w_probe_norm = F.normalize(w_probe.unsqueeze(0), dim=1).squeeze(0)

            # Cosine similarity: (n_features,)
            cos_sim = (W_dec_norm.T @ w_probe_norm).cpu().numpy()
            alignment[prop_name] = cos_sim

        return alignment

    # --------------------- #
    # 8. Feature dashboard  #
    # --------------------- #

    def feature_dashboard(
        self,
        H: torch.Tensor,
        labels: dict[str, np.ndarray],
        n_top_features: int = 20,
        top_p: float | None = None,
        stats: dict[str, np.ndarray] | None = None,
        corrs: dict[str, np.ndarray] | None = None,
        align: dict[str, np.ndarray] | None = None,
        enrich: dict[int, list] | None = None,
    ) -> list[dict]:
        """
        Build a unified summary table of the most interesting features.

        "Interesting" = highest absolute correlation with any property.

        Parameters
        ----------
        H : Tensor (N, n_features)
        labels : dict of property labels
        n_top_features : int
            Absolute number of features to include. Ignored if top_p is set.
        top_p : float or None
            If set, include the top ``top_p`` fraction of all features
            (e.g. 0.10 for 10%). Overrides ``n_top_features``.
        stats : pre-computed activation stats (avoids recomputation)
        corrs : pre-computed property correlations (avoids recomputation)
        align : pre-computed probe alignment
        enrich : pre-computed element enrichment

        Returns
        -------
        dashboard : list of dicts, one per feature, with keys:
            "feature_idx", "fire_rate", "mean_act", "std_act",
            "top_property", "top_corr", "all_corrs",
            "probe_alignment" (if align given),
            "enriched_elements" (if enrich given)
        """
        if stats is None:
            stats = self.activation_stats(H)
        if corrs is None:
            corrs = self.property_correlations(H, labels)

        # Find peak |correlation| per feature across all properties
        n_features = H.shape[1]

        # Resolve how many features to include
        if top_p is not None:
            n_top_features = max(1, int(top_p * n_features))
        peak_corr = np.zeros(n_features)
        peak_prop = [""] * n_features

        for prop_name, c in corrs.items():
            abs_c = np.abs(c)
            better = abs_c > np.abs(peak_corr)
            peak_corr[better] = c[better]
            for j in np.where(better)[0]:
                peak_prop[j] = prop_name

        # Sort by |peak correlation|, take top N
        top_idx = np.argsort(np.abs(peak_corr))[::-1][:n_top_features]

        dashboard = []
        for j in top_idx:
            j = int(j)
            entry = {
                "feature_idx": j,
                "fire_rate": float(stats["fire_rate"][j]),
                "mean_act": float(stats["mean_act"][j]),
                "std_act": float(stats["std_act"][j]),
                "top_property": peak_prop[j],
                "top_corr": float(peak_corr[j]),
                "all_corrs": {
                    prop: float(corrs[prop][j]) for prop in corrs
                },
            }

            if align is not None:
                entry["probe_alignment"] = {
                    prop: float(align[prop][j]) for prop in align
                }

            if enrich is not None and j in enrich:
                entry["enriched_elements"] = enrich[j]

            dashboard.append(entry)

        return dashboard

    def print_dashboard(
        self,
        dashboard: list[dict],
        show_elements: bool = True,
        show_alignment: bool = True,
    ) -> None:
        """Pretty-print the feature dashboard."""
        print(f"\n{'=' * 90}")
        print("SAE FEATURE DASHBOARD — Top features by property correlation")
        print(f"{'=' * 90}")

        # Build header
        header = (
            f"{'Feat':>5s}  {'Fire%':>6s}  {'MeanAct':>8s}  "
            f"{'TopProp':>20s}  {'Corr':>7s}"
        )
        if show_alignment and dashboard and "probe_alignment" in dashboard[0]:
            header += f"  {'PrbAlign':>8s}"
        if show_elements and dashboard and "enriched_elements" in dashboard[0]:
            header += f"  {'Top Elements':>25s}"
        print(header)
        print("-" * len(header))

        for entry in dashboard:
            line = (
                f"{entry['feature_idx']:5d}  "
                f"{entry['fire_rate']*100:5.1f}%  "
                f"{entry['mean_act']:8.4f}  "
                f"{entry['top_property']:>20s}  "
                f"{entry['top_corr']:+7.4f}"
            )

            if show_alignment and "probe_alignment" in entry:
                # Show alignment for the top-correlated property
                top_prop = entry["top_property"]
                if top_prop in entry["probe_alignment"]:
                    line += f"  {entry['probe_alignment'][top_prop]:+8.4f}"
                else:
                    line += f"  {'--':>8s}"

            if show_elements and "enriched_elements" in entry:
                elems = entry["enriched_elements"]
                if elems:
                    elem_str = ", ".join(
                        f"{sym}({ratio:.1f}x)" for sym, ratio in elems[:3]
                    )
                    line += f"  {elem_str:>25s}"
                else:
                    line += f"  {'--':>25s}"

            print(line)

        print(f"{'=' * 90}")

    # ----------------------------------------- #
    # 9. Variance explained by feature subsets  #
    # ----------------------------------------- #

    @torch.no_grad()
    def variance_explained_by_features(
        self,
        X: torch.Tensor,
        H: torch.Tensor,
        feature_indices: list[int],
        per_feature: bool = False,
    ) -> dict:
        """
        Compute variance explained when decoding with only a subset of features.

        Reconstructions are denormalized back to raw space before comparing
        against the raw input X.

        Parameters
        ----------
        X : Tensor (N, input_dim) — original raw activations
        H : Tensor (N, n_features) — sparse SAE activations
        feature_indices : list of int
            Which features to include in the reconstruction.
        per_feature : bool
            If True, also compute each feature's marginal variance explained
            (decoding with that single feature alone). Note: marginals don't
            sum to the joint value because features can share variance.

        Returns
        -------
        result : dict with keys:
            "joint"     : float — variance explained by all features together
            "marginals" : dict[int, float] — per-feature variance explained
                          (only if per_feature=True)
        """
        var_x = X.var().item()

        def _ve_for_subset(indices: list[int]) -> float:
            H_masked = torch.zeros_like(H)
            H_masked[:, indices] = H[:, indices]
            x_hat = self.sae.decode_denorm(H_masked)
            return 1.0 - (X - x_hat).var().item() / var_x

        result = {"joint": _ve_for_subset(feature_indices)}

        if per_feature:
            marginals = {}
            for j in feature_indices:
                marginals[j] = _ve_for_subset([j])
            result["marginals"] = marginals

        return result

    def cumulative_variance_curve(
        self,
        X: torch.Tensor,
        H: torch.Tensor,
    ) -> dict:
        """
        Sort all features by marginal variance explained and compute
        the cumulative variance curve as features are added one by one.

        This reveals how concentrated the representation is — if a small
        fraction of features captures most variance, the SAE has found
        a compact basis.

        Parameters
        ----------
        X : Tensor (N, input_dim) — original raw activations
        H : Tensor (N, n_features) — sparse SAE activations

        Returns
        -------
        result : dict with keys:
            "feature_order" : list[int] — feature indices sorted by
                              descending marginal variance explained
            "marginals"     : list[float] — marginal VE for each feature
                              (in the same sorted order)
            "cumulative"    : list[float] — cumulative VE when adding
                              features one at a time in sorted order
        """
        var_x = X.var().item()
        n_features = H.shape[1]

        # Step 1: compute marginal VE for every feature
        marginals = np.zeros(n_features)
        for j in range(n_features):
            h_j = H[:, j]  # (N,)
            if h_j.abs().max() == 0:
                continue  # dead feature, VE = 0
            H_single = torch.zeros_like(H)
            H_single[:, j] = h_j
            x_hat_j = self.sae.decode_denorm(H_single)
            marginals[j] = 1.0 - (X - x_hat_j).var().item() / var_x

        # Step 2: sort by descending marginal VE
        order = np.argsort(marginals)[::-1]
        sorted_marginals = marginals[order].tolist()

        # Step 3: cumulative VE — greedily add features in sorted order
        cumulative = []
        H_accum = torch.zeros_like(H)
        for j in order:
            j = int(j)
            H_accum[:, j] = H[:, j]
            x_hat = self.sae.decode_denorm(H_accum)
            ve = 1.0 - (X - x_hat).var().item() / var_x
            cumulative.append(ve)

        return {
            "feature_order": [int(j) for j in order],
            "marginals": sorted_marginals,
            "cumulative": cumulative,
        }

    # --------------------------- #
    # (*) Full analysis pipeline  #
    # --------------------------- #

    def full_analysis(
        self,
        X: torch.Tensor,
        labels: dict[str, np.ndarray],
        probe_weights: dict[str, torch.Tensor] | None = None,
        dataset=None,
        n_top_features: int = 30,
        top_p: float | None = None,
        sim_threshold: float = 0.8,
    ) -> dict:
        """
        Run the complete analysis pipeline and return all results.

        Parameters
        ----------
        X : Tensor (N, input_dim) — raw activations to analyse
        labels : dict of property labels
        probe_weights : optional linear probe weights
        dataset : optional CrystDataset for element enrichment
        n_top_features : int
            Absolute number of features for the dashboard. Ignored if
            top_p is set.
        top_p : float or None
            If set, include the top ``top_p`` fraction of all features
            in the dashboard (e.g. 0.10 for 10%). Overrides
            ``n_top_features``.

        Returns
        -------
        results : dict with all analysis outputs
        """
        print("Encoding dataset through SAE...")
        H = self.encode_dataset(X)

        print("Computing activation statistics...")
        stats = self.activation_stats(H)

        print("Computing property correlations...")
        corrs = self.property_correlations(H, labels)

        print("Finding top exemplars...")
        exemplars = self.top_exemplars(H, n_top=10)

        print("Computing feature co-occurrence matrix...")
        cooc = self.cooccurrence_matrix(H)
        top_pairs = self.top_cooccurring_pairs(cooc, n_top=20)

        print("Computing decoder similarity matrix...")
        dec_sim = self.decoder_similarity()
        clusters = self.decoder_similarity_clusters(dec_sim, threshold=sim_threshold)

        # Optional analyses
        align = None
        if probe_weights is not None:
            print("Computing probe direction alignment...")
            align = self.probe_alignment(probe_weights)

        enrich = None
        if dataset is not None:
            print("Computing element enrichment...")
            enrich = self.element_enrichment(H, dataset)

        print("Building feature dashboard...")
        dashboard = self.feature_dashboard(
            H, labels,
            n_top_features=n_top_features,
            top_p=top_p,
            stats=stats,
            corrs=corrs,
            align=align,
            enrich=enrich,
        )

        # Summary statistics
        n_dead = int((stats["fire_rate"] == 0).sum())
        n_rare = int((stats["fire_rate"] < 0.01).sum())

        # Variance explained by feature subsets
        print("Computing variance explained by feature subsets...")
        dashboard_indices = [e["feature_idx"] for e in dashboard]
        dashboard_ve = self.variance_explained_by_features(
            X, H, dashboard_indices, per_feature=True,
        )

        # Rare features: fire rate < 1%
        rare_indices = [
            j for j in range(H.shape[1])
            if 0 < stats["fire_rate"][j] < 0.01
        ]
        rare_ve = self.variance_explained_by_features(
            X, H, rare_indices,
        ) if rare_indices else {"joint": 0.0}

        # Dead features (sanity check — should be 0)
        dead_indices = [
            j for j in range(H.shape[1]) if stats["fire_rate"][j] == 0
        ]
        dead_ve = self.variance_explained_by_features(
            X, H, dead_indices,
        ) if dead_indices else {"joint": 0.0}

        # Decoder similarity cluster VE
        cluster_ve = {}
        for i, cluster_members in enumerate(clusters):
            cve = self.variance_explained_by_features(X, H, cluster_members)
            cluster_ve[i] = {
                "features": cluster_members,
                "joint_ve": cve["joint"],
            }

        return {
            "H": H,
            "stats": stats,
            "correlations": corrs,
            "exemplars": exemplars,
            "cooccurrence": cooc,
            "top_cooccurring_pairs": top_pairs,
            "decoder_similarity": dec_sim,
            "decoder_clusters": clusters,
            "probe_alignment": align,
            "element_enrichment": enrich,
            "dashboard": dashboard,
            "n_dead": n_dead,
            "n_rare": n_rare,
            "sim_threshold": sim_threshold,
            "variance_explained": {
                "dashboard": dashboard_ve,
                "rare": rare_ve,
                "dead": dead_ve,
                "decoder_clusters": cluster_ve,
            },
        }

    def print_full_report(self, results: dict) -> None:
        """Print a comprehensive report from full_analysis results."""

        # Dashboard
        self.print_dashboard(results["dashboard"])

        # Feature health
        print(f"\nFeature health:")
        print(f"  Dead (0% fire rate):  {results['n_dead']}")
        print(f"  Rare (<1% fire rate): {results['n_rare']}")

        # Variance explained by feature subsets
        ve = results.get("variance_explained")
        if ve:
            n_dash = len(results["dashboard"])
            n_total = results["stats"]["fire_rate"].shape[0]
            print(f"\nVariance explained by feature subsets:")
            print(f"  Dashboard (top {n_dash} by corr):  {ve['dashboard']['joint']:.4f}")
            if "marginals" in ve["dashboard"]:
                marginals = ve["dashboard"]["marginals"]
                top5 = sorted(marginals.items(), key=lambda kv: kv[1], reverse=True)[:5]
                print(f"  Top 5 individual contributors:")
                for feat_idx, mve in top5:
                    print(f"    Feature {feat_idx:5d}:  {mve:.4f}")
            print(f"  Rare (<1% fire rate, n={results['n_rare']}):  {ve['rare']['joint']:.4f}")
            print(f"  Dead (n={results['n_dead']}):  {ve['dead']['joint']:.4f}  (sanity: should be ~0)")

            cluster_ve = ve.get("decoder_clusters", {})
            if cluster_ve:
                print(f"\n  Decoder similarity cluster variance:")
                for i, cinfo in sorted(cluster_ve.items()):
                    members = cinfo["features"]
                    print(
                        f"    Cluster {i} ({len(members)} features): "
                        f" VE={cinfo['joint_ve']:.4f}  features={members}"
                    )

        # Top co-occurring pairs
        print(f"\nTop co-occurring feature pairs:")
        for fi, fj, rate in results["top_cooccurring_pairs"][:10]:
            print(f"  Feature {fi:4d} + Feature {fj:4d}  co-occur {rate*100:.1f}%")

        # Decoder similarity clusters
        clusters = results["decoder_clusters"]
        threshold = results.get("sim_threshold", "?")
        if clusters:
            print(f"\nDecoder similarity clusters (cosine > {threshold}):")
            for i, cluster in enumerate(clusters[:10]):
                print(f"  Cluster {i}: features {cluster}")
        else:
            print(f"\nNo decoder similarity clusters found (threshold {threshold})")

        print()
