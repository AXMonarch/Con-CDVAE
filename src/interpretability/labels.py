# labels.py — Label extraction from dataset and CSV
#
# Extracts ground-truth labels for each crystal in the dataset, to be paired
# with activations for probe training.
#
# Label sources:
#   - formation_energy_per_atom : prelo_prop in cached dataset (always available)
#   - band_gap                  : prelo_prop in cached dataset (always available)
#   - e_above_hull              : prelo_prop in cached dataset (always available)
#   - spacegroup                : CSV column 'spacegroup.number' (loaded separately)
#   - crystal_system            : derived from spacegroup via trivial lookup
#   - is_metal                  : derived from band_gap < 0.01
#   - num_atoms                 : batch.num_atoms (always in Data object)
#
# The prelo_prop mechanism in Con-CDVAE caches formation_energy_per_atom,
# band_gap, and e_above_hull into the preprocessed .pt file. However, only
# properties listed in `prop` (just formation_energy_per_atom) are attached
# to each Data object in __getitem__. The others exist in cached_data but
# must be accessed from the underlying dict.

import csv
from pathlib import Path

import torch
import numpy as np


# ---- Space group -> crystal system lookup -----------------------------------
#
# ITA standard: triclinic(1-2), monoclinic(3-15), orthorhombic(16-74),
#               tetragonal(75-142), trigonal(143-167), hexagonal(168-194),
#               cubic(195-230)

CRYSTAL_SYSTEM_RANGES = [
    (1,   2,   0, "triclinic"),
    (3,   15,  1, "monoclinic"),
    (16,  74,  2, "orthorhombic"),
    (75,  142, 3, "tetragonal"),
    (143, 167, 4, "trigonal"),
    (168, 194, 5, "hexagonal"),
    (195, 230, 6, "cubic"),
]

CRYSTAL_SYSTEM_NAMES = [name for _, _, _, name in CRYSTAL_SYSTEM_RANGES]
NUM_CRYSTAL_SYSTEMS = len(CRYSTAL_SYSTEM_NAMES)   # 7
NUM_SPACEGROUPS = 230


def spacegroup_to_crystal_system(sg: int) -> int:
    """Map a space group number (1-230) to crystal system index (0-6)."""
    for lo, hi, idx, _ in CRYSTAL_SYSTEM_RANGES:
        if lo <= sg <= hi:
            return idx
    raise ValueError(f"Invalid space group number: {sg}")


# ---- Label extractor --------------------------------------------------------

class LabelExtractor:
    """
    Extracts ground-truth labels for probing from a Con-CDVAE dataset.

    Loads labels from two sources:
      1. The dataset's cached_data (for prelo_prop: energy, band_gap, ehull)
      2. The original CSV file (for spacegroup.number)

    Parameters
    ----------
    dataset : CrystDataset
        A loaded Con-CDVAE dataset with cached_data populated.
    csv_path : str or Path
        Path to the original CSV (e.g., data/mp_20/train.csv).
    """

    def __init__(self, dataset, csv_path: str | Path):
        self.dataset = dataset
        self.csv_path = Path(csv_path)
        self._spacegroups: np.ndarray | None = None

    # -- Lazy loading of spacegroups from CSV ---------------------------------

    def _load_spacegroups(self) -> np.ndarray:
        """Load spacegroup numbers from CSV. Cached after first call."""
        if self._spacegroups is not None:
            return self._spacegroups

        spacegroups = []
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                spacegroups.append(int(row["spacegroup.number"]))
        self._spacegroups = np.array(spacegroups, dtype=np.int64)
        return self._spacegroups

    # -- Property accessors ---------------------------------------------------

    def _get_from_cached(self, key: str) -> np.ndarray:
        """
        Extract a property from the dataset's cached_data dicts.

        The prelo_prop mechanism stores these as torch.Tensor([value]) in
        each cached_data entry. We extract and flatten them.
        """
        values = []
        for entry in self.dataset.cached_data:
            val = entry[key]
            if isinstance(val, torch.Tensor):
                values.append(val.item())
            else:
                values.append(float(val))
        return np.array(values, dtype=np.float32)

    def formation_energy(self) -> np.ndarray:
        """Per-crystal formation energy (eV/atom). Shape: (N_crystals,)."""
        return self._get_from_cached("formation_energy_per_atom")

    def band_gap(self) -> np.ndarray:
        """Per-crystal band gap (eV). Shape: (N_crystals,)."""
        return self._get_from_cached("band_gap")

    def e_above_hull(self) -> np.ndarray:
        """Per-crystal energy above hull (eV/atom). Shape: (N_crystals,)."""
        return self._get_from_cached("e_above_hull")

    def spacegroup(self) -> np.ndarray:
        """Per-crystal space group number (1-230). Shape: (N_crystals,)."""
        return self._load_spacegroups()

    def crystal_system(self) -> np.ndarray:
        """Per-crystal crystal system index (0-6). Shape: (N_crystals,)."""
        sgs = self.spacegroup()
        return np.array(
            [spacegroup_to_crystal_system(sg) for sg in sgs],
            dtype=np.int64,
        )

    def is_metal(self) -> np.ndarray:
        """Per-crystal binary label: 1 if band_gap < 0.01 eV. Shape: (N_crystals,)."""
        bg = self.band_gap()
        return (bg < 0.01).astype(np.int64)

    def num_atoms(self) -> np.ndarray:
        """Per-crystal atom count. Shape: (N_crystals,)."""
        values = []
        for entry in self.dataset.cached_data:
            _, atom_types, _, _, _, _, num_atoms = entry["graph_arrays"]
            values.append(int(num_atoms))
        return np.array(values, dtype=np.float32)

    # -- Convenience: get all labels as a dict --------------------------------

    def get_all_labels(self) -> dict[str, np.ndarray]:
        """
        Return a dict of all probe labels.

        Returns
        -------
        dict mapping label name -> numpy array of shape (N_crystals,)
        """
        return {
            "formation_energy": self.formation_energy(),
            "band_gap":         self.band_gap(),
            "e_above_hull":     self.e_above_hull(),
            "spacegroup":       self.spacegroup(),
            "crystal_system":   self.crystal_system(),
            "is_metal":         self.is_metal(),
            "num_atoms":        self.num_atoms(),
        }


# ---- Probe task definitions -------------------------------------------------
#
# Each entry defines: label name, task type, number of output classes (if
# classification), and which hook points this probe is applied to.

PROBE_TASKS = {
    "formation_energy": {
        "task": "regression",
        "hooks": ["fc_mu", "z_con", "cond_emb", "out_blk_0", "out_blk_4"],
    },
    "band_gap": {
        "task": "regression",
        "hooks": ["fc_mu", "z_con", "cond_emb", "out_blk_0", "out_blk_4"],
    },
    "e_above_hull": {
        "task": "regression",
        "hooks": ["fc_mu", "z_con", "cond_emb", "out_blk_0", "out_blk_4"],
    },
    "spacegroup": {
        "task": "classification",
        "num_classes": NUM_SPACEGROUPS,
        "hooks": ["fc_mu", "z_con"],
    },
    "crystal_system": {
        "task": "classification",
        "num_classes": NUM_CRYSTAL_SYSTEMS,
        "hooks": ["fc_mu", "z_con"],
    },
    "is_metal": {
        "task": "classification",
        "num_classes": 2,
        "hooks": ["fc_mu", "z_con", "cond_emb"],
    },
    "num_atoms": {
        "task": "regression",
        "hooks": ["fc_mu", "z_con"],
    },
}
