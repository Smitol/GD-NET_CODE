"""Data augmentation + PyTorch dataset for GD-Net (paper Section 2.1.2 & Fig. 1A).

Adapted from the authors' official `pcl/loader.py` (which itself derives from
CLEAR, Han et al. 2022 -- reference [29] of the paper). Changes: removed
unused image/scRNA-specific code (PIL, torchvision, obs labels), added
comments, and made the augmentation strengths explicit parameters.

For every patient sample, the dataset returns TWO independently augmented
"views" of the fused multi-omics vector. The two views form the POSITIVE PAIR
of the contrastive loss; all other patients act as negatives.

Augmentations used (paper: "Gaussian noise, simulated dropout, data swap and
data mask"):
    random_mask ............ set a random subset of features to 0 (dropout/mask)
    random_gaussian_noise .. add N(0, sigma) noise to a random subset
    random_swap ............ swap the values of random feature pairs
    instance_crossover ..... exchange a random subset of features with another
                             random patient (keeps values biologically plausible)

TUNING NOTE: the default percentages/probabilities below are the CLEAR/official
defaults. If training is unstable on your dataset, LOWER the percentages
(e.g. mask 0.10, noise 0.15) -- weaker augmentation = easier contrastive task.
"""

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

# Default augmentation configuration (same values as the official repo).
# Each entry is (fraction of features affected, probability the op is applied).
DEFAULT_AUG = {
    "mask_percentage": 0.15, "apply_mask_prob": 0.5,
    "noise_percentage": 0.20, "sigma": 0.5, "apply_noise_prob": 0.3,
    "swap_percentage": 0.10, "apply_swap_prob": 0.5,
    "cross_percentage": 0.25, "apply_cross_prob": 0.4,
}


class MultiOmicsDataset(Dataset):
    """Wraps the fused (patients x features) matrix.

    Args:
        data:      np.ndarray of shape (n_patients, n_features) -- the fused
                   multi-omics matrix X = rowbind(mRNA, methylation, miRNA)
                   AFTER preprocessing (imputed, log-transformed, z-scored).
        transform: if True, __getitem__ returns [view1, view2] (two random
                   augmentations) for contrastive training; if False, returns
                   the raw vector (used at eval time to extract embeddings).
        aug:       dict of augmentation strengths (see DEFAULT_AUG).
    """

    def __init__(self, data: np.ndarray, transform: bool = True, aug: dict = None):
        super().__init__()
        self.data = np.asarray(data, dtype=np.float32)
        self.transform = transform
        self.aug = dict(DEFAULT_AUG, **(aug or {}))
        self.num_cells, self.num_genes = self.data.shape

    def _augment(self, profile: np.ndarray) -> torch.Tensor:
        tr = Transformation(self.data, profile)
        tr.random_mask(self.aug["mask_percentage"], self.aug["apply_mask_prob"])
        tr.random_gaussian_noise(self.aug["noise_percentage"], self.aug["sigma"],
                                 self.aug["apply_noise_prob"])
        tr.random_swap(self.aug["swap_percentage"], self.aug["apply_swap_prob"])
        tr.instance_crossover(self.aug["cross_percentage"], self.aug["apply_cross_prob"])
        return torch.from_numpy(tr.cell_profile.astype(np.float32))

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            return [self._augment(sample), self._augment(sample)], index
        return torch.from_numpy(sample), index

    def __len__(self):
        return self.num_cells


class Transformation:
    """The four random augmentations (official implementation, commented)."""

    def __init__(self, dataset: np.ndarray, cell_profile: np.ndarray):
        self.dataset = dataset                      # whole matrix (for crossover)
        self.cell_profile = deepcopy(cell_profile)  # the vector being augmented
        self.gene_num = len(self.cell_profile)
        self.cell_num = len(self.dataset)

    def _build_mask(self, frac: float) -> np.ndarray:
        """Random boolean mask selecting `frac` of the features."""
        k = int(self.gene_num * frac)
        mask = np.zeros(self.gene_num, dtype=bool)
        mask[:k] = True
        np.random.shuffle(mask)
        return mask

    def random_mask(self, mask_percentage=0.15, apply_mask_prob=0.5):
        """Simulated dropout: zero-out a random subset of features."""
        if np.random.uniform(0, 1) < apply_mask_prob:
            self.cell_profile[self._build_mask(mask_percentage)] = 0

    def random_gaussian_noise(self, noise_percentage=0.2, sigma=0.5, apply_noise_prob=0.3):
        """Add Gaussian noise N(0, sigma) to a random subset of features."""
        if np.random.uniform(0, 1) < apply_noise_prob:
            mask = self._build_mask(noise_percentage)
            self.cell_profile[mask] += np.random.normal(0, sigma, int(mask.sum()))

    def random_swap(self, swap_percentage=0.1, apply_swap_prob=0.5):
        """Swap the values of random feature pairs (permutes ~swap_percentage of features)."""
        if np.random.uniform(0, 1) < apply_swap_prob:
            n_pairs = int(self.gene_num * swap_percentage / 2)
            pairs = np.random.randint(self.gene_num, size=(n_pairs, 2))
            self.cell_profile[pairs[:, 0]], self.cell_profile[pairs[:, 1]] = \
                self.cell_profile[pairs[:, 1]], self.cell_profile[pairs[:, 0]].copy()

    def instance_crossover(self, cross_percentage=0.25, apply_cross_prob=0.4):
        """Replace a random subset of features with the values of another random patient."""
        if np.random.uniform(0, 1) < apply_cross_prob:
            other = self.dataset[np.random.randint(self.cell_num)]
            mask = self._build_mask(cross_percentage)
            self.cell_profile[mask] = other[mask]
