# datasets.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import scipy.io as sio

class HSIDataset(Dataset):
    """
    Hyperspectral Dataset loader for .mat files (e.g., Indian Pines, PaviaU).
    Loads a single data cube (H×W×S) and its label map (H×W), then extracts
    non-overlapping patches centered at labeled pixels only.
    """
    def __init__(self, data_path, spectral_dim, data, patch_size=12):
        super().__init__()
        self.spectral_dim = spectral_dim

        # Decide file names based on dataset identifier
        data_file  = os.path.join(data_path, f"{data}.mat")
        label_file = os.path.join(data_path, "labels.mat")

        data_mat  = sio.loadmat(data_file)
        label_mat = sio.loadmat(label_file)

        # Dynamically pick the first non-metadata variable
        data_key  = next(k for k in data_mat  if not k.startswith('__'))
        label_key = next(k for k in label_mat if not k.startswith('__'))

        self.data   = data_mat[data_key]    # shape: H, W, S
        self.labels = label_mat[label_key]  # shape: H, W

        H, W, S = self.data.shape
        assert S == spectral_dim, f"Expected spectral_dim={spectral_dim}, got {S}"

        self.patch_size = patch_size
        pad = patch_size // 2

        # Pad spatial dims so border patches are full size
        self.padded = np.pad(
            self.data,
            pad_width=((pad, pad), (pad, pad), (0, 0)),
            mode='reflect'
        )

        # Generate non-overlapping patch centers at labeled pixels only
        self.coords = []
        for i in range(pad, pad + H, patch_size):
            for j in range(pad, pad + W, patch_size):
                i_orig = i - pad
                j_orig = j - pad
                # include patch only if label exists (>0)
                if self.labels[i_orig, j_orig] > 0:
                    self.coords.append((i, j))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        p = self.patch_size // 2

        # Extract patch from padded cube
        patch = self.padded[
            i - p : i + p + 1,
            j - p : j + p + 1,
            :
        ]
        # Reorder to (C, H, W): here C = spectral channels
        patch = np.transpose(patch, (2, 0, 1)).astype(np.float32)
        patch = torch.from_numpy(patch).unsqueeze(0)  # shape: (1, S, p*2+1, p*2+1)

        # Corresponding label: zero-based
        label = int(self.labels[i - p, j - p]) - 1

        return patch, label


def split_dataset(full_dataset, train_ratio=0.6, val_ratio=0.2):
    """
    Split a Dataset into train/val/test subsets by ratio.
    """
    total = len(full_dataset)
    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)
    n_test  = total - n_train - n_val
    return random_split(full_dataset, [n_train, n_val, n_test])
