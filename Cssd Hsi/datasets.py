import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import scipy.io as sio

class HSIDataset(Dataset):
    """
    Hyperspectral Dataset loader for .mat files (e.g., Indian Pines)
    Assumes data.mat contains 'indian_pines_corrected' and labels.mat contains 'indian_pines_gt'
    """
    def __init__(self, data_path, spectral_dim, patch_size=5):
        super().__init__()
        self.spectral_dim = spectral_dim
        data_mat = sio.loadmat(os.path.join(data_path, 'data.mat'))
        label_mat = sio.loadmat(os.path.join(data_path, 'labels.mat'))
        # adjust keys as needed
        self.data = data_mat.get('data')  # shape (H, W, S)
        self.labels = label_mat.get('labels')  # shape (H, W)
        H, W, S = self.data.shape
        assert S == spectral_dim, f"Expected spectral_dim={spectral_dim}, got {S}"
        self.patch_size = patch_size
        pad = patch_size // 2
        self.padded = np.pad(self.data, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
        self.coords = [(i, j) for i in range(H) for j in range(W) if self.labels[i,j]>0]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        pad = self.patch_size // 2
        patch = self.padded[i:i+self.patch_size, j:j+self.patch_size, :]
        # shape (patch, patch, spectral) -> (spectral, patch, patch)
        patch = np.transpose(patch, (2,0,1)).astype(np.float32)
        label = int(self.labels[i,j]) - 1  # zero-based
        return torch.from_numpy(patch).unsqueeze(0), label


def split_dataset(full_dataset, train_ratio=0.6, val_ratio=0.2):
    total = len(full_dataset)
    train_len = int(total * train_ratio)
    val_len = int(total * val_ratio)
    test_len = total - train_len - val_len
    return random_split(full_dataset, [train_len, val_len, test_len])