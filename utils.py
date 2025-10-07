import math
import numpy as np
import torch



def denorm_torch(y_norm: torch.Tensor, y_min: torch.Tensor, y_max: torch.Tensor) -> torch.Tensor:
    # y_norm, y_min, y_max are broadcast-compatible, e.g. [B, Dy] and [1, Dy]
    return y_norm * (y_max - y_min) + y_min


# Simple train/val split using a random permutation
def train_val_split(dataset, val_ratio=0.2, seed=42):
    n = len(dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())
    return train_subset, val_subset
