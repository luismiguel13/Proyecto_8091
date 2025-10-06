import math
import numpy as np
import torch

class MinMaxNormalizer:
    """
    Per-feature Min-Max normalizer for inputs (X) and targets (Y).
    Stores data_min_ and data_max_ for later exact denormalization.
    """

    def __init__(self):
        self.x_min_ = None
        self.x_max_ = None
        self.y_min_ = None
        self.y_max_ = None

    @staticmethod
    def _fit_min_max(arr, axis=0):
        data_min = np.min(arr, axis=axis, keepdims=True)
        data_max = np.max(arr, axis=axis, keepdims=True)
        return data_min, data_max

    @staticmethod
    def _transform(arr, data_min, data_max):
        scale = data_max - data_min
        scale[scale == 0] = 1.0  # avoid divide-by-zero if feature is constant
        return (arr - data_min) / scale

    @staticmethod
    def _inverse(arr_norm, data_min, data_max):
        return arr_norm * (data_max - data_min) + data_min

    def fit(self, X: np.ndarray, Y: np.ndarray | None = None):
        # X: [N, Dx], Y: [N, Dy] or None
        self.x_min_, self.x_max_ = self._fit_min_max(X, axis=0)
        if Y is not None:
            self.y_min_, self.y_max_ = self._fit_min_max(Y, axis=0)
        return self

    def transform(self, X: np.ndarray, Y: np.ndarray | None = None):
        Xn = self._transform(X, self.x_min_, self.x_max_)
        Yn = None if Y is None or self.y_min_ is None else self._transform(Y, self.y_min_, self.y_max_)
        return Xn, Yn

    def fit_transform(self, X: np.ndarray, Y: np.ndarray | None = None):
        self.fit(X, Y)
        return self.transform(X, Y)

    # Inverse for predictions (NumPy)
    def inverse_y(self, Yn: np.ndarray):
        return self._inverse(Yn, self.y_min_, self.y_max_)

    # Torch helpers (optional)
    def torch_params(self, device=None, dtype=torch.float32):
        x_min = torch.from_numpy(self.x_min_).to(device=device, dtype=dtype)
        x_max = torch.from_numpy(self.x_max_).to(device=device, dtype=dtype)
        y_min = torch.from_numpy(self.y_min_) .to(device=device, dtype=dtype) if self.y_min_ is not None else None
        y_max = torch.from_numpy(self.y_max_) .to(device=device, dtype=dtype) if self.y_max_ is not None else None
        return x_min, x_max, y_min, y_max

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
