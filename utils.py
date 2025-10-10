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

class EarlyStoppingTrainLoss:
    def __init__(self, patience=200, min_delta=0.0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def step(self, train_loss, model):
        # mejora si disminuye al menos min_delta
        if self.best_loss is None or (self.best_loss - train_loss) > self.min_delta:
            self.best_loss = train_loss
            self.counter = 0
            if self.restore_best:
                # snapshot en CPU para seguridad
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)

