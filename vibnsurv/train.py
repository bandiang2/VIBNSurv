"""Training loop for VIBNSurv (Algorithm 1 of the paper)."""

from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .losses import vibnsurv_loss
from .model import VIBNSurvNet


def get_optimizer(model: torch.nn.Module, lr: float, name: str = "Adam",
                  **kwargs) -> torch.optim.Optimizer:
    name = name.lower()
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    if name not in optimizers:
        raise NotImplementedError(f"Optimizer '{name}' is not supported. "
                                  f"Choose from {list(optimizers)}.")
    return optimizers[name](model.parameters(), lr=lr, **kwargs)


def train_vibnsurv(model: VIBNSurvNet,
                   x_train: torch.Tensor, t_train: torch.Tensor, e_train: torch.Tensor,
                   x_val: torch.Tensor, t_val: torch.Tensor, e_val: torch.Tensor,
                   n_iter: int = 1000, lr: float = 1e-3, weight_decay: float = 1e-3,
                   bs: int = 100, optimizer: str = "Adam", patience: int = 3,
                   shuffle: bool = True, verbose: bool = True,
                   device: Optional[torch.device] = None) -> VIBNSurvNet:
    """Train `model` with mini-batch gradient descent and early stopping on
    the validation loss. Returns the model with the best validation loss.

    Parameters
    ----------
    n_iter : int
        Maximum number of epochs.
    patience : int
        Number of consecutive epochs without validation improvement before
        stopping.
    shuffle : bool
        Reshuffle the training set at every epoch.
    """
    device = device or next(model.parameters()).device
    x_train, t_train, e_train = x_train.to(device), t_train.to(device), e_train.to(device)
    x_val, t_val, e_val = x_val.to(device), t_val.to(device), e_val.to(device)

    optim = get_optimizer(model, lr, optimizer, weight_decay=weight_decay)

    best_loss, previous_loss, wait = np.inf, np.inf, 0
    best_state = deepcopy(model.state_dict())

    n = x_train.shape[0]
    n_batches = int(np.ceil(n / bs))
    index = np.arange(n)
    epochs = tqdm(range(n_iter), disable=not verbose)

    for _ in epochs:
        if shuffle:
            np.random.shuffle(index)

        model.train()
        for j in range(n_batches):
            batch = index[j * bs:(j + 1) * bs]
            if len(batch) == 0:
                continue
            optim.zero_grad()
            loss = vibnsurv_loss(model, x_train[batch], t_train[batch], e_train[batch])
            loss.backward()
            optim.step()

        model.eval()
        val_loss = vibnsurv_loss(model, x_val, t_val, e_val).item()
        epochs.set_description(f"Val loss: {val_loss:.4f}")

        if val_loss < previous_loss:
            wait = 0
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
        elif wait >= patience:
            break
        else:
            wait += 1
        previous_loss = val_loss

    model.load_state_dict(best_state)
    return model.eval()
