
"""
High-level API for VIBNSurv.

    >>> from vibnsurv import VIBNSurv
    >>> model = VIBNSurv(layers=[100, 100], layers_surv=[50], beta=1e-4)
    >>> model.fit(x_train, t_train, e_train, n_iter=500, lr=1e-3, bs=128)
    >>> S = model.predict_survival(x_test, [t1, t2, t3])   # (n, 3)
    >>> R = model.predict_risk(x_test, [t1, t2, t3])       # 1 - S
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .losses import vibnsurv_loss
from .model import VIBNSurvNet
from .train import train_vibnsurv

ArrayLike = Union[np.ndarray, Sequence[float]]


class VIBNSurv:
    """Variational Information Bottleneck Neural Survival Model.

    Parameters
    ----------
    layers : list of int
        Encoder hidden sizes; last entry = bottleneck dimension k.
    layers_surv : list of int
        Mixed (monotone) survival network hidden sizes.
    beta : float
        Compression strength (KL weight). Paper grid: {1e-2, ..., 1e-6}.
    sample_size : int
        Monte-Carlo samples of z per example during training.
    dropout : float
        Dropout rate.
    cuda : bool
        Use GPU if available.
    seed : int or None
        Seed for torch / numpy RNGs at construction time.
    """

    def __init__(self, layers: List[int] = (100, 100, 100),
                 layers_surv: List[int] = (100,), beta: float = 1e-4,
                 sample_size: int = 5, dropout: float = 0.0,
                 cuda: bool = torch.cuda.is_available(),
                 seed: Optional[int] = 42):
        self.layers = list(layers)
        self.layers_surv = list(layers_surv)
        self.beta = beta
        self.sample_size = sample_size
        self.dropout = dropout
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.fitted = False
        self.net: Optional[VIBNSurvNet] = None

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _seed(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def _to_tensor(self, a: ArrayLike) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=self.device)

    def _split_validation(self, x, t, e, vsize: float, val_data, random_state: int):
        if val_data is not None:
            x_val, t_val, e_val = val_data
            return x, t, e, x_val, t_val, e_val
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(x))
        n_val = int(vsize * len(x))
        val, tr = idx[:n_val], idx[n_val:]
        return x[tr], t[tr], e[tr], x[val], t[val], e[val]

    # ------------------------------------------------------------------ #
    # Fit / evaluate / predict
    # ------------------------------------------------------------------ #
    def fit(self, x: np.ndarray, t: np.ndarray, e: np.ndarray,
            vsize: float = 0.15,
            val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
            random_state: int = 42, **train_kwargs) -> "VIBNSurv":
        """Train the model.

        Parameters
        ----------
        x : (n, p) covariates (should already be standardised).
        t : (n,) observed times. Scaling t to [0, 1] is recommended.
        e : (n,) event indicators (1 = event, 0 = censored).
        vsize : fraction of `x` held out for early stopping if `val_data` is None.
        val_data : optional (x_val, t_val, e_val) tuple.
        **train_kwargs : forwarded to `train_vibnsurv` (n_iter, lr, bs,
            weight_decay, optimizer, patience, shuffle, verbose).
        """
        x, t, e = np.asarray(x, dtype=float), np.asarray(t, dtype=float), np.asarray(e)
        x_tr, t_tr, e_tr, x_val, t_val, e_val = self._split_validation(
            x, t, e, vsize, val_data, random_state)

        self._seed()
        self.net = VIBNSurvNet(x.shape[1], self.layers, self.layers_surv,
                               beta=self.beta, sample_size=self.sample_size,
                               dropout=self.dropout).double().to(self.device)

        self.net = train_vibnsurv(
            self.net,
            self._to_tensor(x_tr), self._to_tensor(t_tr), self._to_tensor(e_tr),
            self._to_tensor(x_val), self._to_tensor(t_val), self._to_tensor(e_val),
            device=self.device, **train_kwargs)
        self.fitted = True
        return self

    def _check_fitted(self, method: str) -> None:
        if not self.fitted:
            raise RuntimeError(f"Call `fit` before `{method}`.")

    def compute_nll(self, x: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
        """Total loss (censored NLL + beta * KL) on the given data."""
        self._check_fitted("compute_nll")
        self.net.eval()
        loss = vibnsurv_loss(self.net, self._to_tensor(x), self._to_tensor(t),
                             self._to_tensor(e))
        return float(loss.item())

    def predict_survival(self, x: np.ndarray, t: Union[float, ArrayLike]) -> np.ndarray:
        """S(t | x) for each row of x and each time in t. Shape (n, len(t))."""
        self._check_fitted("predict_survival")
        times = np.atleast_1d(np.asarray(t, dtype=float))
        xt = self._to_tensor(x)
        self.net.eval()
        out = []
        for t_ in times:
            tt = torch.full((len(xt),), float(t_), dtype=torch.float64, device=self.device)
            out.append(self.net.predict_survival(xt, tt).cpu().numpy())
        return np.concatenate(out, axis=1)

    def predict_risk(self, x: np.ndarray, t: Union[float, ArrayLike]) -> np.ndarray:
        """P(T <= t | x) = 1 - S(t | x)."""
        return 1.0 - self.predict_survival(x, t)

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Deterministic bottleneck representation mu(x). Shape (n, k)."""
        self._check_fitted("embed")
        self.net.eval()
        with torch.no_grad():
            mu, _ = self.net.encode(self._to_tensor(x))
        return mu.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        self._check_fitted("save")
        torch.save({"config": dict(layers=self.layers, layers_surv=self.layers_surv,
                                   beta=self.beta, sample_size=self.sample_size,
                                   dropout=self.dropout, input_dim=self.net.input_dim),
                    "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str, cuda: bool = torch.cuda.is_available()) -> "VIBNSurv":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        obj = cls(layers=cfg["layers"], layers_surv=cfg["layers_surv"], beta=cfg["beta"],
                  sample_size=cfg["sample_size"], dropout=cfg["dropout"], cuda=cuda)
        obj.net = VIBNSurvNet(cfg["input_dim"], cfg["layers"], cfg["layers_surv"],
                              beta=cfg["beta"], sample_size=cfg["sample_size"],
                              dropout=cfg["dropout"]).double().to(obj.device)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        obj.fitted = True
        return obj
