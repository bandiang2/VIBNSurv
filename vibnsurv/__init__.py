"""
VIBNSurv: Variational Information Bottleneck Neural Survival Model.

Reference
---------
A. Bandiang Massoua, A. Banire Diallo, M. Bouguessa,
"Towards Robust Time-to-Event Prediction: Integrating the Variational
Information Bottleneck with Neural Survival Model", IJCNN 2024.
https://doi.org/10.1109/IJCNN60899.2024.10651066
"""

from .api import VIBNSurv
from .model import VIBNSurvNet, PositiveLinear
from .losses import vibnsurv_loss, right_censored_nll
from .train import train_vibnsurv
from . import datasets, metrics

__version__ = "1.0.0"
__all__ = ["VIBNSurv", "VIBNSurvNet", "PositiveLinear", "vibnsurv_loss",
           "right_censored_nll", "train_vibnsurv", "datasets", "metrics"]
