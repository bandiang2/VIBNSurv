"""
Network components of VIBNSurv.

The architecture follows Fig. 1 of the paper:

    x  --encoder f_theta-->  h  --IB layer-->  z ~ N(mu(h), Sigma(h))
    (z, t) --mixed network (positive weights)-->  h(t, z)
    S(t | z) = 1 - sigmoid(h(t, z))
    f(t | z) = -dS(t | z)/dt        (via autograd)

The mixed network has strictly positive weights, which guarantees that the
survival function is monotonically decreasing in t (Chilinski & Silva, 2020;
Rindt et al., 2022).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class PositiveLinear(nn.Module):
    """Linear layer whose effective weights are the square of the stored
    parameters, hence always non-negative."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.raw_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.raw_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        # sqrt(|w|) so that raw_weight ** 2 starts at the xavier magnitude
        with torch.no_grad():
            self.raw_weight.abs_().sqrt_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.raw_weight ** 2, self.bias)


def build_mlp(input_dim: int, layers: List[int], dropout: float = 0.0,
              activation: nn.Module = nn.Tanh) -> nn.Sequential:
    """Unconstrained MLP used as the encoder f_theta."""
    modules: List[nn.Module] = []
    prev = input_dim
    for hidden in layers:
        modules.append(nn.Linear(prev, hidden))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        modules.append(activation())
        prev = hidden
    return nn.Sequential(*modules)


def build_positive_mlp(input_dim: int, layers: List[int],
                       dropout: float = 0.0) -> nn.Sequential:
    """Monotone MLP (positive weights, Tanh hidden activations) ending with a
    Sigmoid. Used as the mixed survival network."""
    modules: List[nn.Module] = []
    prev = input_dim
    for hidden in layers:
        modules.append(PositiveLinear(prev, hidden, bias=True))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        modules.append(nn.Tanh())
        prev = hidden
    modules[-1] = nn.Sigmoid()  # replace last Tanh by Sigmoid -> output in (0, 1)
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# VIBNSurv network
# --------------------------------------------------------------------------- #
class VIBNSurvNet(nn.Module):
    """Variational Information Bottleneck Neural Survival network.

    Parameters
    ----------
    input_dim : int
        Number of covariates.
    layers : list of int
        Hidden sizes of the encoder f_theta. The last entry is also the
        dimension of the bottleneck z.
    layers_surv : list of int
        Hidden sizes of the mixed (monotone) survival network.
    beta : float
        Weight of the KL (compression) term. beta = 0 disables the bottleneck
        penalty (ablation in Table VI of the paper).
    sample_size : int
        Number of Monte-Carlo samples of z drawn per input during training
        (M in Algorithm 1).
    dropout : float
        Dropout applied inside both networks.
    """

    def __init__(self, input_dim: int, layers: List[int] = (100, 100, 100),
                 layers_surv: List[int] = (100,), beta: float = 1e-4,
                 sample_size: int = 5, dropout: float = 0.0):
        super().__init__()
        layers, layers_surv = list(layers), list(layers_surv)
        self.input_dim = input_dim
        self.beta = beta
        self.sample_size = sample_size
        self.bottleneck_dim = layers[-1]

        # Encoder f_theta : R^p -> R^d
        self.encoder = build_mlp(input_dim, layers, dropout)
        # Information bottleneck layer : R^d -> (mu, sigma) in R^k
        self.to_mu = nn.Linear(layers[-1], self.bottleneck_dim)
        self.to_std = nn.Linear(layers[-1], self.bottleneck_dim)
        # Mixed survival network : (z, t) in R^{k+1} -> (0, 1)
        self.survival_net = build_positive_mlp(self.bottleneck_dim + 1,
                                               layers_surv + [1], dropout)

    # ----- VIB ------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.to_mu(h)
        std = nn.functional.softplus(self.to_std(h))
        return mu, std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """KL[ N(mu, std^2) || N(0, I) ], summed over latent dims and batch.

        Note: the KL is *summed* (not averaged) over the batch, exactly as in
        the original implementation used for the paper's results. The
        likelihood term in `losses.py` is averaged over the batch, so the
        effective strength of `beta` scales with batch size. Keep this in mind
        when transferring beta values across batch sizes.
        """
        return 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2 * (std + 1e-8).log() - 1)

    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Draw `sample_size` samples of z and average them (M in Alg. 1)."""
        eps = torch.randn(self.sample_size, *mu.shape, device=mu.device, dtype=mu.dtype)
        return (mu + std * eps).mean(dim=0)

    # ----- Survival -------------------------------------------------------- #
    def forward(self, x: torch.Tensor, t: torch.Tensor, gradient: bool = False,
                inference: bool = False):
        """
        Returns
        -------
        survival : (n, 1) tensor  S(t | z)
        density  : (n, 1) tensor  f(t | z) = -dS/dt, or None if gradient=False
        kl       : scalar         KL divergence of the bottleneck
        """
        mu, std = self.encode(x)
        kl = self.kl_divergence(mu, std)

        # Deterministic embedding at inference, stochastic during training
        z = mu if inference else self.reparameterize(mu, std)

        t = t.clone().detach().requires_grad_(gradient)
        cdf = self.survival_net(torch.cat((z, t.unsqueeze(1)), dim=1))  # F(t|z)
        survival = 1.0 - cdf

        density: Optional[torch.Tensor] = None
        if gradient:
            # f(t|z) = dF/dt = -dS/dt  (>= 0 thanks to positive weights)
            density = grad(cdf.sum(), t, create_graph=True)[0].unsqueeze(1)

        return survival, density, kl

    @torch.no_grad()
    def predict_survival(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(x, t, gradient=False, inference=True)[0]
