"""
Loss function of VIBNSurv (Eq. 10 of the paper):

    L = - (1/N) sum_i [ delta_i log f(t_i | z_i) + (1 - delta_i) log S(t_i | z_i) ]
        + beta * KL[ p(z|x) || N(0, I) ]
"""

import torch

from .model import VIBNSurvNet


def right_censored_nll(survival: torch.Tensor, density: torch.Tensor,
                       e: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative right-censored log-likelihood, averaged over the batch."""
    survival = survival.clamp(min=eps)
    density = density.clamp(min=eps)
    log_lik = torch.log(survival[e == 0]).sum() + torch.log(density[e != 0]).sum()
    return -log_lik / len(e)


def vibnsurv_loss(model: VIBNSurvNet, x: torch.Tensor, t: torch.Tensor,
                  e: torch.Tensor) -> torch.Tensor:
    """Total training objective: censored NLL + beta * KL."""
    survival, density, kl = model(x, t, gradient=True)
    return right_censored_nll(survival, density, e) + model.beta * kl
