"""
Evaluation helpers wrapping scikit-survival metrics.

The paper reports the time-dependent concordance index C^td (Antolini et al.,
2005) and the cumulative/dynamic AUC (Hung & Chiang, 2010) at the 25th, 50th
and 75th quantiles of the observed event times.
"""

from typing import Dict, Sequence

import numpy as np


def to_structured(e: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert (event, time) arrays to the structured array sksurv expects."""
    return np.array([(bool(ei), float(ti)) for ei, ti in zip(e, t)],
                    dtype=[("e", bool), ("t", float)])


def event_time_quantiles(t: np.ndarray, e: np.ndarray,
                         horizons: Sequence[float] = (0.25, 0.5, 0.75)) -> np.ndarray:
    """Evaluation times = quantiles of the *uncensored* event times."""
    return np.quantile(t[e == 1], horizons)


def evaluate(risk: np.ndarray, survival: np.ndarray, times: Sequence[float],
             t_train: np.ndarray, e_train: np.ndarray,
             t_test: np.ndarray, e_test: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute C^td, AUC and Brier score at each evaluation time.

    Parameters
    ----------
    risk : (n_test, len(times)) predicted P(T <= t | x)
    survival : (n_test, len(times)) predicted S(t | x)
    times : evaluation horizons (on the *original* time scale)
    """
    from sksurv.metrics import (brier_score, concordance_index_ipcw,
                                cumulative_dynamic_auc)

    et_train = to_structured(e_train, t_train)
    et_test = to_structured(e_test, t_test)

    ctd = np.array([concordance_index_ipcw(et_train, et_test, risk[:, i], tau)[0]
                    for i, tau in enumerate(times)])
    auc = np.array([cumulative_dynamic_auc(et_train, et_test, risk[:, i], tau)[0][0]
                    for i, tau in enumerate(times)])
    brier = brier_score(et_train, et_test, survival, times)[1]
    return {"ctd": ctd, "auc": auc, "brier": brier}


def print_results(results: Dict[str, np.ndarray],
                  horizons: Sequence[float] = (0.25, 0.5, 0.75)) -> None:
    print(f"{'Quantile':>10} {'C-td':>8} {'AUC':>8} {'Brier':>8}")
    for i, h in enumerate(horizons):
        print(f"{h:>10.2f} {results['ctd'][i]:>8.3f} {results['auc'][i]:>8.3f} "
              f"{results['brier'][i]:>8.3f}")
