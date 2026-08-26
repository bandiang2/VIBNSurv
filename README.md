# VIBNSurv — Variational Information Bottleneck Neural Survival Model

Official PyTorch implementation of

> **Towards Robust Time-to-Event Prediction: Integrating the Variational Information Bottleneck with Neural Survival Model**
> Armand Bandiang Massoua, Abdoulaye Banire Diallo, Mohamed Bouguessa
> *International Joint Conference on Neural Networks (IJCNN), 2024*
> [DOI: 10.1109/IJCNN60899.2024.10651066](https://doi.org/10.1109/IJCNN60899.2024.10651066) · [PDF](assets/VIBNSurv_IJCNN2024.pdf)

<p align="center">
  <img src="assets/vibnsurv_architecture.png" width="900" alt="VIBNSurv architecture">
</p>

## Overview

Neural survival models learn rich feature embeddings but are sensitive to noisy and irrelevant covariates, which is a real problem in small clinical datasets. **VIBNSurv** addresses this by inserting a **Variational Information Bottleneck** (VIB) between the encoder and the survival head:

1. An encoder $f_\theta$ maps covariates $x$ to a Gaussian latent $z \sim \mathcal{N}(\mu_x, \Sigma_x)$.
2. A KL penalty $\beta \, \mathrm{KL}[p(z|x)\,\|\,\mathcal{N}(0, I)]$ compresses $z$, discarding information about $x$ that is not useful for predicting the event time.
3. A **monotone mixed network** (positive weights) takes $(z, t)$ and outputs $\hat S(t|z) = 1 - \sigma(h(t,z))$; the density $\hat f(t|z) = -\partial \hat S/\partial t$ is obtained by autograd, so the **full right-censored likelihood** is optimised directly with **no assumption on the survival distribution**.

Training objective (Eq. 10):

$$
\mathcal{L} = -\frac{1}{N}\sum_i \Big[\delta_i \log \hat f(t_i|z_i) + (1-\delta_i)\log \hat S(t_i|z_i)\Big] + \beta \, \mathrm{KL}\big[p(z|x)\,\|\,r(z)\big]
$$

Across SUPPORT, FLCHAIN, GBSG and TCGA-LGG, VIBNSurv outperforms CoxPH, DeepSurv, DeepHit, DSM, DCM and SuMo-net in $C^{td}$ and time-dependent AUC, and degrades far less when synthetic noise modalities are added to the inputs.

## Installation

```bash
git clone https://github.com/<your-username>/VIBNSurv.git
cd VIBNSurv
pip install -e .            # core package
pip install pycox           # needed for the GBSG / METABRIC loaders
```

Requires Python ≥ 3.8 and PyTorch ≥ 1.10. GPU is used automatically when available.

## Quick start

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from vibnsurv import VIBNSurv, datasets, metrics

# 1. Data  (x: covariates, t: observed time, e: 1 = event / 0 = censored)
x, t, e, names = datasets.load_gbsg()
x_tr, x_te, t_tr, t_te, e_tr, e_te = train_test_split(x, t, e, test_size=0.2, random_state=42)

# 2. Standardise covariates and rescale time to [0, 1] (fit on training data only)
xs = StandardScaler().fit(x_tr);            x_tr, x_te = xs.transform(x_tr), xs.transform(x_te)
ts = MinMaxScaler().fit(t_tr.reshape(-1, 1)); scale = lambda a: ts.transform(np.reshape(a, (-1, 1))).ravel()

# 3. Train
model = VIBNSurv(layers=[100, 100], layers_surv=[50], beta=1e-4, sample_size=5)
model.fit(x_tr, scale(t_tr), e_tr, n_iter=500, lr=1e-3, bs=128)   # 15 % of train used for early stopping

# 4. Predict & evaluate at the 25 / 50 / 75 % event-time quantiles
times = metrics.event_time_quantiles(t_tr, e_tr)
S = model.predict_survival(x_te, scale(times))       # (n_test, 3)
results = metrics.evaluate(1 - S, S, times, t_tr, e_tr, t_te, e_te)
metrics.print_results(results)
```

Or from the command line:

```bash
python examples/quickstart.py --dataset gbsg
python examples/quickstart.py --dataset support --noise-level 2 --beta 1e-5 --epochs 300
```

## API

### `VIBNSurv(layers, layers_surv, beta, sample_size, dropout, cuda, seed)`

| Argument       | Default           | Description |
|----------------|-------------------|-------------|
| `layers`       | `[100, 100, 100]` | Encoder hidden sizes; the last value is the bottleneck dimension $k$ |
| `layers_surv`  | `[100]`           | Hidden sizes of the monotone mixed network |
| `beta`         | `1e-4`            | KL weight (compression strength). `0` recovers the no-bottleneck ablation |
| `sample_size`  | `5`               | Monte-Carlo samples of $z$ per example during training ($M$ in Alg. 1) |
| `dropout`      | `0.0`             | Dropout in both networks |

| Method | Description |
|--------|-------------|
| `fit(x, t, e, vsize=0.15, val_data=None, n_iter, lr, bs, weight_decay, optimizer, patience)` | Train with early stopping on validation loss |
| `predict_survival(x, t)` / `predict_risk(x, t)` | $\hat S(t|x)$ / $1-\hat S(t|x)$ for a float or list of times → `(n, len(t))` |
| `compute_nll(x, t, e)` | Censored NLL + $\beta$·KL on held-out data (used for model selection) |
| `embed(x)` | Deterministic bottleneck representation $\mu_x$ → `(n, k)` |
| `save(path)` / `VIBNSurv.load(path)` | Persistence |

The raw network is also exposed as `vibnsurv.VIBNSurvNet` for custom training loops.

> **Note on `beta` scaling.** As in the original code, the KL term is *summed* over the batch while the likelihood is *averaged*, so the effective strength of `beta` grows with batch size. Re-tune `beta` if you change `bs` substantially.

## Datasets

All datasets download automatically to `data/` on first use.

| Name | n | Covariates (after one-hot) | Censored | Loader |
|------|---|----------------------------|----------|--------|
| SUPPORT  | 9,105 | 44 | 32 % | `datasets.load_support()` |
| FLCHAIN  | 6,524 | 16 | 70 % | `datasets.load_flchain()` |
| GBSG     | 2,232 | 11 | 43 % | `datasets.load_gbsg()` |
| METABRIC | 1,904 |  9 | 42 % | `datasets.load_metabric()` |
| TCGA-LGG |   411 | 20,216 | 76 % | `datasets.load_tcga_csv(path)` |

The TCGA-LGG multi-omics matrix (clinical + gene expression + miRNA + mutation) is not redistributed. Download it from the [GDC portal](https://portal.gdc.cancer.gov/) or the preprocessed release of [Wissel et al.](https://github.com/BoevaLab/Multi-omics-noise-resistance), save it as a CSV with `OS_days` / `OS` columns, and pass the path to `load_tcga_csv`.

**Semi-synthetic noise (RQ2).** `datasets.load_noisy_dataset(name, level)` appends $\mathcal{N}(0,1)$ noise modalities of width $\kappa = 10$: level 1 = 1 modality, level 2 = 3, level 3 = 5.

## Reproducing the paper

All scripts use the paper's protocol: 5-fold CV with identical folds for every model, 20 % of each training fold held out for early stopping and model selection on validation NLL, random search over the grid of Table II.

```bash
# RQ1 – real-world datasets (Table III)
python experiments/run_cv.py --dataset support --n-configs 100
python experiments/run_cv.py --dataset flchain --n-configs 100
python experiments/run_cv.py --dataset gbsg    --n-configs 100

# RQ2 – noise robustness (Tables IV–V)
python experiments/run_cv.py --dataset support --noise-level 1   # also 2, 3
python experiments/run_cv.py --dataset gbsg    --noise-level 1

# RQ3 – ablation without compression (Table VI) and β sweep (Fig. 2)
python experiments/run_cv.py --dataset gbsg --fixed-beta 0
python experiments/beta_sweep.py --dataset gbsg
```

Results are written to `results/*.csv` (one row per fold, best hyper-parameters included).

### Baselines

`experiments/benchmark.py` runs CoxPH, DeepSurv, DeepHit, DSM, DCM, SuMo-net and VIBNSurv under the same folds:

```bash
pip install lifelines torchtuples pycox
# SuMo-net and the DSM fork it depends on (the original autonlab repo no longer ships the `dsm` package)
git clone https://github.com/Jeanselme/SuMo-net third_party/SuMo-net
git clone https://github.com/Jeanselme/DeepSurvivalMachines third_party/SuMo-net/DeepSurvivalMachines

python experiments/benchmark.py --dataset gbsg --models all
python experiments/benchmark.py --dataset support --noise-level 3 --models cox sumo vibnsurv
```

### Reported results ($C^{td}$, mean over 5 folds; Table III)

| Model     | SUPPORT 25/50/75 | FLCHAIN 25/50/75 | GBSG 25/50/75 | TCGA-LGG 25/50/75 |
|-----------|------------------|------------------|---------------|-------------------|
| CoxPH     | .682 / .666 / .665 | .797 / .792 / .785 | .721 / .694 / .675 | – |
| DeepSurv  | .685 / .670 / .666 | .799 / .798 / .793 | .703 / .677 / .657 | .850 / .853 / .765 |
| DeepHit   | .746 / .689 / .613 | .787 / .788 / .760 | .715 / .671 / .616 | .820 / .842 / .727 |
| DSM       | .728 / .699 / .651 | .776 / .774 / .770 | .737 / .706 / .684 | .852 / .869 / .758 |
| DCM       | .727 / .693 / .668 | .773 / .772 / .773 | .691 / .668 / .646 | – |
| SuMo-net  | .753 / .712 / .679 | .798 / .798 / .793 | .745 / .707 / .683 | .864 / .810 / .741 |
| **VIBNSurv** | **.767 / .722 / .683** | **.807 / .802 / .796** | **.747 / .713 / .685** | **.878 / .879 / .814** |

## Repository structure

```
vibnsurv/
  model.py       VIBNSurvNet, PositiveLinear, encoder / monotone MLP builders
  losses.py      right-censored NLL + β·KL
  train.py       mini-batch training with early stopping (Algorithm 1)
  api.py         VIBNSurv: scikit-learn-style fit / predict / save / load
  datasets.py    SUPPORT, FLCHAIN, GBSG, METABRIC, TCGA loaders + noise protocol
  metrics.py     C^td, cumulative-dynamic AUC, Brier score
examples/        quickstart.py
experiments/     run_cv.py, benchmark.py, beta_sweep.py
tests/           pytest sanity checks  (python -m pytest tests/)
assets/          paper PDF and figure
```

## Citation

```bibtex
@inproceedings{massoua2024vibnsurv,
  title     = {Towards Robust Time-to-Event Prediction: Integrating the Variational Information Bottleneck with Neural Survival Model},
  author    = {Bandiang Massoua, Armand and Banire Diallo, Abdoulaye and Bouguessa, Mohamed},
  booktitle = {2024 International Joint Conference on Neural Networks (IJCNN)},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/IJCNN60899.2024.10651066}
}
```

## Acknowledgements

The monotone survival network builds on [SuMo-net](https://github.com/Jeanselme/SuMo-net) (Rindt et al., AISTATS 2022) and the neural CDF estimator of Chilinski & Silva (UAI 2020). The VIB formulation follows [Alemi et al. (2017)](https://arxiv.org/abs/1612.00410). Dataset loaders rely on [pycox](https://github.com/havakv/pycox) and [hbiostat](https://hbiostat.org/data/).

## License

MIT — see [LICENSE](LICENSE).
