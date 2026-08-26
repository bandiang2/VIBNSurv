"""
Dataset loaders used in the paper plus the semi-synthetic noise protocol.

All loaders return ``(x, t, e, feature_names)`` where

    x : (n, p) float array of covariates (one-hot encoded, NaNs imputed, NOT scaled)
    t : (n,)   observed time
    e : (n,)   event indicator, 1 = event observed, 0 = right-censored

Standardisation is intentionally left to the caller so that it can be fitted
on the training folds only (see ``examples/quickstart.py``).

Datasets
--------
SUPPORT   9,105 patients / 30 covariates (support2.csv, downloaded from hbiostat.org)
FLCHAIN   6,524 patients /  8 covariates (flchain.csv, downloaded from Rdatasets)
GBSG      2,232 patients /  7 covariates (via pycox)
METABRIC  1,904 patients /  9 covariates (via pycox; not in the paper, kept for convenience)
TCGA      user-provided CSV (see ``load_tcga_csv``)
"""

import io
import os
import urllib.request
import zipfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

SUPPORT_URL = "https://hbiostat.org/data/repo/support2csv.zip"
FLCHAIN_URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
               "master/csv/survival/flchain.csv")
SUPPORT_NUMERIC = ["age", "num.co", "meanbp", "wblc", "hrt", "resp", "temp", "pafi",
                   "alb", "bili", "crea", "sod", "ph", "glucose", "bun", "urine",
                   "adlp", "adls"]
SUPPORT_CATEGORICAL = ["sex", "dzgroup", "dzclass", "income", "race", "ca"]

Dataset = Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _split_xte(df: pd.DataFrame) -> Dataset:
    cov = df.drop(columns=["duration", "event"])
    return (cov.values.astype(float),
            df["duration"].values.astype(float),
            df["event"].values.astype(int),
            cov.columns.tolist())


def _download_support(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, "support2.csv")
    if not os.path.exists(target):
        print(f"Downloading SUPPORT from {SUPPORT_URL} ...")
        with urllib.request.urlopen(SUPPORT_URL) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                with zf.open("support2.csv") as src, open(target, "wb") as dst:
                    dst.write(src.read())
    return target


# --------------------------------------------------------------------------- #
# Real-world datasets
# --------------------------------------------------------------------------- #
def load_support(data_dir: str = DEFAULT_DATA_DIR) -> Dataset:
    """SUPPORT (Knaus et al., 1995): 9,105 seriously ill hospitalised adults."""
    from sklearn.impute import SimpleImputer

    raw = pd.read_csv(_download_support(data_dir))
    num = raw[SUPPORT_NUMERIC]
    cat = pd.get_dummies(raw[SUPPORT_CATEGORICAL], dtype=float)
    x = np.concatenate([num.values, cat.values], axis=1)
    x = SimpleImputer(strategy="mean").fit_transform(x)

    df = pd.DataFrame(x, columns=SUPPORT_NUMERIC + cat.columns.tolist())
    df["duration"] = raw["d.time"].values
    df["event"] = raw["death"].values
    return _split_xte(df)


def _download_flchain(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, "flchain.csv")
    if not os.path.exists(target):
        print(f"Downloading FLCHAIN from {FLCHAIN_URL} ...")
        urllib.request.urlretrieve(FLCHAIN_URL, target)
    return target


def load_flchain(data_dir: str = DEFAULT_DATA_DIR, eps: float = 1e-8) -> Dataset:
    """FLCHAIN (Kyle et al., 2006): serum free light chain assay, 6,524 subjects.

    Preprocessing mirrors ``pycox.datasets.flchain`` (drop ``chapter``, drop
    rows with missing creatinine, binarise sex) and then, as in the paper,
    drops ``sample.yr`` and one-hot encodes ``flc.grp``.
    """
    df = pd.read_csv(_download_flchain(data_dir))
    df = df.drop(columns=[c for c in ("chapter", "Unnamed: 0", "rownames") if c in df.columns])
    df = df.loc[df["creatinine"].notna()].reset_index(drop=True)
    df["sex"] = (df["sex"] == "M").astype(float)
    df = df.rename(columns={"futime": "duration", "death": "event"})
    df = df.drop(columns=["sample.yr"])
    df["duration"] = df["duration"].astype(float) + eps  # avoid t = 0
    df = pd.get_dummies(df, columns=["flc.grp"], dtype=float)
    return _split_xte(df)


def load_gbsg() -> Dataset:
    """GBSG (Schumacher et al., 1994): German Breast Cancer Study Group, 2,232 patients."""
    from pycox import datasets

    df = datasets.gbsg.read_df()
    df = pd.get_dummies(df, columns=["x0", "x1", "x2"], dtype=float)
    return _split_xte(df)


def load_metabric(eps: float = 1e-8) -> Dataset:
    """METABRIC breast-cancer cohort, 1,904 patients (via pycox)."""
    from pycox import datasets

    df = datasets.metabric.read_df()
    df["duration"] = df["duration"].astype(float) + eps
    return _split_xte(df)


def load_tcga_csv(path: str, time_col: str = "OS_days", event_col: str = "OS",
                  drop_cols: Tuple[str, ...] = ("patient_id",)) -> Dataset:
    """Generic loader for a TCGA-style multi-omics CSV (e.g. TCGA-LGG).

    Expects one row per patient with a time column, an event column and any
    number of numeric / categorical covariate columns. Categorical columns
    (dtype object) are one-hot encoded; constant columns are dropped.

    The TCGA-LGG multi-omics matrix used in the paper (411 patients,
    20,216 features: clinical + gene expression + miRNA + mutation) is not
    redistributed here. See the README for how to obtain it.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    t = df.pop(time_col).astype(float).values
    e = df.pop(event_col).astype(int).values

    obj_cols = df.columns[df.dtypes == object].tolist()
    df[obj_cols] = df[obj_cols].fillna("MISSING")
    df = pd.get_dummies(df, columns=obj_cols, dtype=float)
    df = df.fillna(df.mean(numeric_only=True))
    df = df.loc[:, (df != df.iloc[0]).any()]  # drop constant columns
    return df.values.astype(float), t, e, df.columns.tolist()


LOADERS = {
    "support": load_support,
    "flchain": load_flchain,
    "gbsg": load_gbsg,
    "metabric": load_metabric,
}


def load_dataset(name: str, **kwargs) -> Dataset:
    """Load one of the built-in datasets by name (case-insensitive)."""
    key = name.lower()
    if key not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(LOADERS)}")
    return LOADERS[key](**kwargs)


# --------------------------------------------------------------------------- #
# Semi-synthetic noise protocol (Section IV-A of the paper)
# --------------------------------------------------------------------------- #
NOISE_LEVELS = {1: 1, 2: 3, 3: 5}  # level -> number of noise modalities


def add_noise_modalities(x: np.ndarray, n_modalities: int, kappa: int = 10,
                         feature_names: Optional[List[str]] = None,
                         random_state: int = 42) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Append ``n_modalities`` blocks of i.i.d. N(0, 1) noise, each of width
    ``kappa``, to the covariate matrix.

    Paper protocol: Level 1 = 1 modality, Level 2 = 3, Level 3 = 5, kappa = 10.
    """
    rng = np.random.RandomState(random_state)
    noise = rng.normal(size=(x.shape[0], n_modalities * kappa))
    x_noisy = np.concatenate([x, noise], axis=1)
    if feature_names is not None:
        names = feature_names + [f"noise_m{m}_{j}" for m in range(n_modalities)
                                 for j in range(kappa)]
        return x_noisy, names
    return x_noisy, None


def load_noisy_dataset(name: str, level: int, kappa: int = 10,
                       random_state: int = 42, **kwargs) -> Dataset:
    """Real dataset + synthetic Gaussian noise modalities at the given level (1-3)."""
    if level not in NOISE_LEVELS:
        raise ValueError(f"level must be one of {sorted(NOISE_LEVELS)}")
    x, t, e, names = load_dataset(name, **kwargs)
    x, names = add_noise_modalities(x, NOISE_LEVELS[level], kappa, names, random_state)
    return x, t, e, names
