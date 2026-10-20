"""
ml/dataset_generator.py
=======================
Génération du dataset d'apprentissage supervisé pour le surrogate model.

Chaque exemple est un vecteur de features (S₀, K, r, σ, T, type)
associé à son label y = prix Black-Scholes exact.

Le surrogate model apprend la fonction de pricing :
    f : (S₀, K, r, σ, T) → prix

Référence : Ludkovski, M. (2018). Kriging Metamodels and Experimental Design
            for Bermudan Option Pricing. Journal of Computational Finance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal
import time
import os
import sys

# Import relatif robuste
_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from core.black_scholes import black_scholes_price


# ── Dataclass configuration ────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Configuration du plan d'expérience pour la génération du dataset."""
    n_samples      : int   = 2000
    option_types   : list  = None      # ['call', 'put'] par défaut
    S_range        : tuple = (50.0, 200.0)
    K_range        : tuple = (40.0, 220.0)
    r_range        : tuple = (0.0, 0.10)
    sigma_range    : tuple = (0.05, 0.80)
    T_range        : tuple = (0.10, 3.0)
    seed           : int   = 42
    sampling       : str   = "uniform"  # 'uniform' | 'lhs' (Latin Hypercube)

    def __post_init__(self):
        if self.option_types is None:
            self.option_types = ["call", "put"]


# ── Featurisation ──────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "S0", "K", "r", "sigma", "T", "type_num",
    "moneyness",       # ln(S/K)
    "sqrt_T",          # √T
    "vol_sqrt_T",      # σ·√T  (vol surface proxy)
    "discount",        # e^(-rT)
    "S_over_K",        # S/K
    "rT",              # r·T
]


def featurize(
    S: float, K: float, r: float, sigma: float, T: float,
    option_type: str,
) -> np.ndarray:
    """
    Transforme les paramètres bruts en vecteur de features enrichi.

    Features : S, K, r, σ, T, type_num, ln(S/K), √T, σ√T, e^(-rT), S/K, r·T

    Parameters
    ----------
    S, K, r, sigma, T : paramètres de l'option
    option_type       : 'call' (→ 1) ou 'put' (→ 0)

    Returns
    -------
    np.ndarray de shape (12,)
    """
    type_num   = 1.0 if option_type == "call" else 0.0
    moneyness  = np.log(S / K) if K > 0 else 0.0
    sqrt_T     = np.sqrt(max(T, 1e-8))
    vol_sqrt_T = sigma * sqrt_T
    discount   = np.exp(-r * T)
    S_over_K   = S / K if K > 0 else 1.0
    rT         = r * T

    return np.array([
        S, K, r, sigma, T, type_num,
        moneyness, sqrt_T, vol_sqrt_T,
        discount, S_over_K, rT,
    ], dtype=np.float64)


def featurize_batch(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorise `featurize` sur un DataFrame complet.

    Parameters
    ----------
    df : DataFrame avec colonnes S0, K, r, sigma, T, option_type

    Returns
    -------
    np.ndarray de shape (n, 12)
    """
    S      = df["S0"].values
    K      = df["K"].values
    r      = df["r"].values
    sigma  = df["sigma"].values
    T      = df["T"].values
    tnum   = (df["option_type"] == "call").astype(float).values

    moneyness  = np.log(np.clip(S / K, 1e-8, np.inf))
    sqrt_T     = np.sqrt(np.clip(T, 1e-8, np.inf))
    vol_sqrt_T = sigma * sqrt_T
    discount   = np.exp(-r * T)
    S_over_K   = np.clip(S / K, 1e-8, np.inf)
    rT         = r * T

    return np.column_stack([
        S, K, r, sigma, T, tnum,
        moneyness, sqrt_T, vol_sqrt_T,
        discount, S_over_K, rT,
    ])


# ── Génération du dataset ──────────────────────────────────────────────────

def generate_dataset(config: DatasetConfig | None = None, **kwargs) -> pd.DataFrame:
    """
    Génère un dataset d'apprentissage supervisé pour le surrogate model.

    Parameters
    ----------
    config : DatasetConfig ou None (utilise les kwargs pour construction)
    **kwargs : passés à DatasetConfig si config=None

    Returns
    -------
    pd.DataFrame avec colonnes :
        S0, K, r, sigma, T, option_type,
        price (label Black-Scholes exact),
        moneyness, vol_sqrt_T, discount,
        intrinsic, time_value

    Examples
    --------
    >>> df = generate_dataset(n_samples=1000, seed=42)
    >>> len(df)
    1000
    >>> "price" in df.columns
    True
    """
    if config is None:
        config = DatasetConfig(**kwargs)

    t_start = time.perf_counter()
    rng     = np.random.default_rng(config.seed)

    N = config.n_samples

    if config.sampling == "lhs":
        samples = _latin_hypercube(N, 5, rng)
    else:
        samples = rng.random((N, 5))

    # ── Décoder les échantillons ───────────────────────────────────────────
    def scale(x, lo, hi): return lo + x * (hi - lo)

    S_arr     = scale(samples[:, 0], *config.S_range)
    K_arr     = scale(samples[:, 1], *config.K_range)
    r_arr     = scale(samples[:, 2], *config.r_range)
    sigma_arr = scale(samples[:, 3], *config.sigma_range)
    T_arr     = scale(samples[:, 4], *config.T_range)

    # Type alterné ou aléatoire
    if len(config.option_types) == 2:
        type_idx  = rng.integers(0, 2, N)
        type_arr  = [config.option_types[i] for i in type_idx]
    else:
        type_arr  = [config.option_types[0]] * N

    # ── Calcul vectorisé des labels ────────────────────────────────────────
    prices, intrinsics, time_values = [], [], []
    valid_mask = []

    for i in range(N):
        try:
            res = black_scholes_price(
                S_arr[i], K_arr[i], r_arr[i], sigma_arr[i], T_arr[i],
                option_type=type_arr[i]
            )
            if np.isfinite(res.price) and res.price >= 0:
                prices.append(res.price)
                intrinsics.append(res.intrinsic_value)
                time_values.append(res.time_value)
                valid_mask.append(i)
        except Exception:
            continue

    idx = np.array(valid_mask)
    S_arr     = S_arr[idx]
    K_arr     = K_arr[idx]
    r_arr     = r_arr[idx]
    sigma_arr = sigma_arr[idx]
    T_arr     = T_arr[idx]
    type_arr  = [type_arr[i] for i in valid_mask]

    df = pd.DataFrame({
        "S0"          : S_arr,
        "K"           : K_arr,
        "r"           : r_arr,
        "sigma"       : sigma_arr,
        "T"           : T_arr,
        "option_type" : type_arr,
        "price"       : prices,
        "intrinsic"   : intrinsics,
        "time_value"  : time_values,
    })

    # ── Features dérivées ─────────────────────────────────────────────────
    df["moneyness"]  = np.log(df["S0"] / df["K"])
    df["vol_sqrt_T"] = df["sigma"] * np.sqrt(df["T"])
    df["discount"]   = np.exp(-df["r"] * df["T"])
    df["S_over_K"]   = df["S0"] / df["K"]
    df["type_num"]   = (df["option_type"] == "call").astype(int)

    elapsed = time.perf_counter() - t_start
    print(f"  Dataset généré : {len(df):,} exemples valides / {N:,} tentatives")
    print(f"  Calls : {(df.option_type=='call').sum():,}  |  Puts : {(df.option_type=='put').sum():,}")
    print(f"  Prix moyen BS : {df.price.mean():.4f}  |  Écart-type : {df.price.std():.4f}")
    print(f"  Temps : {elapsed*1000:.1f} ms")

    return df


def _latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Échantillonnage Latin Hypercube : meilleure couverture de l'espace."""
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.random(n)) / n
    return result


# ── Split et normalisation ─────────────────────────────────────────────────

def train_test_split_df(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare le dataset en jeu d'entraînement et de test.

    Parameters
    ----------
    df          : DataFrame complet (sortie de generate_dataset)
    train_ratio : Fraction pour le train (défaut 0.80)
    seed        : Graine aléatoire

    Returns
    -------
    (df_train, df_test)
    """
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train     = int(len(df_shuffled) * train_ratio)
    return df_shuffled.iloc[:n_train].copy(), df_shuffled.iloc[n_train:].copy()


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple:
    """
    Normalise les features en Z-scores (μ=0, σ=1) basé sur le train set.

    Parameters
    ----------
    X_train : Features d'entraînement (n_train, n_features)
    X_test  : Features de test (optionnel)

    Returns
    -------
    (X_train_norm, X_test_norm, mean_, std_) ou
    (X_train_norm, mean_, std_) si X_test=None
    """
    mean_ = X_train.mean(axis=0)
    std_  = X_train.std(axis=0)
    std_[std_ < 1e-10] = 1.0            # évite division par zéro

    X_train_norm = (X_train - mean_) / std_
    if X_test is not None:
        X_test_norm = (X_test - mean_) / std_
        return X_train_norm, X_test_norm, mean_, std_
    return X_train_norm, mean_, std_


def save_dataset(df: pd.DataFrame, path: str = "data/ml_dataset.csv") -> None:
    """Sauvegarde le dataset au format CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    df.to_csv(path, index=False)
    print(f"  Dataset sauvegardé → {path} ({len(df):,} lignes)")


def load_dataset(path: str = "data/ml_dataset.csv") -> pd.DataFrame:
    """Charge le dataset depuis un fichier CSV."""
    df = pd.read_csv(path)
    print(f"  Dataset chargé ← {path} ({len(df):,} lignes)")
    return df


# ── Point d'entrée de test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Generator — Test")
    print("=" * 60)
    cfg = DatasetConfig(n_samples=500, seed=42, sampling="lhs")
    df  = generate_dataset(cfg)
    print(f"\n  Colonnes : {list(df.columns)}")
    print(f"\n  Aperçu :")
    print(df[["S0","K","r","sigma","T","option_type","price"]].head(5).to_string())

    df_tr, df_te = train_test_split_df(df)
    print(f"\n  Train : {len(df_tr):,} | Test : {len(df_te):,}")

    X_all = featurize_batch(df)
    print(f"\n  Features shape : {X_all.shape}")
    print(f"  Feature names  : {FEATURE_NAMES}")
    print("=" * 60)
