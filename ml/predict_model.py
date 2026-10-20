"""Prédiction du prix par modèle ML sauvegardé."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from .dataset_generator import featurize


def load_model(model_path: str | Path):
    return joblib.load(model_path)


def predict_batch(bundle: dict, rows: list) -> list:
    """Prédit plusieurs options d'un coup. rows : liste de (S, K, r, sigma, T, type)."""
    X = np.vstack([featurize(S, K, r, sig, T, typ) for S, K, r, sig, T, typ in rows])
    scaler = bundle.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)
    pred = bundle["model"].predict(X)
    if bundle.get("target") == "price_over_K":
        pred = pred * np.array([row[1] for row in rows], dtype=float)
    return [max(float(v), 0.0) for v in pred]


def predict_option_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    model_path: str | Path = "models/option_pricing_rf.pkl",
) -> dict:
    """Prédit le prix d'une option avec le modèle ML sauvegardé."""
    t0 = time.perf_counter()
    model_path = Path(model_path)
    if not model_path.exists():
        # Aucun modèle entraîné : on ne renvoie PAS le prix Black-Scholes
        # (ce serait présenter la référence comme une prédiction ML).
        return {
            "price": None,
            "model_name": "Modèle non entraîné",
            "elapsed_s": time.perf_counter() - t0,
            "is_trained": False,
            "mae": None,
            "rmse": None,
            "r2": None,
        }

    bundle = load_model(model_path)
    price = predict_batch(bundle, [(S, K, r, sigma, T, option_type)])[0]
    metrics = bundle.get("metrics", {})
    return {
        "price": price,
        "model_name": bundle.get("model_name", "Modèle ML"),
        "elapsed_s": time.perf_counter() - t0,
        "is_trained": True,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
    }
