"""Entraînement des modèles ML de pricing."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .dataset_generator import generate_dataset, DatasetConfig, featurize_batch, train_test_split_df, save_dataset, FEATURE_NAMES
from .preprocessing import SimpleScaler


@dataclass
class TrainingResult:
    model_name: str
    mae: float
    rmse: float
    r2: float
    training_time: float
    n_train: int
    n_test: int
    model_path: str
    dataset_path: str
    feature_names: list
    target: str = "price"
    dataset_summary: dict = None
    test_sample: dict = None
    feature_importances: dict = None
    learning_curve: dict = None


def train_model(
    model_type: str = "rf",
    n_samples: int = 3000,
    dataset_path: str | Path = "data/ml_dataset.csv",
    model_path: str | Path = "models/option_pricing_rf.pkl",
    seed: int = 42,
    option_types: list | None = None,
    sigma_range: tuple = (0.05, 0.80),
    train_ratio: float = 0.80,
    complexity: int | None = None,
    learning_curve: bool = False,
) -> TrainingResult:
    """Génère un dataset Black-Scholes et entraîne RF ou MLP."""
    t0 = time.perf_counter()
    model_type = model_type.lower().strip()
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)

    cfg = DatasetConfig(n_samples=int(n_samples), seed=seed, sampling="lhs",
                        option_types=option_types or ["call", "put"], sigma_range=tuple(sigma_range))
    df = generate_dataset(cfg)
    save_dataset(df, str(dataset_path))

    train_df, test_df = train_test_split_df(df, train_ratio=float(train_ratio), seed=seed)
    X_train = featurize_batch(train_df)
    X_test = featurize_batch(test_df)
    y_train = train_df["price"].values
    y_test = test_df["price"].values
    K_train = train_df["K"].values
    K_test = test_df["K"].values

    scaler = None
    if model_type == "mlp":
        scaler = SimpleScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)
        width = int(complexity) if complexity else 64
        model = MLPRegressor(
            hidden_layer_sizes=(width, max(width // 2, 4)),
            activation="relu",
            solver="adam",
            max_iter=800,
            random_state=seed,
            early_stopping=True,
        )
        name = "MLP Regressor"
    else:
        X_train_fit, X_test_fit = X_train, X_test
        model = RandomForestRegressor(
            n_estimators=int(complexity) if complexity else 180,
            max_depth=16,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        name = "Random Forest"

    # Le Random Forest apprend le prix rapporté au strike (C/K) : la formule de
    # Black-Scholes est homogène de degré 1 en (S, K), ce qui réduit l'espace
    # à apprendre. Le MLP, qui travaille sur des features normalisées, apprend
    # directement le prix.
    target = "price_over_K" if model_type != "mlp" else "price"
    y_fit = y_train / K_train if target == "price_over_K" else y_train

    model.fit(X_train_fit, y_fit)
    pred = model.predict(X_test_fit)
    if target == "price_over_K":
        pred = pred * K_test
    pred = np.maximum(pred, 0.0)

    # Courbe d'apprentissage réelle : MAE train/test selon la taille de l'échantillon
    curve = None
    if learning_curve:
        curve = {"sizes": [], "mae_train": [], "mae_test": []}
        for frac in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            m_ = max(int(len(X_train_fit) * frac), 50)
            mdl = clone(model)
            if hasattr(mdl, "n_estimators"):
                mdl.set_params(n_estimators=min(mdl.n_estimators, 60))
            mdl.fit(X_train_fit[:m_], y_fit[:m_])
            scale_tr = K_train[:m_] if target == "price_over_K" else 1.0
            scale_te = K_test if target == "price_over_K" else 1.0
            p_tr = mdl.predict(X_train_fit[:m_]) * scale_tr
            p_te = mdl.predict(X_test_fit) * scale_te
            curve["sizes"].append(int(m_))
            curve["mae_train"].append(float(mean_absolute_error(y_train[:m_], p_tr)))
            curve["mae_test"].append(float(mean_absolute_error(y_test, p_te)))

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    elapsed = time.perf_counter() - t0

    bundle = {
        "model": model,
        "model_type": model_type,
        "model_name": name,
        "scaler": scaler,
        "target": target,
        "feature_names": FEATURE_NAMES,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    return TrainingResult(
        model_name=name,
        mae=mae,
        rmse=rmse,
        r2=r2,
        training_time=elapsed,
        n_train=len(train_df),
        n_test=len(test_df),
        model_path=str(model_path),
        dataset_path=str(dataset_path),
        feature_names=FEATURE_NAMES,
        target=target,
        dataset_summary={
            "n": int(len(df)),
            "calls": int((df["option_type"] == "call").sum()),
            "puts": int((df["option_type"] == "put").sum()),
            "mean_price": float(df["price"].mean()),
        },
        test_sample={
            "y_true": [float(v) for v in y_test],
            "y_pred": [float(v) for v in pred],
            "residuals": [float(v) for v in (pred - y_test)],
        },
        feature_importances=(
            {n: float(v) for n, v in zip(FEATURE_NAMES, model.feature_importances_)}
            if hasattr(model, "feature_importances_") else None
        ),
        learning_curve=curve,
    )
