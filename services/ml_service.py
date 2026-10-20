"""Services Machine Learning."""
import time

from config import RF_MODEL_PATH, MLP_MODEL_PATH, DATASET_PATH
from ml.predict_model import predict_option_price, predict_batch, load_model
from ml.train_model import train_model


def _path(model_type: str):
    return RF_MODEL_PATH if model_type != "mlp" else MLP_MODEL_PATH


def ensure_model(model_type: str = "rf") -> None:
    """Entraîne le modèle s'il n'existe pas encore, pour que l'interface
    affiche toujours une vraie prédiction."""
    if not _path(model_type).exists():
        train_ml_model(model_type)


def compute_ml_prediction(params: dict, model_type: str = "rf") -> dict:
    ensure_model(model_type)
    return predict_option_price(
        params["S"], params["K"], params["r"], params["sigma"], params["T"],
        params["option_type"], model_path=_path(model_type)
    )


def predict_many(rows: list, model_type: str = "rf") -> dict:
    """rows : liste de dicts {S, K, r, sigma, T, option_type}."""
    ensure_model(model_type)
    bundle = load_model(_path(model_type))
    t0 = time.perf_counter()
    prices = predict_batch(bundle, [
        (float(r["S"]), float(r["K"]), float(r["r"]), float(r["sigma"]), float(r["T"]),
         str(r.get("option_type", "call")).lower())
        for r in rows
    ])
    return {
        "prices": prices,
        "model_name": bundle.get("model_name"),
        "elapsed_s": time.perf_counter() - t0,
        "metrics": bundle.get("metrics", {}),
    }


def model_status(model_type: str = "rf") -> dict:
    path = _path(model_type)
    if not path.exists():
        return {"is_trained": False}
    bundle = load_model(path)
    return {"is_trained": True, "model_name": bundle.get("model_name"),
            "metrics": bundle.get("metrics", {}), "target": bundle.get("target", "price")}


def train_ml_model(model_type: str = "rf", n_samples: int = 3000, **options) -> dict:
    res = train_model(model_type=model_type, n_samples=n_samples,
                      dataset_path=DATASET_PATH, model_path=_path(model_type), **options)
    return res.__dict__
