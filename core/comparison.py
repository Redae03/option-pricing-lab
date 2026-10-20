"""Comparaison globale des méthodes de pricing."""
from __future__ import annotations

from .metrics import absolute_error, relative_error_pct


def build_comparison_rows(bs, mc=None, em=None, ml=None) -> list[dict]:
    """Construit les lignes du tableau comparatif."""
    rows = [{
        "method": "Black-Scholes",
        "price": float(bs.price),
        "error": 0.0,
        "relative_error": 0.0,
        "time": float(getattr(bs, "elapsed_s", 0.0)),
        "status": "Référence",
    }]

    for name, obj, attr_time in [
        ("Monte-Carlo", mc, "elapsed_s"),
        ("Euler-Maruyama", em, "elapsed_s"),
        ("Machine Learning", ml, "elapsed_s"),
    ]:
        if obj is None:
            continue
        price = float(obj.price if hasattr(obj, "price") else obj["price"])
        rows.append({
            "method": name,
            "price": price,
            "error": absolute_error(price, bs.price),
            "relative_error": relative_error_pct(price, bs.price),
            "time": float(getattr(obj, attr_time, obj.get("elapsed_s", 0.0) if isinstance(obj, dict) else 0.0)),
            "status": "Estimation" if name != "Machine Learning" else "Prédiction",
        })
    return rows


def best_method(rows: list[dict]) -> dict:
    """Retourne la meilleure méthode hors Black-Scholes."""
    candidates = [r for r in rows if r["method"] != "Black-Scholes"]
    return min(candidates, key=lambda r: r["relative_error"]) if candidates else rows[0]
