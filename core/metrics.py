"""Métriques numériques communes."""
import numpy as np


def absolute_error(value: float, reference: float) -> float:
    return float(abs(value - reference))


def relative_error_pct(value: float, reference: float) -> float:
    return float(abs(value - reference) / abs(reference) * 100) if abs(reference) > 1e-12 else 0.0


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_custom(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
