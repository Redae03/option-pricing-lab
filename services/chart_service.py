"""Préparation des données pour Chart.js."""
from __future__ import annotations
import numpy as np


def to_list(x, max_points: int | None = None):
    if x is None:
        return []
    arr = np.asarray(x).astype(float).ravel()
    if max_points and arr.size > max_points:
        idx = np.linspace(0, arr.size - 1, max_points).astype(int)
        arr = arr[idx]
    return arr.tolist()


def histogram(values, bins=30):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {"labels": [], "values": []}
    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {"labels": centers.round(4).tolist(), "values": counts.astype(int).tolist()}
