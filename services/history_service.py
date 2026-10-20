"""Sauvegarde et lecture des scénarios."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from config import HISTORY_PATH


def load_history(path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_scenario(payload: dict, path: Path = HISTORY_PATH) -> dict:
    history = load_history(path)
    item = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    history.append(item)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return item


def clear_history(path: Path = HISTORY_PATH) -> None:
    path.write_text("[]", encoding="utf-8")
