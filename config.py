"""Configuration générale de l'application Option Pricing Lab."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_PARAMS = {
    "option_type": "call",
    "S": 100.0,
    "K": 100.0,
    "r": 0.05,
    "sigma": 0.20,
    "T": 1.0,
    "M": 10_000,
    "N": 252,
    "seed": 42,
}

RF_MODEL_PATH = MODEL_DIR / "option_pricing_rf.pkl"
MLP_MODEL_PATH = MODEL_DIR / "option_pricing_mlp.pkl"
DATASET_PATH = DATA_DIR / "ml_dataset.csv"
HISTORY_PATH = DATA_DIR / "scenarios_history.json"
