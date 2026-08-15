from pathlib import Path
import json


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"


# Model files
XGBOOST_MODEL_PATH = (
    MODEL_DIR / "xgboost_model.joblib"
)

ISOLATION_FOREST_PATH = (
    MODEL_DIR / "isolation_forest.joblib"
)

ANOMALY_PREPROCESSOR_PATH = (
    MODEL_DIR / "anomaly_preprocessor.joblib"
)

CONFIG_PATH = (
    MODEL_DIR / "model_config.json"
)


def load_config():
    """Load model configuration."""

    with open(CONFIG_PATH, "r") as file:
        return json.load(file)