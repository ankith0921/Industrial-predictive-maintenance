import joblib
import pandas as pd

from src.utils.config import (
    XGBOOST_MODEL_PATH,
    load_config
)


# Load trained XGBoost pipeline
xgb_model = joblib.load(
    XGBOOST_MODEL_PATH
)

# Load configuration
config = load_config()


def predict_failure(
    df: pd.DataFrame
) -> dict:
    """
    Predict machine failure probability.
    """

    probability = float(
        xgb_model.predict_proba(df)[0, 1]
    )

    threshold = float(
        config["xgb_threshold"]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "failure_probability": probability,
        "predicted_failure": prediction
    }