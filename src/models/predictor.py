import joblib
import pandas as pd

from src.utils.config import (
    XGBOOST_MODEL_PATH,
    load_config
)


# --------------------------------------------------
# Load trained model and configuration
# --------------------------------------------------

xgb_model = joblib.load(
    XGBOOST_MODEL_PATH
)

config = load_config()


# --------------------------------------------------
# XGBoost Failure Prediction
# --------------------------------------------------

def predict_failure(
    df: pd.DataFrame
) -> dict:
    """
    Predict machine failure probability
    using the trained XGBoost pipeline.
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


# --------------------------------------------------
# Hybrid Risk Engine
# --------------------------------------------------

def calculate_hybrid_risk(
    failure_probability: float,
    anomaly_risk: float
) -> float:
    """
    Combine XGBoost failure probability and
    Isolation Forest anomaly risk.
    """

    xgb_weight = float(
        config["hybrid_xgb_weight"]
    )

    anomaly_weight = float(
        config["hybrid_anomaly_weight"]
    )

    hybrid_risk = (
        xgb_weight * failure_probability
        +
        anomaly_weight * anomaly_risk
    )

    return float(hybrid_risk)


# --------------------------------------------------
# Risk State
# --------------------------------------------------

def determine_risk_state(
    failure_probability: float,
    anomaly_risk: float,
    hybrid_risk: float
) -> str:
    """
    Determine the machine's operational risk state.
    """

    hybrid_threshold = float(
        config["hybrid_threshold"]
    )

    xgb_threshold = float(
        config["xgb_threshold"]
    )

    anomaly_threshold = float(
        config["anomaly_threshold"]
    )

    if hybrid_risk >= hybrid_threshold:

        if (
            failure_probability >= xgb_threshold
            and anomaly_risk >= anomaly_threshold
        ):
            return "CRITICAL"

        elif failure_probability >= xgb_threshold:
            return "PREDICTED FAILURE"

        else:
            return "ANOMALOUS"

    return "HEALTHY"


# --------------------------------------------------
# Maintenance Recommendation
# --------------------------------------------------

def maintenance_action(
    risk_state: str
) -> str:
    """
    Generate a recommended maintenance action.
    """

    if risk_state == "CRITICAL":
        return "Immediate inspection"

    elif risk_state == "PREDICTED FAILURE":
        return "Schedule preventive maintenance"

    elif risk_state == "ANOMALOUS":
        return "Inspect operating conditions"

    return "Continue monitoring"