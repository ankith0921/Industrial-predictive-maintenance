import joblib
import pandas as pd
import shap

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
# SHAP Explainability
# --------------------------------------------------

def calculate_shap_explanation(
    df: pd.DataFrame
) -> list:
    """
    Calculate SHAP feature contributions for
    the current machine prediction.

    The saved XGBoost model is a sklearn Pipeline
    containing:

        ColumnTransformer
            -> StandardScaler
            -> OneHotEncoder
            -> XGBClassifier

    SHAP is calculated directly on the transformed
    features that reach the XGBoost classifier.
    """

    # --------------------------------------------------
    # Extract pipeline components
    # --------------------------------------------------

    preprocessor = xgb_model.named_steps[
        "preprocessor"
    ]

    classifier = xgb_model.named_steps[
        "classifier"
    ]

    # --------------------------------------------------
    # Transform input using the SAME preprocessing
    # used during model training
    # --------------------------------------------------

    transformed = preprocessor.transform(
        df
    )

    # --------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # --------------------------------------------------
    # Create SHAP TreeExplainer
    # --------------------------------------------------

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        transformed
    )

    # --------------------------------------------------
    # Get SHAP values for first prediction
    # --------------------------------------------------

    values = shap_values[0]

    # --------------------------------------------------
    # Convert SHAP output into readable format
    # --------------------------------------------------

    explanations = []

    for name, value in zip(
        feature_names,
        values
    ):

        # Remove preprocessing prefixes
        clean_name = name

        if clean_name.startswith("num__"):
            clean_name = clean_name.replace(
                "num__",
                "",
                1
            )

        elif clean_name.startswith("cat__"):
            clean_name = clean_name.replace(
                "cat__",
                "",
                1
            )

        explanations.append(
            {
                "feature": clean_name,
                "impact": float(value)
            }
        )

    # --------------------------------------------------
    # Combine one-hot encoded Type features
    # --------------------------------------------------

    type_features = [
        item
        for item in explanations
        if item["feature"].startswith("Type_")
    ]

    if type_features:

        type_impact = sum(
            item["impact"]
            for item in type_features
        )

        explanations = [
            item
            for item in explanations
            if not item["feature"].startswith(
                "Type_"
            )
        ]

        explanations.append(
            {
                "feature": "Product Type",
                "impact": float(type_impact)
            }
        )

    # --------------------------------------------------
    # Sort by absolute impact
    # --------------------------------------------------

    explanations.sort(
        key=lambda item: abs(item["impact"]),
        reverse=True
    )

    return explanations


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