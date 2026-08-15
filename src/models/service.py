import pandas as pd

from src.features.engineering import (
    create_engineered_features
)

from src.models.predictor import (
    predict_failure,
    calculate_hybrid_risk,
    determine_risk_state,
    maintenance_action,
    calculate_shap_explanation
)

from src.models.anomaly import (
    calculate_anomaly_risk
)


# --------------------------------------------------
# Complete Machine Prediction
# --------------------------------------------------

def predict_machine(
    machine_data
) -> dict:
    """
    Run the complete predictive maintenance pipeline.

    Accepts either:

        - Python dictionary containing one machine
          observation
        - pandas DataFrame containing machine data

    Pipeline:

        Raw machine data
            ↓
        Feature engineering
            ↓
        XGBoost failure prediction
            ↓
        Isolation Forest anomaly detection
            ↓
        Hybrid risk calculation
            ↓
        Risk state
            ↓
        Maintenance recommendation
            ↓
        SHAP explanation
    """

    # --------------------------------------------------
    # Normalize input
    # --------------------------------------------------

    if isinstance(machine_data, dict):

        machine_data = pd.DataFrame(
            [machine_data]
        )

    elif not isinstance(
        machine_data,
        pd.DataFrame
    ):

        raise TypeError(
            "machine_data must be a dictionary "
            "or pandas DataFrame"
        )


    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    features = create_engineered_features(
        machine_data
    )


    # --------------------------------------------------
    # XGBoost Failure Prediction
    # --------------------------------------------------

    failure_result = predict_failure(
        features
    )

    failure_probability = (
        failure_result["failure_probability"]
    )


    # --------------------------------------------------
    # Anomaly Detection
    # --------------------------------------------------

    anomaly_risk = calculate_anomaly_risk(
        features
    )


    # --------------------------------------------------
    # Hybrid Risk
    # --------------------------------------------------

    hybrid_risk = calculate_hybrid_risk(
        failure_probability,
        anomaly_risk
    )


    # --------------------------------------------------
    # Risk State
    # --------------------------------------------------

    risk_state = determine_risk_state(
        failure_probability,
        anomaly_risk,
        hybrid_risk
    )


    # --------------------------------------------------
    # Maintenance Recommendation
    # --------------------------------------------------

    action = maintenance_action(
        risk_state
    )


    # --------------------------------------------------
    # SHAP Explainability
    # --------------------------------------------------

    shap_explanation = calculate_shap_explanation(
        features
    )


    # --------------------------------------------------
    # Complete Result
    # --------------------------------------------------

    return {
        "failure_probability": failure_probability,
        "predicted_failure": failure_result[
            "predicted_failure"
        ],
        "anomaly_risk": anomaly_risk,
        "hybrid_risk": hybrid_risk,
        "risk_state": risk_state,
        "recommended_action": action,
        "shap_explanation": shap_explanation
    }