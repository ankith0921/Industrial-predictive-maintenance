import pandas as pd

from src.features.engineering import (
    create_engineered_features
)

from src.models.predictor import (
    predict_failure,
    calculate_hybrid_risk,
    determine_risk_state,
    maintenance_action
)

from src.models.anomaly import (
    calculate_anomaly_risk
)


def predict_machine(
    machine_data: dict
) -> dict:
    """
    Run the complete predictive-maintenance
    pipeline for a single machine observation.
    """

    # ----------------------------------------------
    # 1. Convert input to DataFrame
    # ----------------------------------------------

    df = pd.DataFrame([machine_data])

    # ----------------------------------------------
    # 2. Feature engineering
    # ----------------------------------------------

    df = create_engineered_features(df)

    # ----------------------------------------------
    # 3. XGBoost failure prediction
    # ----------------------------------------------

    failure_result = predict_failure(df)

    failure_probability = (
        failure_result["failure_probability"]
    )

    # ----------------------------------------------
    # 4. Isolation Forest anomaly detection
    # ----------------------------------------------

    anomaly_risk = calculate_anomaly_risk(df)

    # ----------------------------------------------
    # 5. Hybrid risk
    # ----------------------------------------------

    hybrid_risk = calculate_hybrid_risk(
        failure_probability,
        anomaly_risk
    )

    # ----------------------------------------------
    # 6. Risk state
    # ----------------------------------------------

    risk_state = determine_risk_state(
        failure_probability,
        anomaly_risk,
        hybrid_risk
    )

    # ----------------------------------------------
    # 7. Maintenance recommendation
    # ----------------------------------------------

    action = maintenance_action(
        risk_state
    )

    # ----------------------------------------------
    # 8. Return final result
    # ----------------------------------------------

    return {
        "failure_probability": failure_probability,
        "predicted_failure": failure_result[
            "predicted_failure"
        ],
        "anomaly_risk": anomaly_risk,
        "hybrid_risk": hybrid_risk,
        "risk_state": risk_state,
        "recommended_action": action
    }