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


def predict_scenario(
    machine_data: dict
) -> dict:
    """
    Run a lightweight prediction for scenario analysis.

    Unlike the normal prediction service, this does not
    calculate SHAP explanations because scenario analysis
    may evaluate many machine configurations.
    """

    # --------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------

    machine_df = pd.DataFrame(
        [machine_data]
    )


    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    features = create_engineered_features(
        machine_df
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


def run_scenario(
    machine_data: dict,
    parameter: str,
    values: list[float]
) -> pd.DataFrame:
    """
    Run a what-if analysis by changing one machine
    parameter while keeping all other parameters constant.

    Parameters
    ----------
    machine_data:
        Base machine parameters.

    parameter:
        Parameter to vary.

    values:
        Values to test for that parameter.

    Returns
    -------
    pd.DataFrame
        Prediction results for every scenario.
    """

    results = []

    for value in values:

        scenario = machine_data.copy()

        scenario[parameter] = value

        prediction = predict_scenario(
            scenario
        )

        results.append({
            "Parameter": value,
            "Failure Probability": (
                prediction["failure_probability"]
            ),
            "Anomaly Risk": (
                prediction["anomaly_risk"]
            ),
            "Hybrid Risk": (
                prediction["hybrid_risk"]
            ),
            "Risk State": (
                prediction["risk_state"]
            ),
            "Recommended Action": (
                prediction["recommended_action"]
            )
        })

    return pd.DataFrame(results)