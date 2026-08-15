import pandas as pd

from src.models.service import predict_machine


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

        prediction = predict_machine(
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