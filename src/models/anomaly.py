import joblib
import numpy as np
import pandas as pd

from src.utils.config import (
    ISOLATION_FOREST_PATH,
    ANOMALY_PREPROCESSOR_PATH,
    load_config
)


# Load anomaly detection components
isolation_forest = joblib.load(
    ISOLATION_FOREST_PATH
)

anomaly_preprocessor = joblib.load(
    ANOMALY_PREPROCESSOR_PATH
)

config = load_config()


def calculate_anomaly_risk(
    df: pd.DataFrame
) -> float:
    """
    Calculate normalized anomaly risk.
    """

    transformed = (
        anomaly_preprocessor.transform(df)
    )

    anomaly_score = (
        isolation_forest
        .decision_function(transformed)[0]
    )

    # Higher value = higher anomaly risk
    anomaly_risk = -anomaly_score

    normal_min = float(
        config["normal_anomaly_min"]
    )

    normal_max = float(
        config["normal_anomaly_max"]
    )

    normalized_risk = (
        (anomaly_risk - normal_min)
        /
        (normal_max - normal_min)
    )

    normalized_risk = float(
        np.clip(
            normalized_risk,
            0,
            1
        )
    )

    return normalized_risk