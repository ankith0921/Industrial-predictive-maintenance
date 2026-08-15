import pandas as pd
import numpy as np


def create_engineered_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create engineered features used by the
    predictive maintenance models.
    """

    df = df.copy()

    # Temperature difference
    df["Temperature Difference"] = (
        df["Process temperature"]
        - df["Air temperature"]
    )

    # Mechanical power
    df["Mechanical Power"] = (
        2
        * np.pi
        * df["Torque"]
        * df["Rotational speed"]
        / 60
    )

    # Mechanical power in kW
    df["Mechanical Power kW"] = (
        df["Mechanical Power"] / 1000
    )

    # Temperature ratio
    df["Temperature Ratio"] = (
        df["Process temperature"]
        / df["Air temperature"]
    )

    return df