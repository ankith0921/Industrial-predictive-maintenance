import sys
from pathlib import Path

import pandas as pd


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.features.engineering import (
    create_engineered_features
)

from src.models.predictor import (
    predict_failure
)

from src.models.anomaly import (
    calculate_anomaly_risk
)


# --------------------------------------------------
# Test data
# --------------------------------------------------

sample = pd.DataFrame([
    {
        "Type": "L",
        "Air temperature": 298.1,
        "Process temperature": 308.6,
        "Rotational speed": 1551,
        "Torque": 42.8,
        "Tool wear": 0
    }
])


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

sample_features = create_engineered_features(
    sample
)

print("\nEngineered features:")
print(sample_features)


# --------------------------------------------------
# XGBoost prediction
# --------------------------------------------------

failure_result = predict_failure(
    sample_features
)

print("\nXGBoost result:")
print(failure_result)


# --------------------------------------------------
# Anomaly detection
# --------------------------------------------------

anomaly_risk = calculate_anomaly_risk(
    sample_features
)

print("\nAnomaly risk:")
print(anomaly_risk)