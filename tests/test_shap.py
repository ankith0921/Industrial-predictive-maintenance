import pandas as pd
import shap

from src.features.engineering import (
    create_engineered_features
)

from src.utils.config import (
    XGBOOST_MODEL_PATH
)

import joblib


# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------

model = joblib.load(
    XGBOOST_MODEL_PATH
)

preprocessor = model.named_steps[
    "preprocessor"
]

classifier = model.named_steps[
    "classifier"
]


# --------------------------------------------------
# Test machine
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
# Apply trained preprocessing
# --------------------------------------------------

transformed = preprocessor.transform(
    sample_features
)


print("\nTransformed shape:")
print(transformed.shape)


# --------------------------------------------------
# Feature names after preprocessing
# --------------------------------------------------

feature_names = (
    preprocessor
    .get_feature_names_out()
)


print("\nTransformed feature names:")

for name in feature_names:
    print(name)


# --------------------------------------------------
# SHAP explainer
# --------------------------------------------------

explainer = shap.TreeExplainer(
    classifier
)


shap_values = explainer.shap_values(
    transformed
)


print("\nSHAP values shape:")
print(shap_values.shape)


# --------------------------------------------------
# Display feature contributions
# --------------------------------------------------

values = shap_values[0]


print("\nFeature contributions:")

for name, value in zip(
    feature_names,
    values
):

    print(
        f"{name}: {value:.6f}"
    )