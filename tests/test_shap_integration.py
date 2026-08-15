import pandas as pd

from src.features.engineering import (
    create_engineered_features
)

from src.models.predictor import (
    calculate_shap_explanation
)


def test_shap_explanation():

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

    sample_features = create_engineered_features(
        sample
    )

    explanation = calculate_shap_explanation(
        sample_features
    )

    print("\nSHAP Explanation:")

    for item in explanation:
        print(
            f"{item['feature']}: "
            f"{item['impact']:.6f}"
        )

    assert len(explanation) == 9

    assert all(
        "feature" in item
        and "impact" in item
        for item in explanation
    )