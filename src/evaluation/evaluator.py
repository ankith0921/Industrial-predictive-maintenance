import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from src.features.engineering import (
    create_engineered_features
)

from src.models.anomaly import (
    isolation_forest,
    anomaly_preprocessor
)

from src.models.predictor import (
    calculate_hybrid_risk
)

from src.utils.config import (
    XGBOOST_MODEL_PATH,
    load_config
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/raw/ai4i2020.csv"


# --------------------------------------------------
# Load production model and configuration
# --------------------------------------------------

xgb_model = joblib.load(
    XGBOOST_MODEL_PATH
)

config = load_config()


# --------------------------------------------------
# Dataset preparation
# --------------------------------------------------

def load_evaluation_data():
    """
    Load the original AI4I dataset and recreate
    the same test split used during model development.
    """

    df = pd.read_csv(DATA_PATH)

    # Feature engineering
    df = create_engineered_features(df)

    # Target
    y = df["Machine failure"]

    # Features expected by the production model
    feature_columns = config["features"]

    X = df[feature_columns]

    # Same split used during model development
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_test, y_test


# --------------------------------------------------
# XGBoost evaluation
# --------------------------------------------------

def evaluate_xgboost():
    """
    Evaluate the saved production XGBoost model
    on the recreated untouched test set.
    """

    X_test, y_test = load_evaluation_data()

    # Failure probabilities
    probabilities = xgb_model.predict_proba(
        X_test
    )[:, 1]

    # Production threshold
    threshold = float(
        config["xgb_threshold"]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }


# --------------------------------------------------
# Anomaly risk calculation
# --------------------------------------------------

def calculate_anomaly_risks(
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Calculate normalized anomaly risk for every
    observation in the test set using the same
    Isolation Forest components as production.
    """

    transformed = anomaly_preprocessor.transform(
        X_test
    )

    anomaly_scores = (
        isolation_forest
        .decision_function(transformed)
    )

    # Higher value = higher anomaly risk
    anomaly_risk = -anomaly_scores

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

    normalized_risk = np.clip(
        normalized_risk,
        0,
        1
    )

    return normalized_risk.astype(float)


# --------------------------------------------------
# Hybrid evaluation
# --------------------------------------------------

def evaluate_hybrid():
    """
    Evaluate the complete hybrid predictive
    maintenance model.

    Hybrid risk combines:

        XGBoost failure probability
        +
        Isolation Forest anomaly risk
    """

    X_test, y_test = load_evaluation_data()

    # --------------------------------------------------
    # XGBoost probability
    # --------------------------------------------------

    failure_probabilities = (
        xgb_model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------
    # Isolation Forest anomaly risk
    # --------------------------------------------------

    anomaly_risks = calculate_anomaly_risks(
        X_test
    )

    # --------------------------------------------------
    # Hybrid risk
    # --------------------------------------------------

    hybrid_risks = np.array([
        calculate_hybrid_risk(
            float(failure_probability),
            float(anomaly_risk)
        )
        for failure_probability, anomaly_risk
        in zip(
            failure_probabilities,
            anomaly_risks
        )
    ])

    # --------------------------------------------------
    # Hybrid threshold
    # --------------------------------------------------

    hybrid_threshold = float(
        config["hybrid_threshold"]
    )

    hybrid_predictions = (
        hybrid_risks >= hybrid_threshold
    ).astype(int)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        hybrid_predictions
    )

    precision = precision_score(
        y_test,
        hybrid_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        hybrid_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        hybrid_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        hybrid_risks
    )

    pr_auc = average_precision_score(
        y_test,
        hybrid_risks
    )

    cm = confusion_matrix(
        y_test,
        hybrid_predictions
    )

    return {
        "threshold": hybrid_threshold,
        "xgb_weight": float(
            config["hybrid_xgb_weight"]
        ),
        "anomaly_weight": float(
            config["hybrid_anomaly_weight"]
        ),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }


# --------------------------------------------------
# Console evaluation
# --------------------------------------------------

if __name__ == "__main__":

    xgb_results = evaluate_xgboost()
    hybrid_results = evaluate_hybrid()

    print()
    print("=" * 60)
    print("PRODUCTION XGBOOST MODEL EVALUATION")
    print("=" * 60)

    print(
        f"Threshold : "
        f"{xgb_results['threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{xgb_results['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{xgb_results['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{xgb_results['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{xgb_results['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{xgb_results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{xgb_results['pr_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        xgb_results["confusion_matrix"]
    )

    print()
    print("=" * 60)
    print("HYBRID MODEL EVALUATION")
    print("=" * 60)

    print(
        f"XGBoost Weight : "
        f"{hybrid_results['xgb_weight']:.2f}"
    )

    print(
        f"Anomaly Weight : "
        f"{hybrid_results['anomaly_weight']:.2f}"
    )

    print(
        f"Threshold      : "
        f"{hybrid_results['threshold']:.2f}"
    )

    print(
        f"Accuracy       : "
        f"{hybrid_results['accuracy']:.4f}"
    )

    print(
        f"Precision      : "
        f"{hybrid_results['precision']:.4f}"
    )

    print(
        f"Recall         : "
        f"{hybrid_results['recall']:.4f}"
    )

    print(
        f"F1 Score       : "
        f"{hybrid_results['f1']:.4f}"
    )

    print(
        f"ROC-AUC        : "
        f"{hybrid_results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{hybrid_results['pr_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        hybrid_results["confusion_matrix"]
    )

    print("=" * 60)