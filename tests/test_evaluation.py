from src.evaluation.evaluator import evaluate_xgboost, load_evaluation_data
import pytest
from src.evaluation.evaluator import (
    evaluate_xgboost,
    evaluate_hybrid,
    load_evaluation_data
)

def test_evaluation_data_shape():
    X_test, y_test = load_evaluation_data()

    assert len(X_test) == 2000
    assert len(y_test) == 2000


def test_evaluation_data_contains_both_classes():
    _, y_test = load_evaluation_data()

    assert set(y_test.unique()) == {0, 1}


def test_xgboost_evaluation():
    results = evaluate_xgboost()

    assert results["threshold"] == pytest.approx(0.65)

    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1"] <= 1
    assert 0 <= results["roc_auc"] <= 1
    assert 0 <= results["pr_auc"] <= 1


def test_confusion_matrix():
    results = evaluate_xgboost()

    matrix = results["confusion_matrix"]

    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    assert len(matrix[1]) == 2

    assert all(
        isinstance(value, int)
        for row in matrix
        for value in row
    )


def test_expected_model_performance():
    results = evaluate_xgboost()

    # Sanity checks against the currently validated
    # production model.
    assert results["accuracy"] > 0.95
    assert results["roc_auc"] > 0.90
    assert results["pr_auc"] > 0.70

def test_hybrid_evaluation():
    results = evaluate_hybrid()

    assert results["threshold"] == pytest.approx(0.56)

    assert results["xgb_weight"] == pytest.approx(0.50)
    assert results["anomaly_weight"] == pytest.approx(0.50)

    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1"] <= 1
    assert 0 <= results["roc_auc"] <= 1
    assert 0 <= results["pr_auc"] <= 1


def test_hybrid_confusion_matrix():
    results = evaluate_hybrid()

    matrix = results["confusion_matrix"]

    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    assert len(matrix[1]) == 2

    assert all(
        isinstance(value, int)
        for row in matrix
        for value in row
    )


def test_hybrid_recall_not_lower_than_xgboost():
    xgb_results = evaluate_xgboost()
    hybrid_results = evaluate_hybrid()

    assert (
        hybrid_results["recall"]
        >= xgb_results["recall"]
    )