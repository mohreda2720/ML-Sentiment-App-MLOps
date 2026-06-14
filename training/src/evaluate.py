"""Evaluation utilities for sentiment classification."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, f1, precision, recall.
    """
    labels = ["positive", "negative", "neutral"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
    }


def get_confusion_matrix(
    y_true: list[str], y_pred: list[str]
) -> list[list[int]]:
    """Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix as nested list.
    """
    labels = ["positive", "negative", "neutral"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm.tolist()


def get_classification_report(y_true: list[str], y_pred: list[str]) -> str:
    """Generate a text classification report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Formatted classification report string.
    """
    labels = ["positive", "negative", "neutral"]
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)
