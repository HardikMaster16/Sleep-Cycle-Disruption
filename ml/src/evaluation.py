import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_wake_detection(results):
    """
    Evaluate anomaly detection against Wake vs Sleep labels.

    Wake = 1
    Non-Wake = 0
    """

    y_true = (
        results["sleep_stage"] == 0
    ).astype(int)

    anomaly_scores = results[
        "anomaly_score"
    ]

    y_pred = results[
        "is_anomaly"
    ].astype(int)

    roc_auc = roc_auc_score(
        y_true,
        anomaly_scores
    )

    pr_auc = average_precision_score(
        y_true,
        anomaly_scores
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


def analyze_sleep_anomalies(results):
    """
    Analyze anomalies occurring during non-Wake sleep stages.

    These are interpreted as candidate movement/restlessness
    events, not clinically validated restlessness.
    """

    sleep_anomalies = results[
        (results["is_anomaly"]) &
        (results["sleep_stage"] != 0)
    ].copy()

    summary = (
        sleep_anomalies
        .groupby("sleep_stage")
        .agg(
            anomalous_epochs=("epoch", "count"),
            mean_movement_energy=(
                "movement_energy",
                "mean"
            ),
            max_movement_energy=(
                "movement_energy",
                "max"
            ),
            mean_anomaly_score=(
                "anomaly_score",
                "mean"
            ),
            max_anomaly_score=(
                "anomaly_score",
                "max"
            )
        )
    )

    return sleep_anomalies, summary


def detect_anomaly_events(results):
    """
    Group consecutive anomalous epochs into events.
    """

    results = (
        results
        .sort_values("epoch")
        .reset_index(drop=True)
        .copy()
    )

    results["anomaly_group"] = (
        results["is_anomaly"]
        .ne(results["is_anomaly"].shift())
        .cumsum()
    )

    events = (
        results[
            results["is_anomaly"]
        ]
        .groupby("anomaly_group")
        .agg(
            start_epoch=("epoch", "min"),
            end_epoch=("epoch", "max"),
            duration_epochs=("epoch", "count"),
            max_anomaly_score=(
                "anomaly_score",
                "max"
            ),
            max_movement_energy=(
                "movement_energy",
                "max"
            )
        )
        .reset_index(drop=True)
    )

    events["duration_seconds"] = (
        events["duration_epochs"] * 30
    )

    return events