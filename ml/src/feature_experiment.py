import sys

sys.path.append("ML/src")

import pandas as pd

from dataset_loader import load_all_recordings

from feature_engineering import (
    FEATURE_COLUMNS,
    MOVEMENT_FEATURE_COLUMNS
)

from eif_model import (
    split_by_recording,
    fit_scaler,
    transform_features,
    train_eif,
    calculate_anomaly_scores,
    add_anomaly_scores,
    select_threshold,
    apply_threshold
)

from evaluation import evaluate_wake_detection


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = "ML/data/raw"

TRAIN_RECORDINGS = [
    "Bidslab01_1",
    "Bidslab01_2",
    "Bidslab02_1"
]

VALIDATION_RECORDINGS = [
    "Bidslab01_3"
]

TEST_RECORDINGS = [
    "Bidslab02_2"
]

N_TREES = 200
SAMPLE_SIZE = 256
RANDOM_SEED = 42

THRESHOLD_PERCENTILE = 95


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

features = load_all_recordings(
    RAW_DIR
)

print(
    "\nTotal epochs:",
    len(features)
)


# ============================================================
# SPLIT BY RECORDING
# ============================================================

train, validation, test = split_by_recording(
    features,
    TRAIN_RECORDINGS,
    VALIDATION_RECORDINGS,
    TEST_RECORDINGS
)

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(
    "Train:",
    len(train)
)

print(
    "Validation:",
    len(validation)
)

print(
    "Test:",
    len(test)
)


# ============================================================
# EXPERIMENT FUNCTION
# ============================================================

def run_experiment(
    feature_columns,
    experiment_name
):

    print("\n")
    print("=" * 60)
    print(experiment_name)
    print("=" * 60)

    print(
        "Number of features:",
        len(feature_columns)
    )

    print(
        "Features:"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    X_train_scaled, scaler = fit_scaler(
        train,
        feature_columns
    )

    X_validation_scaled = transform_features(
        validation,
        feature_columns,
        scaler
    )

    X_test_scaled = transform_features(
        test,
        feature_columns,
        scaler
    )


    # --------------------------------------------------------
    # TRAIN EIF
    # --------------------------------------------------------

    forest = train_eif(
        X_train_scaled,
        ntrees=N_TREES,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )


    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    validation_scores = (
        calculate_anomaly_scores(
            forest,
            X_validation_scaled
        )
    )

    test_scores = (
        calculate_anomaly_scores(
            forest,
            X_test_scaled
        )
    )


    validation_results = (
        add_anomaly_scores(
            validation,
            validation_scores
        )
    )

    test_results = (
        add_anomaly_scores(
            test,
            test_scores
        )
    )


    # --------------------------------------------------------
    # THRESHOLD FROM VALIDATION
    # --------------------------------------------------------

    threshold = select_threshold(
        validation_results,
        percentile=THRESHOLD_PERCENTILE
    )

    test_results = apply_threshold(
        test_results,
        threshold
    )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    metrics = evaluate_wake_detection(
        test_results
    )


    print("\nResults:")

    print(
        f"Threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score: "
        f"{metrics['f1']:.4f}"
    )

    print(
        "Anomalies:",
        test_results["is_anomaly"].sum()
    )

    return {
        "experiment": experiment_name,
        "features": len(feature_columns),
        "threshold": threshold,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "anomalies": int(
            test_results["is_anomaly"].sum()
        )
    }


# ============================================================
# RUN BASELINE
# ============================================================

baseline = run_experiment(
    FEATURE_COLUMNS,
    "BASELINE — 19 FEATURES"
)


# ============================================================
# RUN MOVEMENT EXPERIMENT
# ============================================================

movement = run_experiment(
    MOVEMENT_FEATURE_COLUMNS,
    "MOVEMENT-FOCUSED — 14 FEATURES"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPARISON")
print("=" * 60)

comparison = pd.DataFrame([
    baseline,
    movement
])

print(
    comparison.to_string(
        index=False
    )
)


print(
    "\nFeature experiment completed."
)