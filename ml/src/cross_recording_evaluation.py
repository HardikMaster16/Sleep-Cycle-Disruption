import sys

sys.path.append("ML/src")

import pandas as pd

from dataset_loader import load_all_recordings

from feature_engineering import (
    FEATURE_COLUMNS,
    MOVEMENT_FEATURE_COLUMNS
)

from eif_model import (
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

RECORDINGS = [
    "Bidslab01_1",
    "Bidslab01_2",
    "Bidslab01_3",
    "Bidslab02_1",
    "Bidslab02_2"
]

N_TREES = 200
SAMPLE_SIZE = 256
RANDOM_SEED = 42
THRESHOLD_PERCENTILE = 95


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING ALL RECORDINGS")
print("=" * 70)

features = load_all_recordings(RAW_DIR)

print(
    "\nTotal epochs:",
    len(features)
)

print(
    "Total recordings:",
    features["recording_id"].nunique()
)


# ============================================================
# EXPERIMENT FUNCTION
# ============================================================

def run_loro_experiment(
    all_features,
    feature_columns,
    experiment_name
):

    print("\n")
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    results = []

    for test_recording in RECORDINGS:

        print("\n" + "-" * 70)

        print(
            "TEST RECORDING:",
            test_recording
        )

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        train = all_features[
            all_features["recording_id"]
            != test_recording
        ].copy()

        test = all_features[
            all_features["recording_id"]
            == test_recording
        ].copy()

        print(
            "Training epochs:",
            len(train)
        )

        print(
            "Test epochs:",
            len(test)
        )

        print(
            "Features:",
            len(feature_columns)
        )


        # ----------------------------------------------------
        # FIT SCALER ONLY ON TRAINING DATA
        # ----------------------------------------------------

        X_train_scaled, scaler = fit_scaler(
            train,
            feature_columns
        )

        X_test_scaled = transform_features(
            test,
            feature_columns,
            scaler
        )


        # ----------------------------------------------------
        # TRAIN EIF
        # ----------------------------------------------------

        forest = train_eif(
            X_train_scaled,
            ntrees=N_TREES,
            sample_size=SAMPLE_SIZE,
            random_seed=RANDOM_SEED
        )


        # ----------------------------------------------------
        # CALCULATE TRAINING SCORES
        # ----------------------------------------------------

        train_scores = calculate_anomaly_scores(
            forest,
            X_train_scaled
        )

        train_results = add_anomaly_scores(
            train,
            train_scores
        )


        # ----------------------------------------------------
        # CALCULATE TEST SCORES
        # ----------------------------------------------------

        test_scores = calculate_anomaly_scores(
            forest,
            X_test_scaled
        )

        test_results = add_anomaly_scores(
            test,
            test_scores
        )


        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We do NOT use the test recording to determine
        # the threshold.
        #
        # For this diagnostic LORO experiment, the threshold
        # is derived from the training score distribution.
        #
        # A stricter nested validation design can be added
        # later if required.
        # ----------------------------------------------------

        threshold = train_results[
            "anomaly_score"
        ].quantile(
            THRESHOLD_PERCENTILE / 100
        )

        test_results = apply_threshold(
            test_results,
            threshold
        )


        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        metrics = evaluate_wake_detection(
            test_results
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        result = {
            "experiment": experiment_name,
            "test_recording": test_recording,
            "train_epochs": len(train),
            "test_epochs": len(test),
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

        results.append(result)


        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

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


    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    return results_df


# ============================================================
# BASELINE EXPERIMENT
# ============================================================

baseline_results = run_loro_experiment(
    features,
    FEATURE_COLUMNS,
    "BASELINE — 19 FEATURES"
)


# ============================================================
# MOVEMENT-FOCUSED EXPERIMENT
# ============================================================

movement_results = run_loro_experiment(
    features,
    MOVEMENT_FEATURE_COLUMNS,
    "MOVEMENT-FOCUSED — 14 FEATURES"
)


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = pd.concat(
    [
        baseline_results,
        movement_results
    ],
    ignore_index=True
)


# ============================================================
# PER-RECORDING RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PER-RECORDING RESULTS")
print("=" * 70)

print(
    all_results[
        [
            "experiment",
            "test_recording",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "anomalies"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("LORO SUMMARY")
print("=" * 70)

summary = (
    all_results
    .groupby("experiment")
    [
        [
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1"
        ]
    ]
    .agg(
        ["mean", "std"]
    )
)

print(
    summary.to_string()
)


# ============================================================
# SIMPLE MEAN COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("MEAN PERFORMANCE")
print("=" * 70)

for experiment in all_results[
    "experiment"
].unique():

    subset = all_results[
        all_results["experiment"]
        == experiment
    ]

    print(
        f"\n{experiment}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{subset['roc_auc'].mean():.4f}"
    )

    print(
        f"Mean PR-AUC: "
        f"{subset['pr_auc'].mean():.4f}"
    )

    print(
        f"Mean Precision: "
        f"{subset['precision'].mean():.4f}"
    )

    print(
        f"Mean Recall: "
        f"{subset['recall'].mean():.4f}"
    )

    print(
        f"Mean F1: "
        f"{subset['f1'].mean():.4f}"
    )


print("\n")
print("=" * 70)
print("LORO EXPERIMENT COMPLETED")
print("=" * 70)