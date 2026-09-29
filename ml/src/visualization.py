import sys

sys.path.append("ML/src")

import matplotlib.pyplot as plt
import pandas as pd

from dataset_loader import load_all_recordings
from feature_engineering import MOVEMENT_FEATURE_COLUMNS

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

from evaluation import detect_anomaly_events


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

RESULTS_DIR = "ML/results"


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
    "Total epochs:",
    len(features)
)


# ============================================================
# SPLIT DATA
# ============================================================

train, validation, test = split_by_recording(
    features,
    TRAIN_RECORDINGS,
    VALIDATION_RECORDINGS,
    TEST_RECORDINGS
)

print(
    "\nTrain epochs:",
    len(train)
)

print(
    "Validation epochs:",
    len(validation)
)

print(
    "Test epochs:",
    len(test)
)


# ============================================================
# FIT SCALER
# ============================================================

print("\n" + "=" * 60)
print("FITTING SCALER")
print("=" * 60)

X_train_scaled, scaler = fit_scaler(
    train,
    MOVEMENT_FEATURE_COLUMNS
)

X_validation_scaled = transform_features(
    validation,
    MOVEMENT_FEATURE_COLUMNS,
    scaler
)

X_test_scaled = transform_features(
    test,
    MOVEMENT_FEATURE_COLUMNS,
    scaler
)


# ============================================================
# TRAIN EIF
# ============================================================

print("\n" + "=" * 60)
print("TRAINING EIF")
print("=" * 60)

forest = train_eif(
    X_train_scaled,
    ntrees=N_TREES,
    sample_size=SAMPLE_SIZE,
    random_seed=RANDOM_SEED
)

print(
    "EIF training complete."
)


# ============================================================
# CALCULATE SCORES
# ============================================================

validation_scores = calculate_anomaly_scores(
    forest,
    X_validation_scaled
)

test_scores = calculate_anomaly_scores(
    forest,
    X_test_scaled
)

validation_results = add_anomaly_scores(
    validation,
    validation_scores
)

test_results = add_anomaly_scores(
    test,
    test_scores
)


# ============================================================
# SELECT VALIDATION THRESHOLD
# ============================================================

threshold = select_threshold(
    validation_results,
    percentile=THRESHOLD_PERCENTILE
)

test_results = apply_threshold(
    test_results,
    threshold
)

print(
    "\nValidation threshold:",
    threshold
)

print(
    "Test anomalies:",
    test_results["is_anomaly"].sum()
)


# ============================================================
# CREATE TIME AXIS
# ============================================================

test_results = test_results.sort_values(
    "epoch"
).reset_index(
    drop=True
)

test_results["time_minutes"] = (
    test_results["epoch"] * 30 / 60
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

import os

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# VISUALIZATION 1
# ANOMALY SCORE OVER TIME
# ============================================================

print("\nCreating anomaly-score plot...")

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    test_results["time_minutes"],
    test_results["anomaly_score"],
    linewidth=1
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.3f}"
)

anomalies = test_results[
    test_results["is_anomaly"]
]

plt.scatter(
    anomalies["time_minutes"],
    anomalies["anomaly_score"],
    s=25,
    label="Detected anomaly"
)

plt.xlabel(
    "Time (minutes)"
)

plt.ylabel(
    "EIF Anomaly Score"
)

plt.title(
    "Extended Isolation Forest Anomaly Scores"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/anomaly_score_over_time.png",
    dpi=300
)

plt.show()


# ============================================================
# VISUALIZATION 2
# MOVEMENT ENERGY OVER TIME
# ============================================================

print(
    "Creating movement-energy plot..."
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    test_results["time_minutes"],
    test_results["movement_energy"],
    linewidth=1
)

plt.xlabel(
    "Time (minutes)"
)

plt.ylabel(
    "Movement Energy"
)

plt.title(
    "Accelerometer Movement Energy During Sleep"
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/movement_energy_over_time.png",
    dpi=300
)

plt.show()


# ============================================================
# VISUALIZATION 3
# ANOMALY SCORE BY SLEEP STAGE
# ============================================================

print(
    "Creating sleep-stage plot..."
)

stage_names = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

stage_order = [
    0,
    1,
    2,
    3,
    4
]

stage_data = []

for stage in stage_order:

    values = test_results.loc[
        test_results["sleep_stage"] == stage,
        "anomaly_score"
    ]

    stage_data.append(
        values
    )

plt.figure(
    figsize=(10, 6)
)

plt.boxplot(
    stage_data,
    tick_labels=[
        stage_names[x]
        for x in stage_order
    ]
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.3f}"
)

plt.xlabel(
    "Sleep Stage"
)

plt.ylabel(
    "EIF Anomaly Score"
)

plt.title(
    "Anomaly Score Distribution Across Sleep Stages"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/anomaly_score_by_sleep_stage.png",
    dpi=300
)

plt.show()


# ============================================================
# ANOMALY EVENTS
# ============================================================

print(
    "\nCreating anomaly-event table..."
)

events = detect_anomaly_events(
    test_results
)

events.to_csv(
    f"{RESULTS_DIR}/test_anomaly_events.csv",
    index=False
)

print(
    "\nAnomaly events:"
)

print(
    events.to_string(
        index=False
    )
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

test_results.to_csv(
    f"{RESULTS_DIR}/test_results.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(
    f"Saved: "
    f"{RESULTS_DIR}/anomaly_score_over_time.png"
)

print(
    f"Saved: "
    f"{RESULTS_DIR}/movement_energy_over_time.png"
)

print(
    f"Saved: "
    f"{RESULTS_DIR}/anomaly_score_by_sleep_stage.png"
)

print(
    f"Saved: "
    f"{RESULTS_DIR}/test_anomaly_events.csv"
)

print(
    f"Saved: "
    f"{RESULTS_DIR}/test_results.csv"
)