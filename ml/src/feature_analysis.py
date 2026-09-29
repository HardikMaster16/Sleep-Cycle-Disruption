import sys

sys.path.append("ML/src")

import pandas as pd

from dataset_loader import load_all_recordings
from feature_engineering import FEATURE_COLUMNS


RAW_DIR = "ML/data/raw"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

features = load_all_recordings(RAW_DIR)

print(
    "\nFeature table:",
    features.shape
)


# ============================================================
# WAKE VS SLEEP
# ============================================================

features["wake"] = (
    features["sleep_stage"] == 0
).astype(int)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)

summary = []

for feature in FEATURE_COLUMNS:

    wake_values = features.loc[
        features["wake"] == 1,
        feature
    ]

    sleep_values = features.loc[
        features["wake"] == 0,
        feature
    ]

    wake_mean = wake_values.mean()
    sleep_mean = sleep_values.mean()

    wake_std = wake_values.std()
    sleep_std = sleep_values.std()

    summary.append({
        "feature": feature,
        "wake_mean": wake_mean,
        "sleep_mean": sleep_mean,
        "wake_std": wake_std,
        "sleep_std": sleep_std,
        "absolute_mean_difference":
            abs(wake_mean - sleep_mean)
    })


summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values(
    "absolute_mean_difference",
    ascending=False
)


print(
    summary_df.to_string(
        index=False
    )
)

MOVEMENT_FEATURE_COLUMNS = [
    "magnitude_range",
    "magnitude_max",
    "magnitude_min",

    "dynamic_mean",
    "dynamic_std",
    "dynamic_max",

    "movement_change_mean",
    "movement_change_std",
    "movement_change_max",

    "movement_energy",
    "strong_movement_ratio",

    "x_std",
    "y_std",
    "z_std"
]


# ============================================================
# STAGE-WISE FEATURE MEANS
# ============================================================

print("\n" + "=" * 60)
print("SLEEP-STAGE FEATURE MEANS")
print("=" * 60)

stage_means = (
    features
    .groupby("sleep_stage")[FEATURE_COLUMNS]
    .mean()
)

print(
    stage_means.to_string()
)


# ============================================================
# MOVEMENT ENERGY BY STAGE
# ============================================================

print("\n" + "=" * 60)
print("MOVEMENT ENERGY BY SLEEP STAGE")
print("=" * 60)

energy_summary = (
    features
    .groupby("sleep_stage")["movement_energy"]
    .agg([
        "count",
        "mean",
        "std",
        "min",
        "max"
    ])
)

print(
    energy_summary.to_string()
)


# ============================================================
# STRONG MOVEMENT RATIO BY STAGE
# ============================================================

print("\n" + "=" * 60)
print("STRONG MOVEMENT RATIO BY SLEEP STAGE")
print("=" * 60)

movement_summary = (
    features
    .groupby("sleep_stage")["strong_movement_ratio"]
    .agg([
        "count",
        "mean",
        "std",
        "min",
        "max"
    ])
)

print(
    movement_summary.to_string()
)


print("\nFeature analysis completed.")