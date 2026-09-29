import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "x_mean",
    "x_std",
    "y_mean",
    "y_std",
    "z_mean",
    "z_std",
    "magnitude_mean",
    "magnitude_std",
    "magnitude_min",
    "magnitude_max",
    "magnitude_range",
    "dynamic_mean",
    "dynamic_std",
    "dynamic_max",
    "movement_change_mean",
    "movement_change_std",
    "movement_change_max",
    "movement_energy",
    "strong_movement_ratio"
]


def prepare_movement_signals(motion_sleep):
    """Calculate accelerometer movement signals."""

    motion_sleep = motion_sleep.copy()

    motion_sleep["acc_magnitude"] = np.sqrt(
        motion_sleep["x"] ** 2
        + motion_sleep["y"] ** 2
        + motion_sleep["z"] ** 2
    )

    motion_sleep["dynamic_magnitude"] = (
        motion_sleep["acc_magnitude"] - 1.0
    ).abs()

    motion_sleep["movement_change"] = (
        motion_sleep
        .groupby("epoch")["acc_magnitude"]
        .diff()
        .abs()
    )

    return motion_sleep


def extract_epoch_features(group):
    """Extract movement features from one 30-second epoch."""

    mag = group["acc_magnitude"]
    dynamic = group["dynamic_magnitude"]
    movement_change = group["movement_change"]

    return pd.Series({
        "x_mean": group["x"].mean(),
        "x_std": group["x"].std(),

        "y_mean": group["y"].mean(),
        "y_std": group["y"].std(),

        "z_mean": group["z"].mean(),
        "z_std": group["z"].std(),

        "magnitude_mean": mag.mean(),
        "magnitude_std": mag.std(),
        "magnitude_min": mag.min(),
        "magnitude_max": mag.max(),
        "magnitude_range": (
            mag.max() - mag.min()
        ),

        "dynamic_mean": dynamic.mean(),
        "dynamic_std": dynamic.std(),
        "dynamic_max": dynamic.max(),

        "movement_change_mean": (
            movement_change.mean()
        ),
        "movement_change_std": (
            movement_change.std()
        ),
        "movement_change_max": (
            movement_change.max()
        ),

        "movement_energy": np.mean(
            dynamic ** 2
        ),

        "strong_movement_ratio": np.mean(
            dynamic > 0.05
        ),

        "sample_count": len(group)
    })


def build_feature_table(
    motion_sleep,
    expert_labels
):
    """Build epoch-level feature table."""

    motion_sleep = prepare_movement_signals(
        motion_sleep
    )

    features = (
        motion_sleep
        .groupby("epoch")
        .apply(extract_epoch_features)
        .reset_index()
    )

    features["sleep_stage"] = (
        features["epoch"]
        .astype(int)
        .map({
            i: int(expert_labels[i])
            for i in range(len(expert_labels))
        })
    )

    return features

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