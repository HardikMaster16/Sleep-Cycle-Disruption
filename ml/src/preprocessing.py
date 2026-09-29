import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_motion_data(filepath):
    """Load and clean smartwatch accelerometer data."""

    motion = pd.read_csv(filepath)

    motion = motion[
        ["Timestamp", "x", "y", "z"]
    ].copy()

    motion["Timestamp"] = pd.to_numeric(
        motion["Timestamp"],
        errors="coerce"
    )

    for column in ["x", "y", "z"]:
        motion[column] = pd.to_numeric(
            motion[column],
            errors="coerce"
        )

    motion = (
        motion
        .dropna()
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    return motion


def load_expert_labels(filepath):
    """Load expert sleep-stage labels from labels.mat."""

    labels = loadmat(filepath)

    expert_labels = (
        labels["expert_label"]
        .flatten()
        .astype(int)
    )

    rec_start_string = str(
        labels["recStart"].flatten()[0]
    )

    rec_start = pd.Timestamp(
        rec_start_string,
        tz="US/Eastern"
    )

    return expert_labels, rec_start


def create_sleep_epochs(
    motion,
    expert_labels,
    rec_start
):
    """Assign accelerometer samples to 30-second epochs."""

    rec_start_unix = rec_start.timestamp()

    motion = motion.copy()

    motion["epoch"] = np.floor(
        (
            motion["Timestamp"]
            - rec_start_unix
        ) / 30
    ).astype(int)

    motion_sleep = motion[
        (motion["epoch"] >= 0)
        &
        (motion["epoch"] < len(expert_labels))
    ].copy()

    epoch_counts = (
        motion_sleep
        .groupby("epoch")
        .size()
    )

    complete_epochs = epoch_counts[
        epoch_counts > 1000
    ].index

    motion_sleep = motion_sleep[
        motion_sleep["epoch"].isin(
            complete_epochs
        )
    ].copy()

    return motion_sleep