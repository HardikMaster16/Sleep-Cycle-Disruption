from pathlib import Path
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------
# Locate project root
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

ML_SRC_DIR = PROJECT_DIR / "ML" / "src"

if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))


# --------------------------------------------------
# Import ML inference
# --------------------------------------------------

from inference import predict_anomaly


# --------------------------------------------------
# Final model feature order
# --------------------------------------------------

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
    "z_std",
]


# --------------------------------------------------
# Existing feature prediction
# --------------------------------------------------

def predict(features: dict) -> dict:
    """
    Send already-engineered feature data
    to the ML inference layer.
    """

    result = predict_anomaly(features)

    if not result:
        raise RuntimeError(
            "ML inference returned no prediction."
        )

    return result[0]


# --------------------------------------------------
# Prepare raw accelerometer data
# --------------------------------------------------

def prepare_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw accelerometer CSV data.
    Expected columns:
        Timestamp, x, y, z
    """

    required_columns = {
        "Timestamp",
        "x",
        "y",
        "z",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df[
        ["Timestamp", "x", "y", "z"]
    ].copy()

    # Convert numeric columns
    df["Timestamp"] = pd.to_numeric(
        df["Timestamp"],
        errors="coerce",
    )

    for column in ["x", "y", "z"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    if df.empty:
        raise ValueError(
            "No valid accelerometer data found."
        )

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    # Start each uploaded recording at epoch 0
    start_timestamp = df["Timestamp"].iloc[0]

    df["epoch"] = np.floor(
        (df["Timestamp"] - start_timestamp)
        / 30
    ).astype(int)

    return df


# --------------------------------------------------
# Extract the same movement features
# --------------------------------------------------

def extract_epoch_features(
    group: pd.DataFrame,
) -> dict:

    x = group["x"]
    y = group["y"]
    z = group["z"]

    magnitude = np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )

    dynamic = (
        magnitude - 1.0
    ).abs()

    movement_change = (
        magnitude.diff().abs()
    )

    features = {
        "magnitude_range": (
            magnitude.max()
            - magnitude.min()
        ),

        "magnitude_max": (
            magnitude.max()
        ),

        "magnitude_min": (
            magnitude.min()
        ),

        "dynamic_mean": (
            dynamic.mean()
        ),

        "dynamic_std": (
            dynamic.std()
        ),

        "dynamic_max": (
            dynamic.max()
        ),

        "movement_change_mean": (
            movement_change.mean()
        ),

        "movement_change_std": (
            movement_change.std()
        ),

        "movement_change_max": (
            movement_change.max()
        ),

        "movement_energy": (
            np.mean(dynamic ** 2)
        ),

        "strong_movement_ratio": (
            np.mean(dynamic > 0.05)
        ),

        "x_std": x.std(),
        "y_std": y.std(),
        "z_std": z.std(),
    }

    return features


# --------------------------------------------------
# CSV prediction
# --------------------------------------------------

def predict_csv(file_object) -> list:
    """
    Process an uploaded accelerometer CSV,
    extract 14 movement features for each
    30-second epoch, and run EIF inference.
    """

    df = pd.read_csv(file_object)

    df = prepare_csv_data(df)

    predictions = []

    for epoch, group in df.groupby("epoch"):

        # Ignore extremely small epochs
        if len(group) < 100:
            continue

        features = extract_epoch_features(
            group
        )

        # Keep exact model feature order
        feature_data = {
            column: features[column]
            for column in MOVEMENT_FEATURE_COLUMNS
        }

        result = predict_anomaly(
            feature_data
        )

        if not result:
            continue

        prediction = result[0]

        epoch_timestamp = pd.to_datetime(
            group["Timestamp"].iloc[0],
            unit="s",
            utc=True,
)

        predictions.append({
            "epoch": int(epoch),
            "timestamp": epoch_timestamp.isoformat(),
            "sample_count": int(len(group)),
            "anomaly_score": prediction["anomaly_score"],
            "threshold": prediction["threshold"],
            "is_anomaly": prediction["is_anomaly"],
            "prediction": prediction["prediction"],
})

    if not predictions:
        raise ValueError(
            "No valid 30-second epochs "
            "were found in the uploaded CSV."
        )

    return predictions