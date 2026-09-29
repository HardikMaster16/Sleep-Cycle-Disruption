from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from feature_engineering import MOVEMENT_FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "extended_isolation_forest.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"


# --------------------------------------------------
# Load artifacts
# --------------------------------------------------

forest = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

THRESHOLD = config["threshold"]
FEATURE_COLUMNS = config["feature_columns"]


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_anomaly(feature_data):
    """
    Predict whether a sleep epoch contains
    anomalous movement.
    """

    if isinstance(feature_data, dict):
        feature_data = pd.DataFrame([feature_data])

    elif isinstance(feature_data, pd.Series):
        feature_data = pd.DataFrame([feature_data])

    elif not isinstance(feature_data, pd.DataFrame):
        raise TypeError(
            "feature_data must be a dictionary, pandas Series, "
            "or pandas DataFrame."
        )

    # Validate features
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in feature_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = feature_data[FEATURE_COLUMNS].copy()

    # Check missing values
    if X.isnull().any().any():
        raise ValueError(
            "Input contains missing feature values."
        )

    # Scale using training scaler
    X_scaled = scaler.transform(X)

    # EIF anomaly score
    scores = forest.compute_paths(
        X_in=X_scaled
    )

    scores = np.asarray(scores)

    # Higher score = more anomalous
    predictions = scores >= THRESHOLD

    results = []

    for score, is_anomaly in zip(scores, predictions):

        results.append({
            "anomaly_score": float(score),
            "threshold": float(THRESHOLD),
            "is_anomaly": bool(is_anomaly),
            "prediction": (
                "Anomalous Movement"
                if is_anomaly
                else "Normal Movement"
            )
        })

    return results