from pathlib import Path
import joblib
import json

from dataset_loader import load_all_recordings
from eif_model import (
    split_by_recording,
    fit_scaler,
    transform_features,
    train_eif,
    calculate_anomaly_scores,
    select_threshold,
)
from feature_engineering import MOVEMENT_FEATURE_COLUMNS


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading recordings...")

features = load_all_recordings(RAW_DIR)

print(f"Total epochs: {len(features)}")


# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------

TRAIN_RECORDINGS = [
    "Bidslab01_1",
    "Bidslab01_2",
    "Bidslab02_1",
]

VALIDATION_RECORDINGS = [
    "Bidslab01_3",
]

TEST_RECORDINGS = [
    "Bidslab02_2",
]

train, validation, test = split_by_recording(
    features,
    TRAIN_RECORDINGS,
    VALIDATION_RECORDINGS,
    TEST_RECORDINGS,
)

print(f"Training epochs:   {len(train)}")
print(f"Validation epochs: {len(validation)}")
print(f"Test epochs:       {len(test)}")


# --------------------------------------------------
# Fit scaler ONLY on training data
# --------------------------------------------------

print("\nFitting scaler...")

X_train_scaled, scaler = fit_scaler(
    train,
    MOVEMENT_FEATURE_COLUMNS
)

X_validation_scaled = transform_features(
    validation,
    MOVEMENT_FEATURE_COLUMNS,
    scaler
)


# --------------------------------------------------
# Train Extended Isolation Forest
# --------------------------------------------------

print("Training Extended Isolation Forest...")

forest = train_eif(
    X_train_scaled,
    ntrees=200,
    sample_size=256,
    random_seed=42
)


# --------------------------------------------------
# Calculate validation anomaly scores
# --------------------------------------------------

validation_scores = calculate_anomaly_scores(
    forest,
    X_validation_scaled
)

validation_results = validation.copy()
validation_results["anomaly_score"] = validation_scores


# --------------------------------------------------
# Select threshold
# --------------------------------------------------

threshold = select_threshold(
    validation_results,
    percentile=95
)

print(f"\nSelected threshold: {threshold:.6f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

model_path = MODEL_DIR / "extended_isolation_forest.joblib"
scaler_path = MODEL_DIR / "scaler.joblib"
config_path = MODEL_DIR / "model_config.json"


joblib.dump(forest, model_path)
joblib.dump(scaler, scaler_path)


# --------------------------------------------------
# Save configuration
# --------------------------------------------------

config = {
    "feature_columns": MOVEMENT_FEATURE_COLUMNS,
    "threshold": float(threshold),
    "n_trees": 200,
    "sample_size": 256,
    "random_seed": 42,
    "threshold_percentile": 95,
    "description": (
        "Extended Isolation Forest for detecting anomalous "
        "sleep-related accelerometer movement patterns."
    )
}

with open(config_path, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nModel artifacts saved successfully:")

print(f"Model:  {model_path}")
print(f"Scaler: {scaler_path}")
print(f"Config: {config_path}")
print(f"Threshold: {threshold:.6f}")