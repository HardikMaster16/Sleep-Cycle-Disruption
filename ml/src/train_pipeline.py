import sys

sys.path.append("ML/src")

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

from evaluation import (
    evaluate_wake_detection,
    analyze_sleep_anomalies,
    detect_anomaly_events
)


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

features = load_all_recordings(RAW_DIR)

print()
print("Total epochs:", len(features))


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

print("Total recordings:", features["recording_id"].nunique())

print("\nRecordings:")
for recording in features["recording_id"].unique():
    count = (
        features["recording_id"] == recording
    ).sum()

    print(
        f"  {recording}: {count} epochs"
    )

print("\nMissing values:")

missing_values = features.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ]
)

if features[FEATURE_COLUMNS].isnull().any().any():
    raise ValueError(
        "Feature table contains missing values."
    )

print("\nData validation passed.")


# ============================================================
# RECORDING-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("RECORDING-LEVEL SPLIT")
print("=" * 60)

train, validation, test = split_by_recording(
    features,
    TRAIN_RECORDINGS,
    VALIDATION_RECORDINGS,
    TEST_RECORDINGS
)

print(
    "Training epochs:",
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

print("\nTraining recordings:")
print(
    train["recording_id"].unique()
)

print("\nValidation recordings:")
print(
    validation["recording_id"].unique()
)

print("\nTest recordings:")
print(
    test["recording_id"].unique()
)


# ============================================================
# VERIFY SPLIT
# ============================================================

print("\n" + "=" * 60)
print("VERIFYING RECORDING SEPARATION")
print("=" * 60)

train_ids = set(
    train["recording_id"].unique()
)

validation_ids = set(
    validation["recording_id"].unique()
)

test_ids = set(
    test["recording_id"].unique()
)

if train_ids & validation_ids:
    raise ValueError(
        "Data leakage: training and validation recordings overlap."
    )

if train_ids & test_ids:
    raise ValueError(
        "Data leakage: training and test recordings overlap."
    )

if validation_ids & test_ids:
    raise ValueError(
        "Data leakage: validation and test recordings overlap."
    )

print(
    "Training/validation/test recordings are completely separate."
)

print("Split verification: PASSED")


# ============================================================
# FIT SCALER ONLY ON TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("FITTING SCALER ON TRAINING DATA")
print("=" * 60)

X_train_scaled, scaler = fit_scaler(
    train,
    FEATURE_COLUMNS
)

X_validation_scaled = transform_features(
    validation,
    FEATURE_COLUMNS,
    scaler
)

X_test_scaled = transform_features(
    test,
    FEATURE_COLUMNS,
    scaler
)

print(
    "Number of features:",
    len(FEATURE_COLUMNS)
)

print(
    "Training shape:",
    X_train_scaled.shape
)

print(
    "Validation shape:",
    X_validation_scaled.shape
)

print(
    "Test shape:",
    X_test_scaled.shape
)


# ============================================================
# TRAIN EXTENDED ISOLATION FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING EXTENDED ISOLATION FOREST")
print("=" * 60)

print(
    "Trees:",
    N_TREES
)

print(
    "Sample size:",
    SAMPLE_SIZE
)

print(
    "Random seed:",
    RANDOM_SEED
)

print(
    "Extension level:",
    len(FEATURE_COLUMNS) - 1
)

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
# CALCULATE VALIDATION AND TEST SCORES
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING ANOMALY SCORES")
print("=" * 60)

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

print(
    "Validation score range:"
)

print(
    f"  Min: {validation_results['anomaly_score'].min():.6f}"
)

print(
    f"  Max: {validation_results['anomaly_score'].max():.6f}"
)

print(
    f"  Mean: {validation_results['anomaly_score'].mean():.6f}"
)

print(
    "\nTest score range:"
)

print(
    f"  Min: {test_results['anomaly_score'].min():.6f}"
)

print(
    f"  Max: {test_results['anomaly_score'].max():.6f}"
)

print(
    f"  Mean: {test_results['anomaly_score'].mean():.6f}"
)


# ============================================================
# SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

print("\n" + "=" * 60)
print("SELECTING THRESHOLD FROM VALIDATION DATA")
print("=" * 60)

threshold = select_threshold(
    validation_results,
    percentile=THRESHOLD_PERCENTILE
)

print(
    f"Percentile: {THRESHOLD_PERCENTILE}%"
)

print(
    f"Validation threshold: {threshold:.10f}"
)


# ============================================================
# APPLY VALIDATION THRESHOLD TO TEST
# ============================================================

print("\n" + "=" * 60)
print("APPLYING THRESHOLD TO TEST DATA")
print("=" * 60)

test_results = apply_threshold(
    test_results,
    threshold
)

test_anomaly_count = (
    test_results["is_anomaly"].sum()
)

test_anomaly_percentage = (
    test_results["is_anomaly"].mean()
    * 100
)

print(
    "Test anomalies:",
    test_anomaly_count
)

print(
    f"Test anomaly percentage: "
    f"{test_anomaly_percentage:.4f}%"
)


# ============================================================
# THRESHOLD VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD VERIFICATION")
print("=" * 60)

flagged_scores = test_results.loc[
    test_results["is_anomaly"],
    "anomaly_score"
]

normal_scores = test_results.loc[
    ~test_results["is_anomaly"],
    "anomaly_score"
]

if len(flagged_scores) > 0:

    minimum_flagged_score = (
        flagged_scores.min()
    )

    print(
        f"Minimum flagged score: "
        f"{minimum_flagged_score:.10f}"
    )

else:

    minimum_flagged_score = None

    print(
        "Minimum flagged score: No anomalies"
    )


if len(normal_scores) > 0:

    maximum_normal_score = (
        normal_scores.max()
    )

    print(
        f"Maximum non-flagged score: "
        f"{maximum_normal_score:.10f}"
    )

else:

    maximum_normal_score = None

    print(
        "Maximum non-flagged score: "
        "No normal samples"
    )


print(
    f"Threshold: {threshold:.10f}"
)


# Verify flagged observations are >= threshold
if len(flagged_scores) > 0:

    assert (
        flagged_scores >= threshold
    ).all(), (
        "ERROR: Anomaly below threshold detected."
    )


# Verify normal observations are < threshold
if len(normal_scores) > 0:

    assert (
        normal_scores < threshold
    ).all(), (
        "ERROR: Normal epoch above threshold detected."
    )


print(
    "Threshold verification: PASSED"
)


# ============================================================
# TEST ANOMALIES BY SLEEP STAGE
# ============================================================

print("\n" + "=" * 60)
print("TEST ANOMALIES BY SLEEP STAGE")
print("=" * 60)

anomaly_stage_counts = (
    test_results[
        test_results["is_anomaly"]
    ]["sleep_stage"]
    .value_counts()
    .sort_index()
)

print(
    anomaly_stage_counts
)


# ============================================================
# SLEEP STAGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TEST SLEEP-STAGE DISTRIBUTION")
print("=" * 60)

stage_counts = (
    test_results["sleep_stage"]
    .value_counts()
    .sort_index()
)

stage_names = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

for stage, count in stage_counts.items():

    stage_name = stage_names.get(
        int(stage),
        f"Unknown ({stage})"
    )

    print(
        f"Stage {stage} ({stage_name}): "
        f"{count}"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

metrics = evaluate_wake_detection(
    test_results
)

print(
    f"ROC-AUC:   {metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC:    {metrics['pr_auc']:.4f}"
)

print(
    f"Precision: {metrics['precision']:.4f}"
)

print(
    f"Recall:    {metrics['recall']:.4f}"
)

print(
    f"F1-score:  {metrics['f1']:.4f}"
)

print("\nConfusion Matrix:")

print(
    metrics["confusion_matrix"]
)


# ============================================================
# CANDIDATE RESTLESSNESS ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CANDIDATE RESTLESSNESS ANALYSIS")
print("=" * 60)

sleep_anomalies, restlessness_summary = (
    analyze_sleep_anomalies(
        test_results
    )
)

if len(restlessness_summary) > 0:

    print(
        restlessness_summary
    )

else:

    print(
        "No non-Wake anomalous epochs detected."
    )


# ============================================================
# ANOMALY EVENTS
# ============================================================

print("\n" + "=" * 60)
print("ANOMALY EVENTS")
print("=" * 60)

events = detect_anomaly_events(
    test_results
)

if len(events) > 0:

    print(
        events.to_string(
            index=False
        )
    )

else:

    print(
        "No anomaly events detected."
    )


print(
    "\nTotal anomaly events:",
    len(events)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

print(
    f"Total recordings: "
    f"{features['recording_id'].nunique()}"
)

print(
    f"Training epochs: "
    f"{len(train)}"
)

print(
    f"Validation epochs: "
    f"{len(validation)}"
)

print(
    f"Test epochs: "
    f"{len(test)}"
)

print(
    f"EIF trees: "
    f"{N_TREES}"
)

print(
    f"EIF sample size: "
    f"{SAMPLE_SIZE}"
)

print(
    f"Threshold percentile: "
    f"{THRESHOLD_PERCENTILE}%"
)

print(
    f"Threshold: "
    f"{threshold:.10f}"
)

print(
    f"Test anomalies: "
    f"{test_anomaly_count}"
)

print(
    f"Test anomaly percentage: "
    f"{test_anomaly_percentage:.4f}%"
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
    f"F1-score: "
    f"{metrics['f1']:.4f}"
)

print("\nPipeline completed successfully.")