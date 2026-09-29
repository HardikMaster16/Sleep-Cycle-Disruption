import numpy as np
import eif as iso

from sklearn.preprocessing import StandardScaler


def split_by_recording(
    features,
    train_recordings,
    validation_recordings,
    test_recordings
):
    """
    Split data by complete recording/night.

    No epochs from the same recording appear
    in different splits.
    """

    train = features[
        features["recording_id"].isin(train_recordings)
    ].copy()

    validation = features[
        features["recording_id"].isin(validation_recordings)
    ].copy()

    test = features[
        features["recording_id"].isin(test_recordings)
    ].copy()

    return train, validation, test


def fit_scaler(train_features, feature_columns):
    """
    Fit the scaler ONLY on training data.
    """

    scaler = StandardScaler()

    X_train = train_features[
        feature_columns
    ].copy()

    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled, scaler


def transform_features(
    features,
    feature_columns,
    scaler
):
    """
    Transform validation/test data using
    the scaler fitted on training data.
    """

    X = features[
        feature_columns
    ].copy()

    X_scaled = scaler.transform(X)

    return X_scaled


def train_eif(
    X_train_scaled,
    ntrees=200,
    sample_size=256,
    random_seed=42
):
    """
    Train Extended Isolation Forest.
    """

    np.random.seed(random_seed)

    n_features = X_train_scaled.shape[1]

    forest = iso.iForest(
        X_train_scaled,
        ntrees=ntrees,
        sample_size=sample_size,
        ExtensionLevel=n_features - 1
    )

    return forest


def calculate_anomaly_scores(
    forest,
    X_scaled
):
    """
    Calculate EIF anomaly scores.

    Higher score = more anomalous.
    """

    scores = forest.compute_paths(
        X_in=X_scaled
    )

    return scores


def add_anomaly_scores(
    features,
    scores
):
    """
    Add anomaly scores to a feature dataframe.
    """

    results = features.copy()

    results["anomaly_score"] = scores

    return results


def select_threshold(
    validation_results,
    percentile=95
):
    """
    Select anomaly threshold using validation data ONLY.
    """

    threshold = validation_results[
        "anomaly_score"
    ].quantile(percentile / 100)

    return threshold


def apply_threshold(
    results,
    threshold
):
    """
    Apply a previously selected threshold.
    """

    results = results.copy()

    results["is_anomaly"] = (
        results["anomaly_score"] >= threshold
    )

    return results