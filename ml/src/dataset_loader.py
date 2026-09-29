from pathlib import Path
import pandas as pd

from preprocessing import load_motion_data, load_expert_labels, create_sleep_epochs
from feature_engineering import build_feature_table


def find_recordings(raw_dir):
    """
    Find all recording folders containing both motion.csv and labels.mat.
    """

    raw_dir = Path(raw_dir)
    recordings = []

    for motion_file in raw_dir.rglob("motion.csv"):
        recording_dir = motion_file.parent
        labels_file = recording_dir / "labels.mat"

        if labels_file.exists():
            recordings.append({
                "recording_id": recording_dir.name,
                "motion_path": motion_file,
                "labels_path": labels_file
            })

    return recordings


def process_recording(recording):
    """
    Load and process one BIDSleep recording.
    """

    motion = load_motion_data(recording["motion_path"])

    expert_labels, rec_start = load_expert_labels(
        recording["labels_path"]
    )

    motion_sleep = create_sleep_epochs(
        motion,
        expert_labels,
        rec_start
    )

    features = build_feature_table(
        motion_sleep,
        expert_labels
    )

    features["recording_id"] = recording["recording_id"]

    return features


def load_all_recordings(raw_dir):
    """
    Process every valid recording in the raw data directory.
    """

    recordings = find_recordings(raw_dir)

    if not recordings:
        raise FileNotFoundError(
            f"No recordings found in {raw_dir}"
        )

    all_features = []

    for recording in recordings:

        print(
            f"Processing: {recording['recording_id']}"
        )

        features = process_recording(recording)

        print(
            f"  Epochs: {len(features)}"
        )

        all_features.append(features)

    combined = pd.concat(
        all_features,
        ignore_index=True
    )

    return combined