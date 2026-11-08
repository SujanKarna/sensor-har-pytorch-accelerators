
import sys
from pathlib import Path

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from paths import (
    TRAIN_SIGNALS_DIR,
    TEST_SIGNALS_DIR,
    TRAIN_LABELS_FILE,
    TEST_LABELS_FILE,
    ACTIVITY_LABELS_FILE,
)


# The six inertial sensor channels
CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]


def load_signals(split):
    """Load and combine the six sensor channels for one dataset split."""

    if split == "train":
        signals_dir = TRAIN_SIGNALS_DIR
    elif split == "test":
        signals_dir = TEST_SIGNALS_DIR
    else:
        raise ValueError("split must be either 'train' or 'test'")

    channel_data = []

    for channel in CHANNELS:
        file_path = signals_dir / f"{channel}_{split}.txt"

        # Each file contains one sensor channel
        data = np.loadtxt(file_path)

        channel_data.append(data)

        print(f"{channel}: {data.shape}")

    # Combine six channels into one array
    signals = np.stack(channel_data, axis=-1)

    return signals


def load_labels(split):
    """Load activity labels for one dataset split."""

    if split == "train":
        labels_file = TRAIN_LABELS_FILE
    elif split == "test":
        labels_file = TEST_LABELS_FILE
    else:
        raise ValueError("split must be either 'train' or 'test'")

    return np.loadtxt(labels_file, dtype=int)


def load_activity_labels():
    """Read activity IDs and their corresponding names."""

    activity_labels = {}

    with open(ACTIVITY_LABELS_FILE, "r") as file:
        for line in file:
            activity_id, activity_name = line.strip().split()
            activity_labels[int(activity_id)] = activity_name

    return activity_labels


def inspect_split(split, activity_labels):
    """Inspect signals, labels, dimensions, and data quality."""

    print(f"\n{'=' * 50}")
    print(f"INSPECTING {split.upper()} DATA")
    print(f"{'=' * 50}")

    signals = load_signals(split)
    labels = load_labels(split)

    print("\nCombined signal shape:", signals.shape)
    print("Labels shape:", labels.shape)

    print("\nData quality:")
    print("All values finite:", np.isfinite(signals).all())
    print("Missing/NaN values:", np.isnan(signals).sum())

    print("\nActivity distribution:")

    unique_ids, counts = np.unique(labels, return_counts=True)

    for activity_id, count in zip(unique_ids, counts):
        activity_name = activity_labels[activity_id]
        print(f"{activity_name}: {count}")

    # Basic consistency checks
    assert signals.ndim == 3, "Signals should have 3 dimensions"
    assert signals.shape[0] == labels.shape[0], (
        "Number of signal samples and labels must match"
    )
    assert signals.shape[1] == 128, "Each sample should have 128 readings"
    assert signals.shape[2] == 6, "Each sample should have 6 channels"

    return signals, labels


def main():
    print("Loading activity labels...")
    activity_labels = load_activity_labels()

    print("\nActivity classes:")
    for activity_id, activity_name in activity_labels.items():
        print(f"{activity_id}: {activity_name}")

    train_signals, train_labels = inspect_split(
        "train", activity_labels
    )

    test_signals, test_labels = inspect_split(
        "test", activity_labels
    )

    print("\n" + "=" * 50)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 50)

    print("Training samples:", len(train_signals))
    print("Testing samples:", len(test_signals))
    print("Total samples:", len(train_signals) + len(test_signals))
    print("Time steps per sample:", train_signals.shape[1])
    print("Sensor channels:", train_signals.shape[2])
    print("Number of activities:", len(activity_labels))


if __name__ == "__main__":
    main()
