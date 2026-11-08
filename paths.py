
from pathlib import Path


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ==================================================
# MAIN PROJECT DIRECTORIES
# ==================================================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"


# ==================================================
# DATASET PATHS
# ==================================================

DATASET_NAME = "UCI HAR Dataset"

DATASET_DIR = RAW_DATA_DIR / DATASET_NAME

OUTER_ARCHIVE = RAW_DATA_DIR / "uci_har_dataset.zip"
INNER_ARCHIVE = RAW_DATA_DIR / "UCI HAR Dataset.zip"


# Dataset partitions
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"


# Sensor signal directories
TRAIN_SIGNALS_DIR = TRAIN_DIR / "Inertial Signals"
TEST_SIGNALS_DIR = TEST_DIR / "Inertial Signals"


# Label and feature information
ACTIVITY_LABELS_FILE = DATASET_DIR / "activity_labels.txt"
FEATURES_FILE = DATASET_DIR / "features.txt"
FEATURES_INFO_FILE = DATASET_DIR / "features_info.txt"


# Training and testing metadata
TRAIN_LABELS_FILE = TRAIN_DIR / "y_train.txt"
TEST_LABELS_FILE = TEST_DIR / "y_test.txt"

TRAIN_SUBJECTS_FILE = TRAIN_DIR / "subject_train.txt"
TEST_SUBJECTS_FILE = TEST_DIR / "subject_test.txt"


# ==================================================
# OUTPUT DIRECTORIES
# ==================================================

MODEL_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"


# ==================================================
# DIRECTORY INITIALIZATION
# ==================================================

def create_project_directories():
    """Create directories required by the project."""

    directories = [
        RAW_DATA_DIR,
        SRC_DIR,
        NOTEBOOKS_DIR,
        RESULTS_DIR,
        MODEL_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
