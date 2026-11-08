
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile, BadZipFile
import shutil
import sys
from paths import (
    RAW_DATA_DIR,
    OUTER_ARCHIVE,
    INNER_ARCHIVE,
    DATASET_DIR,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------



DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

EXPECTED_FILES = [
    "train/Inertial Signals/body_acc_x_train.txt",
    "train/Inertial Signals/body_gyro_z_train.txt",
    "test/Inertial Signals/body_acc_x_test.txt",
    "test/Inertial Signals/body_gyro_z_test.txt",
    "activity_labels.txt",
    "features.txt",
]


# --------------------------------------------------
# Validation
# --------------------------------------------------


def is_dataset_ready():
    """Silently check whether all required dataset files exist."""

    return all(
        (DATASET_DIR / file).is_file()
        for file in EXPECTED_FILES
    )


def validate_dataset():
    """Validate the extracted dataset and report problems."""

    missing = [
        file for file in EXPECTED_FILES
        if not (DATASET_DIR / file).is_file()
    ]

    if missing:
        print("Dataset validation failed:")

        for file in missing:
            print(f"  Missing: {file}")

        return False

    print("Dataset validation successful.")
    return True

# --------------------------------------------------
# Download outer ZIP
# --------------------------------------------------

def download_dataset():

    if OUTER_ARCHIVE.exists():
        try:
            with ZipFile(OUTER_ARCHIVE, "r") as archive:
                if archive.testzip() is None:
                    print("Outer archive already exists.")
                    return
        except BadZipFile:
            print("Invalid outer archive. Downloading again.")

    print("Downloading UCI HAR dataset...")

    temp_file = OUTER_ARCHIVE.with_suffix(".zip.part")

    try:
        with urlopen(DATASET_URL, timeout=60) as response:
            with open(temp_file, "wb") as output:
                shutil.copyfileobj(response, output)

        with ZipFile(temp_file, "r") as archive:
            if archive.testzip() is not None:
                raise BadZipFile("Outer ZIP integrity check failed.")

        temp_file.replace(OUTER_ARCHIVE)

        print("Download completed.")

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


# --------------------------------------------------
# Extract inner ZIP from outer ZIP
# --------------------------------------------------

def extract_inner_archive():

    if INNER_ARCHIVE.exists():
        print("Inner archive already exists.")
        return

    print("Extracting inner ZIP archive...")

    with ZipFile(OUTER_ARCHIVE, "r") as archive:

        inner_name = next(
            (
                name for name in archive.namelist()
                if Path(name).name == "UCI HAR Dataset.zip"
            ),
            None
        )

        if inner_name is None:
            raise FileNotFoundError(
                "UCI HAR Dataset.zip not found inside outer archive."
            )

        with archive.open(inner_name) as source:
            with open(INNER_ARCHIVE, "wb") as destination:
                shutil.copyfileobj(source, destination)

    print("Inner ZIP extracted successfully.")


# --------------------------------------------------
# Extract actual dataset
# --------------------------------------------------


def extract_dataset():
    """Extract the actual dataset from the inner ZIP."""

    print("Extracting actual dataset...")

    with ZipFile(INNER_ARCHIVE, "r") as archive:

        if archive.testzip() is not None:
            raise BadZipFile("Inner ZIP integrity check failed.")

        archive.extractall(RAW_DATA_DIR)

    print("Dataset extraction completed.")

# --------------------------------------------------
# Main
# --------------------------------------------------


def main():

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:

        # Step 1: Check whether the dataset is already available.
        if is_dataset_ready():
            print("Dataset is already ready.")
            return

        # Step 2: Download the outer ZIP if necessary.
        download_dataset()

        # Step 3: Extract the inner ZIP.
        extract_inner_archive()

        # Step 4: Extract the actual dataset.
        extract_dataset()

        # Step 5: Validate after extraction.
        if not validate_dataset():
            raise RuntimeError(
                "Dataset validation failed after extraction."
            )

        print("\nUCI HAR dataset is ready!")
        print(f"Dataset location: {DATASET_DIR}")

    except Exception as error:

        print(f"\nERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
