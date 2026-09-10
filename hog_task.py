import os
from pathlib import Path

import cv2
import pydicom
import numpy as np
import pandas as pd

from skimage.feature import hog


# ==================================================
# 1. PATH SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_CSV = os.path.join(
    OUTPUT_FOLDER,
    "hog_features.csv"
)

SETTINGS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "hog_settings.txt"
)

ERROR_LOG = os.path.join(
    OUTPUT_FOLDER,
    "hog_failed_files.txt"
)


# ==================================================
# 2. HOG IMPLEMENTATION SETTINGS
# ==================================================

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

ORIENTATIONS = 9

PIXELS_PER_CELL = (
    16,
    16
)

CELLS_PER_BLOCK = (
    2,
    2
)

BLOCK_NORM = "L2-Hys"

TRANSFORM_SQRT = False


# ==================================================
# 3. CREATE OUTPUT FOLDER
# ==================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==================================================
# 4. VARIABLES
# ==================================================

results = []

processed_count = 0
failed_count = 0

skipped_segmentation = 0
skipped_temp_files = 0
skipped_non_2d = 0

hog_feature_count = 0


# ==================================================
# 5. CLEAR OLD ERROR LOG
# ==================================================

with open(
    ERROR_LOG,
    "w",
    encoding="utf-8"
) as f:

    f.write("HOG FAILED FILES LOG\n")
    f.write("=" * 80 + "\n\n")


# ==================================================
# 6. SEARCH ALL DICOM FILES
# ==================================================

for root, dirs, files in os.walk(DATASET_PATH):

    for filename in files:

        # ------------------------------------------
        # Only process .dcm files
        # ------------------------------------------

        if not filename.lower().endswith(".dcm"):
            continue


        # ------------------------------------------
        # Skip temporary files
        # Example: ~$1-02.dcm
        # ------------------------------------------

        if filename.startswith("~$"):

            skipped_temp_files += 1

            continue


        # ------------------------------------------
        # Skip Segmentation series
        # ------------------------------------------

        if "segmentation" in root.lower():

            skipped_segmentation += 1

            continue


        file_path = os.path.join(
            root,
            filename
        )


        try:

            # ======================================
            # 7. READ DICOM FILE
            # ======================================

            ds = pydicom.dcmread(
                file_path
            )


            # ======================================
            # 8. GET PIXEL ARRAY
            # ======================================

            image = ds.pixel_array.astype(
                np.float32
            )


            # ======================================
            # 9. CHECK IMAGE DIMENSION
            # HOG requires a 2D grayscale image
            # ======================================

            if image.ndim != 2:

                skipped_non_2d += 1

                with open(
                    ERROR_LOG,
                    "a",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        f"Skipped non-2D file:\n"
                        f"{file_path}\n"
                    )

                    f.write(
                        f"Shape: {image.shape}\n\n"
                    )

                continue


            # ======================================
            # 10. NORMALIZE IMAGE TO 0-255
            # ======================================

            image = image - image.min()

            if image.max() > 0:

                image = (
                    image / image.max()
                )

            image = (
                image * 255
            ).astype(np.uint8)


            # ======================================
            # 11. RESIZE IMAGE
            # ======================================

            image = cv2.resize(
                image,
                (
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT
                )
            )


            # ======================================
            # 12. EXTRACT HOG FEATURES
            # ======================================

            hog_features = hog(

                image,

                orientations=ORIENTATIONS,

                pixels_per_cell=PIXELS_PER_CELL,

                cells_per_block=CELLS_PER_BLOCK,

                block_norm=BLOCK_NORM,

                transform_sqrt=TRANSFORM_SQRT,

                visualize=False,

                feature_vector=True,

                channel_axis=None
            )


            # ======================================
            # 13. STORE NUMBER OF HOG FEATURES
            # ======================================

            hog_feature_count = len(
                hog_features
            )


            # ======================================
            # 14. GET PATIENT ID
            # ======================================

            patient_id = getattr(
                ds,
                "PatientID",
                "Unknown"
            )


            # ======================================
            # 15. CREATE ROW
            # ======================================

            row = {

                "Patient_ID":
                    patient_id,

                "Image_File":
                    filename,

                "File_Path":
                    file_path
            }


            # ======================================
            # 16. ADD HOG FEATURES
            # HOG_1, HOG_2, HOG_3 ...
            # ======================================

            for i, value in enumerate(
                hog_features
            ):

                row[
                    f"HOG_{i + 1}"
                ] = value


            # ======================================
            # 17. STORE RESULT
            # ======================================

            results.append(
                row
            )


            processed_count += 1


            print(
                f"Processed {processed_count}: "
                f"{filename}"
            )


        except Exception as e:

            # ======================================
            # 18. SAVE FAILED FILE INFORMATION
            # ======================================

            failed_count += 1

            with open(
                ERROR_LOG,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"Failed file:\n"
                    f"{file_path}\n"
                )

                f.write(
                    f"Reason: {e}\n\n"
                )


# ==================================================
# 19. CREATE DATAFRAME
# ==================================================

df = pd.DataFrame(
    results
)


# ==================================================
# 20. SAVE HOG FEATURES TO CSV
# ==================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ==================================================
# 21. SAVE HOG SETTINGS TO TEXT FILE
# ==================================================

with open(
    SETTINGS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "HOG FEATURE EXTRACTION SETTINGS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "NOTE:\n"
    )

    f.write(
        "The homework requires HOG feature extraction "
        "from an image directory and saving the values "
        "to a CSV file.\n"
    )

    f.write(
        "The exact HOG parameter values were not "
        "specified in the provided homework slide.\n"
    )

    f.write(
        "Therefore, the following parameters were "
        "selected as implementation settings.\n\n"
    )


    f.write(
        "DATASET INFORMATION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"Dataset Path: {DATASET_PATH}\n"
    )

    f.write(
        "Input Image Type: DICOM (.dcm)\n"
    )

    f.write(
        "Image Data Used: Original 2D CT slices\n"
    )

    f.write(
        "Segmentation DICOM series: Skipped\n"
    )

    f.write(
        "Temporary ~$ files: Skipped\n"
    )

    f.write(
        "Non-2D images: Skipped\n\n"
    )


    f.write(
        "IMAGE PREPROCESSING SETTINGS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Pixel normalization: 0 to 255\n"
    )

    f.write(
        f"Resize Width: {IMAGE_WIDTH} pixels\n"
    )

    f.write(
        f"Resize Height: {IMAGE_HEIGHT} pixels\n"
    )

    f.write(
        "Image Mode: 2D grayscale\n\n"
    )


    f.write(
        "HOG PARAMETERS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"Orientations: {ORIENTATIONS}\n"
    )

    f.write(
        f"Pixels Per Cell: "
        f"{PIXELS_PER_CELL[0]} x "
        f"{PIXELS_PER_CELL[1]}\n"
    )

    f.write(
        f"Cells Per Block: "
        f"{CELLS_PER_BLOCK[0]} x "
        f"{CELLS_PER_BLOCK[1]}\n"
    )

    f.write(
        f"Block Normalization: {BLOCK_NORM}\n"
    )

    f.write(
        f"Transform Sqrt: {TRANSFORM_SQRT}\n"
    )

    f.write(
        "Feature Vector: True\n"
    )

    f.write(
        "Visualization: False\n"
    )

    f.write(
        "Channel Axis: None\n\n"
    )


    f.write(
        "FEATURE OUTPUT\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"HOG Features Per Image: "
        f"{hog_feature_count}\n"
    )

    f.write(
        f"Successfully Processed Images: "
        f"{processed_count}\n"
    )

    f.write(
        f"Skipped Segmentation Files: "
        f"{skipped_segmentation}\n"
    )

    f.write(
        f"Skipped Temporary Files: "
        f"{skipped_temp_files}\n"
    )

    f.write(
        f"Skipped Non-2D Files: "
        f"{skipped_non_2d}\n"
    )

    f.write(
        f"Failed Files: "
        f"{failed_count}\n"
    )

    f.write(
        f"Total CSV Rows: "
        f"{len(df)}\n\n"
    )


    f.write(
        "OUTPUT FILES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"HOG CSV: {OUTPUT_CSV}\n"
    )

    f.write(
        f"Settings File: {SETTINGS_FILE}\n"
    )

    f.write(
        f"Error Log: {ERROR_LOG}\n"
    )


# ==================================================
# 22. SHOW CSV HEAD
# ==================================================

print("\n")
print("=" * 80)
print("CSV HEAD")
print("=" * 80)

print(
    df.head()
)


# ==================================================
# 23. SHOW CSV TAIL
# ==================================================

print("\n")
print("=" * 80)
print("CSV TAIL")
print("=" * 80)

print(
    df.tail()
)


# ==================================================
# 24. FINAL SUMMARY
# ==================================================

print("\n")
print("=" * 80)
print("TASK 2 COMPLETED")
print("=" * 80)

print(
    f"Successfully processed: "
    f"{processed_count}"
)

print(
    f"Skipped segmentation files: "
    f"{skipped_segmentation}"
)

print(
    f"Skipped temporary files: "
    f"{skipped_temp_files}"
)

print(
    f"Skipped non-2D files: "
    f"{skipped_non_2d}"
)

print(
    f"Failed files: "
    f"{failed_count}"
)

print(
    f"Total rows in CSV: "
    f"{len(df)}"
)

print(
    f"HOG features per image: "
    f"{hog_feature_count}"
)

print(
    f"CSV saved at: "
    f"{OUTPUT_CSV}"
)

print(
    f"HOG settings saved at: "
    f"{SETTINGS_FILE}"
)

print(
    f"Error log saved at: "
    f"{ERROR_LOG}"
)
