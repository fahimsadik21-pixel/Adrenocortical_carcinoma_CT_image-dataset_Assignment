import os
from pathlib import Path

import pydicom
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


# ==================================================
# 1. PATH SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_CSV = os.path.join(
    OUTPUT_FOLDER,
    "glcm_features_45_degree.csv"
)

ERROR_LOG = os.path.join(
    OUTPUT_FOLDER,
    "glcm_failed_files.txt"
)


# ==================================================
# 2. CREATE OUTPUT FOLDER
# ==================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==================================================
# 3. VARIABLES
# ==================================================

results = []

processed_count = 0
failed_count = 0
skipped_segmentation = 0
skipped_non_2d = 0
skipped_temp_files = 0


# ==================================================
# 4. CLEAR OLD ERROR LOG
# ==================================================

with open(
    ERROR_LOG,
    "w",
    encoding="utf-8"
) as f:

    f.write("GLCM FAILED FILES LOG\n")
    f.write("=" * 80 + "\n\n")


# ==================================================
# 5. SEARCH ALL FILES RECURSIVELY
# ==================================================

for root, dirs, files in os.walk(DATASET_PATH):

    for filename in files:

        # ------------------------------------------
        # Only process DICOM files
        # ------------------------------------------

        if not filename.lower().endswith(".dcm"):
            continue


        # ------------------------------------------
        # Skip temporary files like ~$1-02.dcm
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
            # 6. READ DICOM FILE
            # ======================================

            ds = pydicom.dcmread(
                file_path
            )


            # ======================================
            # 7. GET IMAGE PIXEL DATA
            # ======================================

            image = ds.pixel_array.astype(
                np.float32
            )


            # ======================================
            # 8. CHECK IMAGE DIMENSION
            # GLCM needs a 2D image
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
            # 9. NORMALIZE IMAGE TO 0-255
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
            # 10. CREATE GLCM
            # 45 degree only = pi / 4
            # ======================================

            glcm = graycomatrix(
                image,
                distances=[1],
                angles=[np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True
            )


            # ======================================
            # 11. EXTRACT GLCM FEATURES
            # ======================================

            dissimilarity = graycoprops(
                glcm,
                "dissimilarity"
            )[0, 0]

            correlation = graycoprops(
                glcm,
                "correlation"
            )[0, 0]

            homogeneity = graycoprops(
                glcm,
                "homogeneity"
            )[0, 0]

            contrast = graycoprops(
                glcm,
                "contrast"
            )[0, 0]

            energy = graycoprops(
                glcm,
                "energy"
            )[0, 0]


            # ======================================
            # 12. GET PATIENT ID
            # ======================================

            patient_id = getattr(
                ds,
                "PatientID",
                "Unknown"
            )


            # ======================================
            # 13. STORE RESULT
            # ======================================

            results.append({

                "Patient_ID":
                    patient_id,

                "Image_File":
                    filename,

                "File_Path":
                    file_path,

                "Dissimilarity":
                    dissimilarity,

                "Correlation":
                    correlation,

                "Homogeneity":
                    homogeneity,

                "Contrast":
                    contrast,

                "Energy":
                    energy
            })


            processed_count += 1


            # ======================================
            # 14. SHOW PROCESSING STATUS
            # ======================================

            print(
                f"Processed {processed_count}: "
                f"{filename}"
            )


        except Exception as e:

            # ======================================
            # 15. SAVE FAILED FILE INFO
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
# 16. CREATE DATAFRAME
# ==================================================

df = pd.DataFrame(
    results
)


# ==================================================
# 17. SAVE CSV
# ==================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ==================================================
# 18. SHOW CSV HEAD
# ==================================================

print("\n")
print("=" * 80)
print("CSV HEAD")
print("=" * 80)

print(
    df.head()
)


# ==================================================
# 19. SHOW CSV TAIL
# ==================================================

print("\n")
print("=" * 80)
print("CSV TAIL")
print("=" * 80)

print(
    df.tail()
)


# ==================================================
# 20. FINAL SUMMARY
# ==================================================

print("\n")
print("=" * 80)
print("TASK 1 COMPLETED")
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
    f"CSV saved at: "
    f"{OUTPUT_CSV}"
)

print(
    f"Error log saved at: "
    f"{ERROR_LOG}"
)
