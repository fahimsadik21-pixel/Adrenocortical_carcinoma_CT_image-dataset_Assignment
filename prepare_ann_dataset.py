import os
from pathlib import Path

import pandas as pd


# ==================================================
# 1. FILE PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

GLCM_FILE = OUTPUT_FOLDER / "glcm_features_45_degree.csv"
HOG_FILE = OUTPUT_FOLDER / "hog_features.csv"
CLINICAL_FILE = (
    PROJECT_ROOT
    / "clinical_data"
    / "Adrenal-ACC-Ki67-Seg_SupportingData_20230522.xlsx"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "ann_dataset_glcm_hog_ki67.csv"
)


# ==================================================
# 2. READ FILES
# ==================================================

print("Reading GLCM CSV...")
glcm_df = pd.read_csv(GLCM_FILE)

print("Reading HOG CSV...")
hog_df = pd.read_csv(HOG_FILE)

print("Reading Clinical Excel...")
clinical_df = pd.read_excel(CLINICAL_FILE)


# ==================================================
# 3. ORIGINAL DATASET INFORMATION
# ==================================================

print("\n")
print("=" * 80)
print("ORIGINAL DATASET INFORMATION")
print("=" * 80)

print(f"GLCM rows: {len(glcm_df)}")
print(f"HOG rows: {len(hog_df)}")
print(f"Clinical rows: {len(clinical_df)}")


# ==================================================
# 4. CHECK File_Path UNIQUENESS
# ==================================================

print("\n")
print("=" * 80)
print("FILE PATH CHECK")
print("=" * 80)

print(
    "Duplicate File_Path in GLCM:",
    glcm_df["File_Path"].duplicated().sum()
)

print(
    "Duplicate File_Path in HOG:",
    hog_df["File_Path"].duplicated().sum()
)


# ==================================================
# 5. PREPARE CLINICAL DATA
# ==================================================

clinical_df = clinical_df[
    [
        "Patient_ID",
        "Ki67"
    ]
].copy()


clinical_df = clinical_df.drop_duplicates(
    subset="Patient_ID"
)


# ==================================================
# 6. CREATE Ki67 BINARY LABEL
#
# Ki67 < 10  = 0 = Low
# Ki67 >= 10 = 1 = High
# ==================================================

clinical_df["Ki67_Label"] = (
    clinical_df["Ki67"] >= 10
).astype(int)


# ==================================================
# 7. PREPARE HOG DATA
#
# Patient_ID and Image_File already exist in GLCM.
# Keep File_Path + HOG columns only.
# ==================================================

hog_feature_columns = [
    col
    for col in hog_df.columns
    if col.startswith("HOG_")
]


hog_clean = hog_df[
    ["File_Path"] + hog_feature_columns
].copy()


# ==================================================
# 8. MERGE GLCM + HOG USING EXACT File_Path
# ==================================================

print("\nMerging GLCM and HOG features using File_Path...")


features_df = glcm_df.merge(
    hog_clean,
    on="File_Path",
    how="inner",
    validate="one_to_one"
)


# ==================================================
# 9. MERGE WITH CLINICAL DATA
# ==================================================

print("Adding Ki67 clinical information...")


final_df = features_df.merge(
    clinical_df,
    on="Patient_ID",
    how="left",
    validate="many_to_one"
)


# ==================================================
# 10. CHECK MISSING VALUES
# ==================================================

missing_ki67 = final_df["Ki67"].isna().sum()


# ==================================================
# 11. CHECK DUPLICATE File_Path
# ==================================================

duplicate_rows = final_df["File_Path"].duplicated().sum()


# ==================================================
# 12. DEFINE GLCM FEATURES
# ==================================================

glcm_feature_columns = [
    "Dissimilarity",
    "Correlation",
    "Homogeneity",
    "Contrast",
    "Energy"
]


# ==================================================
# 13. COUNT FEATURES
# ==================================================

hog_feature_count = len(
    hog_feature_columns
)

total_model_features = (
    len(glcm_feature_columns)
    +
    hog_feature_count
)


# ==================================================
# 14. SAVE FINAL ANN DATASET
# ==================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# 15. SHOW HEAD
# ==================================================

print("\n")
print("=" * 80)
print("FINAL DATASET HEAD")
print("=" * 80)

display_columns = [
    "Patient_ID",
    "Image_File",
    "Dissimilarity",
    "Correlation",
    "Homogeneity",
    "Contrast",
    "Energy",
    "Ki67",
    "Ki67_Label"
]

print(
    final_df[
        display_columns
    ].head()
)


# ==================================================
# 16. SHOW TAIL
# ==================================================

print("\n")
print("=" * 80)
print("FINAL DATASET TAIL")
print("=" * 80)

print(
    final_df[
        display_columns
    ].tail()
)


# ==================================================
# 17. IMAGE-LEVEL LABEL DISTRIBUTION
# ==================================================

print("\n")
print("=" * 80)
print("IMAGE-LEVEL LABEL DISTRIBUTION")
print("=" * 80)

print(
    final_df[
        "Ki67_Label"
    ].value_counts()
)


# ==================================================
# 18. PATIENT-LEVEL LABEL DISTRIBUTION
# ==================================================

patient_level_df = (
    final_df[
        [
            "Patient_ID",
            "Ki67",
            "Ki67_Label"
        ]
    ]
    .drop_duplicates(
        subset="Patient_ID"
    )
    .sort_values(
        "Patient_ID"
    )
)


print("\n")
print("=" * 80)
print("PATIENT-LEVEL LABEL DISTRIBUTION")
print("=" * 80)

print(
    patient_level_df[
        "Ki67_Label"
    ].value_counts()
)


# ==================================================
# 19. PATIENT-WISE INFORMATION
# ==================================================

print("\n")
print("=" * 80)
print("PATIENT-WISE Ki67 INFORMATION")
print("=" * 80)

print(
    patient_level_df.to_string(
        index=False
    )
)


# ==================================================
# 20. FINAL SUMMARY
# ==================================================

print("\n")
print("=" * 80)
print("ANN DATASET PREPARATION COMPLETED")
print("=" * 80)

print(
    f"GLCM input rows: "
    f"{len(glcm_df)}"
)

print(
    f"HOG input rows: "
    f"{len(hog_df)}"
)

print(
    f"Merged feature rows: "
    f"{len(features_df)}"
)

print(
    f"Final ANN rows: "
    f"{len(final_df)}"
)

print(
    f"Unique patients: "
    f"{final_df['Patient_ID'].nunique()}"
)

print(
    f"GLCM features per image: "
    f"{len(glcm_feature_columns)}"
)

print(
    f"HOG features per image: "
    f"{hog_feature_count}"
)

print(
    f"Total model features per image: "
    f"{total_model_features}"
)

print(
    f"Missing Ki67 values: "
    f"{missing_ki67}"
)

print(
    f"Duplicate File_Path rows: "
    f"{duplicate_rows}"
)

print(
    f"Final CSV saved at: "
    f"{OUTPUT_FILE}"
)
