from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# 1. LOAD FINAL ANN DATASET
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "outputs" / "ann_dataset_glcm_hog_ki67.csv"

df = pd.read_csv(DATA_FILE)


# ==================================================
# 2. CREATE ONE ROW PER PATIENT
# ==================================================

patient_df = (
    df[
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


# ==================================================
# 3. PATIENT-WISE TRAIN / TEST SPLIT
#
# 80% patients -> training
# 20% patients -> testing
#
# Stratify keeps Low/High classes balanced
# as much as possible.
# ==================================================

train_patients, test_patients = train_test_split(

    patient_df,

    test_size=0.20,

    random_state=42,

    stratify=patient_df["Ki67_Label"]
)


# ==================================================
# 4. GET PATIENT ID LISTS
# ==================================================

train_patient_ids = (
    train_patients["Patient_ID"].tolist()
)

test_patient_ids = (
    test_patients["Patient_ID"].tolist()
)


# ==================================================
# 5. CREATE IMAGE-LEVEL TRAIN / TEST DATASETS
# ==================================================

train_df = df[
    df["Patient_ID"].isin(
        train_patient_ids
    )
].copy()


test_df = df[
    df["Patient_ID"].isin(
        test_patient_ids
    )
].copy()


# ==================================================
# 6. PRINT PATIENT INFORMATION
# ==================================================

print("\n")
print("=" * 70)
print("TRAIN PATIENTS")
print("=" * 70)

print(
    train_patients.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("TEST PATIENTS")
print("=" * 70)

print(
    test_patients.to_string(
        index=False
    )
)


# ==================================================
# 7. SUMMARY
# ==================================================

print("\n")
print("=" * 70)
print("PATIENT-WISE SPLIT SUMMARY")
print("=" * 70)

print(
    f"Total patients: "
    f"{patient_df['Patient_ID'].nunique()}"
)

print(
    f"Training patients: "
    f"{train_patients['Patient_ID'].nunique()}"
)

print(
    f"Testing patients: "
    f"{test_patients['Patient_ID'].nunique()}"
)


print("\nTraining patient labels:")
print(
    train_patients[
        "Ki67_Label"
    ].value_counts()
)


print("\nTesting patient labels:")
print(
    test_patients[
        "Ki67_Label"
    ].value_counts()
)


print("\nTraining image rows:")
print(
    len(train_df)
)

print("\nTesting image rows:")
print(
    len(test_df)
)


# ==================================================
# 8. CHECK DATA LEAKAGE
# ==================================================

overlap = set(
    train_patient_ids
).intersection(
    set(test_patient_ids)
)


print("\n")
print("=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

print(
    f"Patients appearing in both train and test: "
    f"{len(overlap)}"
)

if len(overlap) == 0:
    print(
        "PASS: No patient leakage detected."
    )

else:
    print(
        "WARNING: Patient leakage detected."
    )
