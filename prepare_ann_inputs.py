from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
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
)


# ==================================================
# 3. PATIENT-WISE TRAIN / TEST SPLIT
# ==================================================

train_patients, test_patients = train_test_split(

    patient_df,

    test_size=0.20,

    random_state=42,

    stratify=patient_df["Ki67_Label"]
)


train_ids = train_patients[
    "Patient_ID"
].tolist()

test_ids = test_patients[
    "Patient_ID"
].tolist()


# ==================================================
# 4. CREATE TRAIN / TEST DATAFRAMES
# ==================================================

train_df = df[
    df["Patient_ID"].isin(train_ids)
].copy()

test_df = df[
    df["Patient_ID"].isin(test_ids)
].copy()


# ==================================================
# 5. SELECT MODEL FEATURES
# ==================================================

glcm_features = [
    "Dissimilarity",
    "Correlation",
    "Homogeneity",
    "Contrast",
    "Energy"
]

hog_features = [
    col
    for col in df.columns
    if col.startswith("HOG_")
]

feature_columns = (
    glcm_features
    +
    hog_features
)


# ==================================================
# 6. CREATE X AND y
# ==================================================

X_train = train_df[
    feature_columns
].values

X_test = test_df[
    feature_columns
].values


y_train = train_df[
    "Ki67_Label"
].values

y_test = test_df[
    "Ki67_Label"
].values


# ==================================================
# 7. FEATURE SCALING
# ==================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==================================================
# 8. SHOW SHAPES
# ==================================================

print("\n")
print("=" * 70)
print("ANN INPUT PREPARATION")
print("=" * 70)

print(
    "Number of model features:",
    len(feature_columns)
)

print(
    "X_train shape:",
    X_train.shape
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_train shape:",
    y_train.shape
)

print(
    "y_test shape:",
    y_test.shape
)


print("\n")
print("=" * 70)
print("AFTER STANDARD SCALING")
print("=" * 70)

print(
    "X_train_scaled shape:",
    X_train_scaled.shape
)

print(
    "X_test_scaled shape:",
    X_test_scaled.shape
)


print("\nTraining label counts:")

print(
    pd.Series(
        y_train
    ).value_counts()
)


print("\nTesting label counts:")

print(
    pd.Series(
        y_test
    ).value_counts()
)
