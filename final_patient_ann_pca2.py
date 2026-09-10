import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# 1. SETTINGS
# ==================================================

SEED = 42
PCA_COMPONENTS = 2
N_SPLITS = 5

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================
# 2. PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "outputs" / "ann_dataset_glcm_hog_ki67.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "final_patient_ann_pca2_results.txt"
)

PREDICTION_FILE = os.path.join(
    OUTPUT_FOLDER,
    "final_patient_ann_pca2_predictions.csv"
)

PATIENT_FEATURE_FILE = os.path.join(
    OUTPUT_FOLDER,
    "final_patient_level_combined_features.csv"
)


# ==================================================
# 3. LOAD DATA
# ==================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)


# ==================================================
# 4. FEATURE COLUMNS
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


# ==================================================
# 5. PATIENT-LEVEL GLCM AGGREGATION
# ==================================================

print("Creating patient-level GLCM features...")

glcm_patient = (
    df.groupby("Patient_ID")[glcm_features]
    .agg([
        "mean",
        "std",
        "median",
        "min",
        "max"
    ])
)

glcm_patient.columns = [
    f"{feature}_{stat}"
    for feature, stat in glcm_patient.columns
]

glcm_patient = (
    glcm_patient
    .reset_index()
    .copy()
)


# ==================================================
# 6. PATIENT-LEVEL HOG AGGREGATION
# ==================================================

print("Creating patient-level HOG features...")

hog_patient = (
    df.groupby("Patient_ID")[hog_features]
    .mean()
    .reset_index()
    .copy()
)


# Rename HOG columns
hog_rename = {
    col: f"{col}_mean"
    for col in hog_features
}

hog_patient = hog_patient.rename(
    columns=hog_rename
)


# ==================================================
# 7. PATIENT LABELS
# ==================================================

patient_labels = (
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
    .copy()
)


# ==================================================
# 8. MERGE FEATURES + LABELS
# ==================================================

patient_df = pd.merge(
    glcm_patient,
    hog_patient,
    on="Patient_ID",
    how="inner",
    validate="one_to_one"
)

patient_df = pd.merge(
    patient_df,
    patient_labels,
    on="Patient_ID",
    how="inner",
    validate="one_to_one"
)

patient_df = (
    patient_df
    .fillna(0)
    .sort_values("Patient_ID")
    .reset_index(drop=True)
)


# ==================================================
# 9. SAVE PATIENT-LEVEL FEATURE DATASET
# ==================================================

patient_df.to_csv(
    PATIENT_FEATURE_FILE,
    index=False
)


# ==================================================
# 10. MODEL FEATURES
# ==================================================

feature_columns = [
    col
    for col in patient_df.columns
    if col not in [
        "Patient_ID",
        "Ki67",
        "Ki67_Label"
    ]
]


X = patient_df[
    feature_columns
].values


y = patient_df[
    "Ki67_Label"
].values


# ==================================================
# 11. DATASET INFORMATION
# ==================================================

print("\n")
print("=" * 80)
print("FINAL ANN DATASET INFORMATION")
print("=" * 80)

print(
    "Total patients:",
    len(patient_df)
)

print(
    "Low Ki67 patients:",
    int((y == 0).sum())
)

print(
    "High Ki67 patients:",
    int((y == 1).sum())
)

print(
    "Original patient-level features:",
    len(feature_columns)
)

print(
    "PCA components:",
    PCA_COMPONENTS
)


# ==================================================
# 12. CROSS VALIDATION
# ==================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)


fold_accuracies = []

all_patient_ids = []
all_true = []
all_pred = []
all_prob = []
all_folds = []


# ==================================================
# 13. RUN EACH FOLD
# ==================================================

for fold, (train_idx, test_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    print("\n")
    print("=" * 80)
    print(f"FOLD {fold}")
    print("=" * 80)


    # ----------------------------------------------
    # Split
    # ----------------------------------------------

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]


    test_patient_ids = (
        patient_df
        .iloc[test_idx]["Patient_ID"]
        .tolist()
    )


    # ----------------------------------------------
    # Scaling
    # Fit on training fold only
    # ----------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ----------------------------------------------
    # PCA
    # Fit on training fold only
    # ----------------------------------------------

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=SEED
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )


    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )


    print(
        "Train patients:",
        len(train_idx)
    )

    print(
        "Test patients:",
        len(test_idx)
    )

    print(
        f"PCA explained variance: "
        f"{explained_variance:.4f}"
    )


    # ==================================================
    # 14. RESET TENSORFLOW
    # ==================================================

    tf.keras.backend.clear_session()

    np.random.seed(
        SEED + fold
    )

    tf.random.set_seed(
        SEED + fold
    )


    # ==================================================
    # 15. BUILD ANN
    # ==================================================

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(PCA_COMPONENTS,)
        ),

        tf.keras.layers.Dense(
            8,
            activation="relu",
            kernel_regularizer=
            tf.keras.regularizers.l2(0.01)
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.Dense(
            4,
            activation="relu",
            kernel_regularizer=
            tf.keras.regularizers.l2(0.01)
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])


    # ==================================================
    # 16. COMPILE
    # ==================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    # ==================================================
    # 17. TRAIN
    # ==================================================

    model.fit(
        X_train_pca,
        y_train,
        epochs=150,
        batch_size=4,
        verbose=0
    )


    # ==================================================
    # 18. PREDICT
    # ==================================================

    probabilities = model.predict(
        X_test_pca,
        verbose=0
    ).ravel()


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    # ==================================================
    # 19. FOLD ACCURACY
    # ==================================================

    fold_accuracy = accuracy_score(
        y_test,
        predictions
    )


    fold_accuracies.append(
        fold_accuracy
    )


    print(
        f"Fold Accuracy: "
        f"{fold_accuracy:.4f}"
    )


    # ==================================================
    # 20. STORE RESULTS
    # ==================================================

    for pid, actual, prob, pred in zip(
        test_patient_ids,
        y_test,
        probabilities,
        predictions
    ):

        all_patient_ids.append(
            pid
        )

        all_true.append(
            int(actual)
        )

        all_prob.append(
            float(prob)
        )

        all_pred.append(
            int(pred)
        )

        all_folds.append(
            fold
        )


        print(
            f"{pid} | "
            f"Actual={actual} | "
            f"Probability={prob:.4f} | "
            f"Predicted={pred}"
        )


# ==================================================
# 21. FINAL METRICS
# ==================================================

all_true = np.array(
    all_true
)

all_pred = np.array(
    all_pred
)


overall_accuracy = accuracy_score(
    all_true,
    all_pred
)


mean_accuracy = np.mean(
    fold_accuracies
)


std_accuracy = np.std(
    fold_accuracies
)


cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1]
)


report = classification_report(
    all_true,
    all_pred,
    labels=[0, 1],
    digits=4,
    zero_division=0
)


# ==================================================
# 22. PRINT FINAL RESULTS
# ==================================================

print("\n")
print("=" * 80)
print("FINAL PCA=2 PATIENT-LEVEL ANN RESULTS")
print("=" * 80)


for i, acc in enumerate(
    fold_accuracies,
    start=1
):

    print(
        f"Fold {i} Accuracy: "
        f"{acc:.4f}"
    )


print(
    f"\nMean 5-Fold Accuracy: "
    f"{mean_accuracy:.4f}"
)


print(
    f"Mean 5-Fold Accuracy (%): "
    f"{mean_accuracy * 100:.2f}%"
)


print(
    f"Accuracy Standard Deviation: "
    f"{std_accuracy:.4f}"
)


print(
    f"Overall Patient Accuracy: "
    f"{overall_accuracy:.4f}"
)


print(
    f"Overall Patient Accuracy (%): "
    f"{overall_accuracy * 100:.2f}%"
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(report)


# ==================================================
# 23. SAVE PREDICTION CSV
# ==================================================

prediction_df = pd.DataFrame({

    "Fold":
        all_folds,

    "Patient_ID":
        all_patient_ids,

    "Actual_Label":
        all_true,

    "Predicted_Probability":
        all_prob,

    "Predicted_Label":
        all_pred
})


prediction_df = (
    prediction_df
    .sort_values("Patient_ID")
    .reset_index(drop=True)
)


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# ==================================================
# 24. SAVE RESULT TXT
# ==================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL PCA=2 PATIENT-LEVEL ANN RESULTS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Total Patients: "
        f"{len(patient_df)}\n"
    )

    f.write(
        f"Low Ki67 Patients: "
        f"{int((y == 0).sum())}\n"
    )

    f.write(
        f"High Ki67 Patients: "
        f"{int((y == 1).sum())}\n"
    )

    f.write(
        f"Original Patient-Level Features: "
        f"{len(feature_columns)}\n"
    )

    f.write(
        f"PCA Components: "
        f"{PCA_COMPONENTS}\n"
    )

    f.write(
        "Cross Validation: "
        "5-Fold Stratified\n\n"
    )


    for i, acc in enumerate(
        fold_accuracies,
        start=1
    ):

        f.write(
            f"Fold {i} Accuracy: "
            f"{acc:.4f}\n"
        )


    f.write(
        f"\nMean 5-Fold Accuracy: "
        f"{mean_accuracy:.4f}\n"
    )

    f.write(
        f"Mean 5-Fold Accuracy (%): "
        f"{mean_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Accuracy Standard Deviation: "
        f"{std_accuracy:.4f}\n"
    )

    f.write(
        f"Overall Patient Accuracy: "
        f"{overall_accuracy:.4f}\n"
    )

    f.write(
        f"Overall Patient Accuracy (%): "
        f"{overall_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(
        report
    )


# ==================================================
# 25. FILE LOCATIONS
# ==================================================

print("\n")
print("=" * 80)
print("FINAL FILES SAVED")
print("=" * 80)

print(
    "Patient features:"
)

print(
    PATIENT_FEATURE_FILE
)

print(
    "\nPredictions:"
)

print(
    PREDICTION_FILE
)

print(
    "\nResults:"
)

print(
    RESULT_FILE
)
