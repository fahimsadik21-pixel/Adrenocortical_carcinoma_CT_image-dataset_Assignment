# Adrenocortical Carcinoma CT Image Feature Analysis

This repository contains the assignment implementation for extracting texture and shape features from CT DICOM slices and classifying patient-level Ki67 status with an artificial neural network (ANN).

The workflow is organized as:

1. Extract 45-degree gray-level co-occurrence matrix (GLCM) features.
2. Extract histogram of oriented gradients (HOG) features.
3. Join image features with the clinical Ki67 labels.
4. Aggregate image features patient-wise to prevent slice-level leakage.
5. Standardize features, reduce them to two principal components (PCA=2), and evaluate a small ANN with 5-fold stratified patient-level cross-validation.

## Dataset

The local dataset is expected under `dataset/` and the supporting clinical spreadsheet under `clinical_data/`. These source files are intentionally excluded from Git because they are large and their redistribution/licensing terms are not established here. Obtain them through the approved course or dataset source before running the pipeline.

The data used for the reported run contained 15 patients: 8 low-Ki67 and 7 high-Ki67 patients. Binary labels are created from the clinical Ki67 percentage using:

- Ki67 `< 10` → label `0` (low)
- Ki67 `>= 10` → label `1` (high)

## Feature extraction

### GLCM

`glcm_task.py` reads 2-D non-segmentation DICOM slices, normalizes each image to 0–255, and computes a GLCM with distance 1 and angle 45° (`π/4`). It saves dissimilarity, correlation, homogeneity, contrast, and energy.

### HOG

`hog_task.py` converts each slice to a 128×128 grayscale image and uses:

- orientations: 9
- pixels per cell: 16×16
- cells per block: 2×2
- block normalization: L2-Hys
- transform square root: disabled

These settings produce 1,764 HOG features per image. The selected settings and processing counts are recorded in `outputs/hog_settings.txt`.

## Patient-level ANN methodology

`final_patient_ann_pca2.py` creates one row per patient. It uses five summary statistics (mean, standard deviation, median, minimum, and maximum) for each of the five GLCM features and the mean of each HOG feature. This produces 1,789 model features per patient (25 GLCM summaries + 1,764 mean HOG features).

For every fold, the scaler and PCA transformation are fitted on the training patients only. PCA retains two components. The ANN is:

```text
Input(2) → Dense(8, ReLU, L2=0.01) → Dropout(0.20)
         → Dense(4, ReLU, L2=0.01) → Dense(1, sigmoid)
```

It is trained with Adam (learning rate 0.001), binary cross-entropy, 150 epochs, and batch size 4. The random seed is 42 and the evaluation uses 5-fold stratified cross-validation at the patient level.

## Final recorded result

The committed result is the completed PCA=2 patient-level run:

```text
Mean 5-fold accuracy:       80.00%
Overall patient accuracy:   80.00%
Accuracy standard deviation: 0.1633
Confusion matrix:            [[7, 1], [2, 5]]
Macro F1:                    0.7964
Weighted F1:                 0.7982
```

This is a tuned cross-validation result on only 15 patients, not an independent external test set. It should be presented as an assignment experiment and interpreted cautiously because the sample is small and PCA/model settings were selected during experimentation.

## Repository contents

Core scripts:

- `glcm_task.py` — image-level GLCM extraction.
- `hog_task.py` — image-level HOG extraction.
- `prepare_ann_dataset.py` — merge GLCM, HOG, and clinical labels.
- `patient_split.py` — inspect a patient-wise train/test split.
- `prepare_ann_inputs.py` — prepare a patient-wise ANN input split.
- `final_patient_ann_pca2.py` — final patient-level PCA=2 ANN evaluation.

Utilities retained from the extraction work include `count_dicom.py`, `test_glcm.py`, and `view_dicom.py`.

Compact derived artifacts in `outputs/` include the GLCM CSV, HOG settings, patient-level feature table, prediction CSV, and final result text. The full HOG matrix and image-level combined ANN matrix are not committed because each is larger than GitHub's 100 MB single-file limit; they can be regenerated locally.

## How to run

From the repository root, create and activate a Python 3.12 environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the approved source data in the expected local folders and run the pipeline in order:

```bash
python glcm_task.py
python hog_task.py
python prepare_ann_dataset.py
python final_patient_ann_pca2.py
```

The two inspection helpers can be run after the combined image-level dataset exists:

```bash
python patient_split.py
python prepare_ann_inputs.py
```

All core scripts resolve `dataset/`, `clinical_data/`, and `outputs/` relative to the repository, so no machine-specific drive letter is required.
