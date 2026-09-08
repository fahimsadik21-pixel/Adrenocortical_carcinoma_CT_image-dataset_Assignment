import os

dataset_path = r"E:\Computer vision\Projects\dataset"

count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".dcm"):
            count += 1

print("Total DICOM files:", count)