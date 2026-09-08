import pydicom
import numpy as np
from skimage.feature import graycomatrix, graycoprops

file_path = r"E:\Computer vision\Projects\dataset\manifest-1788901441878\Adrenal-ACC-Ki67-Seg\Adrenal_Ki67_Seg_001\08-22-2000-NA-CT ABDOMEN-56266\2.000000-Pre Abd 5.0 B40f-18492\1-01.dcm"

ds = pydicom.dcmread(file_path)
image = ds.pixel_array.astype(np.float32)

# Normalize image to 0-255
image = image - image.min()

if image.max() != 0:
    image = image / image.max()

image = (image * 255).astype(np.uint8)

# GLCM at 45 degree only
glcm = graycomatrix(
    image,
    distances=[1],
    angles=[np.pi / 4],
    levels=256,
    symmetric=True,
    normed=True
)

dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
correlation = graycoprops(glcm, "correlation")[0, 0]
homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
contrast = graycoprops(glcm, "contrast")[0, 0]
energy = graycoprops(glcm, "energy")[0, 0]

print("Dissimilarity:", dissimilarity)
print("Correlation:", correlation)
print("Homogeneity:", homogeneity)
print("Contrast:", contrast)
print("Energy:", energy)