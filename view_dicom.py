import pydicom
import matplotlib.pyplot as plt

file_path = r"E:\Computer vision\Projects\dataset\manifest-1788901441878\Adrenal-ACC-Ki67-Seg\Adrenal_Ki67_Seg_001\08-22-2000-NA-CT ABDOMEN-56266\2.000000-Pre Abd 5.0 B40f-18492\1-01.dcm"

ds = pydicom.dcmread(file_path)
image = ds.pixel_array

plt.imshow(image, cmap="gray")
plt.title("DICOM CT Slice")
plt.axis("off")
plt.show()