import nibabel as nib
from pathlib import Path

image_path = Path("data/raw/brain_t1.nii.gz")

print("=" * 50)
print("REAL MEDICAL IMAGE TEST")
print("=" * 50)

image = nib.load(image_path)
data = image.get_fdata()

print("File:", image_path)
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Minimum:", data.min())
print("Maximum:", data.max())
print("Voxel spacing:", image.header.get_zooms())

print("=" * 50)
print("REAL NIFTI TEST SUCCESSFUL!")
print("=" * 50)