import numpy as np
import nibabel as nib
from pathlib import Path

print("=" * 50)
print("3D NIFTI TEST")
print("=" * 50)

# Create a small dummy 3D medical volume
volume = np.random.rand(32, 32, 32).astype(np.float32)

# Identity affine matrix
affine = np.eye(4)

# Create NIfTI image
nii = nib.Nifti1Image(volume, affine)

# Save it
output_path = Path("data/samples/test_volume.nii.gz")
nib.save(nii, output_path)

print("Saved:", output_path)

# Read it back
loaded = nib.load(output_path)
data = loaded.get_fdata()

print("Loaded shape:", data.shape)
print("Data type:", data.dtype)
print("Minimum:", data.min())
print("Maximum:", data.max())

print("=" * 50)
print("NIFTI TEST SUCCESSFUL!")
print("=" * 50)