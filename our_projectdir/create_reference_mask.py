import nibabel as nib
import numpy as np
from pathlib import Path

image_path = Path("data/raw/brain_t1.nii.gz")
mask_path = Path("data/samples/reference_mask.nii.gz")

print("=" * 50)
print("REFERENCE MASK CREATION")
print("=" * 50)

# Load MRI
image = nib.load(image_path)
volume = image.get_fdata()

print("Image shape:", volume.shape)

# Remove very low-intensity background
threshold = np.percentile(volume, 30)

mask = volume > threshold

# Convert to binary uint8 mask
mask = mask.astype(np.uint8)

# Save mask using the same spatial information
mask_image = nib.Nifti1Image(
    mask,
    image.affine,
    image.header
)

nib.save(mask_image, mask_path)

print("Threshold:", threshold)
print("Mask shape:", mask.shape)
print("Mask values:", np.unique(mask))
print("Saved:", mask_path)

print("=" * 50)
print("REFERENCE MASK CREATED!")
print("=" * 50)