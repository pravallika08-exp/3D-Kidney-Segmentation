import nibabel as nib
import numpy as np
from pathlib import Path

IMAGE_PATH = Path("data/raw/brain_t1.nii.gz")
MASK_PATH = Path("data/samples/reference_mask.nii.gz")

OUTPUT_IMAGE = Path("data/processed/image.npy")
OUTPUT_MASK = Path("data/processed/mask.npy")


print("=" * 50)
print("MASS PROJECT - PREPROCESSING")
print("=" * 50)


# -----------------------------
# 1. Load image and mask
# -----------------------------

image_nii = nib.load(IMAGE_PATH)
mask_nii = nib.load(MASK_PATH)

image = image_nii.get_fdata().astype(np.float32)
mask = mask_nii.get_fdata().astype(np.uint8)

print("Original image shape:", image.shape)
print("Original mask shape :", mask.shape)


# -----------------------------
# 2. Normalize image
# -----------------------------

minimum = image.min()
maximum = image.max()

if maximum > minimum:
    image = (image - minimum) / (maximum - minimum)
else:
    image = np.zeros_like(image)

print("Normalized image range:")
print("Minimum:", image.min())
print("Maximum:", image.max())


# -----------------------------
# 3. Make sure mask is binary
# -----------------------------

mask = (mask > 0).astype(np.uint8)

print("Mask values:", np.unique(mask))


# -----------------------------
# 4. Save processed data
# -----------------------------

OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT_IMAGE, image)
np.save(OUTPUT_MASK, mask)

print()
print("Saved image:", OUTPUT_IMAGE)
print("Saved mask :", OUTPUT_MASK)

print("=" * 50)
print("PREPROCESSING SUCCESSFUL!")
print("=" * 50)