from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


input_dir = Path("data/real/kits23_case00002")
output_dir = input_dir / "processed"

output_dir.mkdir(parents=True, exist_ok=True)

image_path = input_dir / "imaging.nii.gz"
mask_path = input_dir / "segmentation.nii.gz"

print("Loading image...")
image = nib.load(image_path).get_fdata().astype(np.float32)

print("Loading mask...")
mask = nib.load(mask_path).get_fdata().astype(np.uint8)

print("Original image shape:", image.shape)
print("Original mask shape:", mask.shape)

# Downsample by 2
print("Downsampling...")

image = zoom(
    image,
    zoom=(0.5, 0.5, 0.5),
    order=1
)

mask = zoom(
    mask,
    zoom=(0.5, 0.5, 0.5),
    order=0
)

# Clip CT values
image = np.clip(image, -1000, 400)

# Normalize to [0, 1]
image = (image + 1000) / 1400

# Convert mask to binary kidney mask
mask = (mask == 1).astype(np.uint8)

print("Processed image shape:", image.shape)
print("Processed mask shape:", mask.shape)

print("Image range:", image.min(), image.max())
print("Mask labels:", np.unique(mask))
print("Kidney voxels:", (mask == 1).sum())

np.save(output_dir / "image.npy", image.astype(np.float32))
np.save(output_dir / "mask.npy", mask)

print("Saved to:", output_dir)