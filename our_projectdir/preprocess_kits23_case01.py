import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path

image_path = "data/real/kits23_case00001/imaging.nii.gz"
mask_path = "data/real/kits23_case00001/segmentation.nii.gz"

output_dir = Path("data/real/kits23_case00001/processed")
output_dir.mkdir(parents=True, exist_ok=True)

image = nib.load(image_path).get_fdata().astype(np.float32)
mask = nib.load(mask_path).get_fdata().astype(np.uint8)

print("Original image:", image.shape)
print("Original mask :", mask.shape)

# Downsample by 2
image_small = zoom(image, 0.5, order=1)
mask_small = zoom(mask, 0.5, order=0)

# Normalize CT
image_small = np.clip(image_small, -1000, 400)
image_small = (image_small + 1000) / 1400

np.save(output_dir / "image.npy", image_small)
np.save(output_dir / "mask.npy", mask_small)

print("Processed image:", image_small.shape)
print("Processed mask :", mask_small.shape)
print("Image range:", image_small.min(), image_small.max())
print("Mask labels:", np.unique(mask_small))
print("Saved to:", output_dir)