from pathlib import Path
import nibabel as nib
import numpy as np

folder = Path("data/real/kits23_case00002")

image_path = folder / "imaging.nii.gz"
mask_path = folder / "segmentation.nii.gz"

image = nib.load(image_path)
mask = nib.load(mask_path)

image_data = image.get_fdata()
mask_data = mask.get_fdata()

print("Image shape:", image_data.shape)
print("Mask shape:", mask_data.shape)

print("Image spacing:", image.header.get_zooms())
print("Mask spacing:", mask.header.get_zooms())

print("Image min:", image_data.min())
print("Image max:", image_data.max())

print("Mask labels:", np.unique(mask_data))

print("Shapes match:", image_data.shape == mask_data.shape)

print("Kidney voxels:", (mask_data == 1).sum())
print("Tumor voxels:", (mask_data == 2).sum())