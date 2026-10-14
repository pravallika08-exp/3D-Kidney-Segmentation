import numpy as np
from pathlib import Path

IMAGE_PATH = Path("data/processed/image.npy")
MASK_PATH = Path("data/processed/mask.npy")

print("=" * 50)
print("CHECKING PROCESSED DATA")
print("=" * 50)

image = np.load(IMAGE_PATH)
mask = np.load(MASK_PATH)

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image min:", image.min())
print("Image max:", image.max())

print()

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Mask values:", np.unique(mask))

print()

if image.shape == mask.shape:
    print("Image and mask shapes match: YES")
else:
    print("Image and mask shapes match: NO")

if image.min() >= 0 and image.max() <= 1:
    print("Image normalization: PASS")
else:
    print("Image normalization: FAIL")

if set(np.unique(mask)).issubset({0, 1}):
    print("Binary mask: PASS")
else:
    print("Binary mask: FAIL")

print("=" * 50)
print("PROCESSED DATA CHECK COMPLETE!")
print("=" * 50)