import numpy as np
from pathlib import Path

PATCH = 64

def get_center_with_label(image, mask, label=1):
    """Get a 64³ patch centered around a labeled voxel."""

    coords = np.argwhere(mask == label)

    if len(coords) == 0:
        raise ValueError(f"No label {label} found.")

    center = coords[len(coords) // 2]

    z, y, x = center

    z0 = max(0, min(z - PATCH // 2, image.shape[0] - PATCH))
    y0 = max(0, min(y - PATCH // 2, image.shape[1] - PATCH))
    x0 = max(0, min(x - PATCH // 2, image.shape[2] - PATCH))

    image_patch = image[
        z0:z0 + PATCH,
        y0:y0 + PATCH,
        x0:x0 + PATCH
    ]

    mask_patch = mask[
        z0:z0 + PATCH,
        y0:y0 + PATCH,
        x0:x0 + PATCH
    ]

    return image_patch, mask_patch


root = Path("data/real")

# Reference: case 00000
ref_image = np.load(
    root / "kits23_case00000/processed/image.npy"
)
ref_mask = np.load(
    root / "kits23_case00000/processed/mask.npy"
)

# Query: case 00001
query_image = np.load(
    root / "kits23_case00001/processed/image.npy"
)
query_mask = np.load(
    root / "kits23_case00001/processed/mask.npy"
)

# Kidney = label 1
ref_img_patch, ref_mask_patch = get_center_with_label(
    ref_image, ref_mask, label=1
)

query_img_patch, query_mask_patch = get_center_with_label(
    query_image, query_mask, label=1
)

output = root / "real_mass_pairs"
output.mkdir(parents=True, exist_ok=True)

np.save(output / "reference_image.npy", ref_img_patch)
np.save(output / "reference_mask.npy", (ref_mask_patch == 1).astype(np.uint8))
np.save(output / "query_image.npy", query_img_patch)
np.save(output / "query_mask.npy", (query_mask_patch == 1).astype(np.uint8))

print("=== REAL MASS PAIR ===")
print("Reference image:", ref_img_patch.shape)
print("Reference mask :", ref_mask_patch.shape)
print("Query image    :", query_img_patch.shape)
print("Query mask     :", query_mask_patch.shape)

print("Reference kidney voxels:",
      (ref_mask_patch == 1).sum())

print("Query kidney voxels:",
      (query_mask_patch == 1).sum())

print("Saved to:", output)