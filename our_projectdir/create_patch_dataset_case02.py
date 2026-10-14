from pathlib import Path
import numpy as np


input_dir = Path("data/real/kits23_case00002/processed")
output_dir = Path("data/real/patch_dataset/case00002")

output_dir.mkdir(parents=True, exist_ok=True)

image = np.load(input_dir / "image.npy")
mask = np.load(input_dir / "mask.npy")

patch_size = 64

kidney_coords = np.argwhere(mask == 1)

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Kidney voxels:", len(kidney_coords))


# Pick 10 evenly distributed kidney locations
indices = np.linspace(
    0,
    len(kidney_coords) - 1,
    10,
    dtype=int
)


def get_start(center, size, max_size):
    start = center - size // 2
    start = max(0, start)
    start = min(start, max_size - size)
    return start


for patch_id, index in enumerate(indices):

    z, y, x = kidney_coords[index]

    z0 = get_start(z, patch_size, image.shape[0])
    y0 = get_start(y, patch_size, image.shape[1])
    x0 = get_start(x, patch_size, image.shape[2])

    image_patch = image[
        z0:z0 + patch_size,
        y0:y0 + patch_size,
        x0:x0 + patch_size
    ]

    mask_patch = mask[
        z0:z0 + patch_size,
        y0:y0 + patch_size,
        x0:x0 + patch_size
    ]

    np.save(
        output_dir / f"image_{patch_id:02d}.npy",
        image_patch.astype(np.float32)
    )

    np.save(
        output_dir / f"mask_{patch_id:02d}.npy",
        mask_patch.astype(np.uint8)
    )

    print(
        f"Patch {patch_id:02d}:",
        image_patch.shape,
        "kidney voxels:",
        int(mask_patch.sum())
    )

print("Saved patches to:", output_dir)