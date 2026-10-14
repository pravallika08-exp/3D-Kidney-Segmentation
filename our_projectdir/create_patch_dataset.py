import numpy as np
from pathlib import Path

PATCH = 64
NUM_PATCHES = 10


def extract_patch(image, mask, center):
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


def create_patches(case_name, output_name):
    root = Path("data/real")

    image = np.load(
        root / case_name / "processed/image.npy"
    )

    mask = np.load(
        root / case_name / "processed/mask.npy"
    )

    # Kidney voxels
    coords = np.argwhere(mask == 1)

    if len(coords) < NUM_PATCHES:
        raise ValueError("Not enough kidney voxels.")

    # Select evenly spaced kidney locations
    indices = np.linspace(
        0,
        len(coords) - 1,
        NUM_PATCHES,
        dtype=int
    )

    output = root / "patch_dataset" / output_name
    output.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):

        center = coords[idx]

        image_patch, mask_patch = extract_patch(
            image,
            mask,
            center
        )

        # Convert to binary kidney mask
        kidney_mask = (mask_patch == 1).astype(np.uint8)

        np.save(
            output / f"image_{i:02d}.npy",
            image_patch.astype(np.float32)
        )

        np.save(
            output / f"mask_{i:02d}.npy",
            kidney_mask
        )

    print(f"{case_name}: created {NUM_PATCHES} patches")
    print("Output:", output)


# Patient-level split
create_patches(
    "kits23_case00000",
    "case00000"
)

create_patches(
    "kits23_case00001",
    "case00001"
)

print("\nPATCH DATASET CREATION COMPLETE")