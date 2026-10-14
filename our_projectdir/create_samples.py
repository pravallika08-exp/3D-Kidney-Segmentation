import numpy as np
from pathlib import Path


IMAGE_PATH = Path("data/processed/image.npy")
MASK_PATH = Path("data/processed/mask.npy")

OUTPUT_DIR = Path("data/processed/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 50)
print("CREATING TRAINING SAMPLES")
print("=" * 50)


image = np.load(IMAGE_PATH)
mask = np.load(MASK_PATH)

print("Original shape:", image.shape)


# -----------------------------
# Sample 1 - original
# -----------------------------

np.save(
    OUTPUT_DIR / "image_0.npy",
    image
)

np.save(
    OUTPUT_DIR / "mask_0.npy",
    mask
)


# -----------------------------
# Sample 2 - flip depth
# -----------------------------

image_1 = np.flip(image, axis=0).copy()
mask_1 = np.flip(mask, axis=0).copy()

np.save(
    OUTPUT_DIR / "image_1.npy",
    image_1
)

np.save(
    OUTPUT_DIR / "mask_1.npy",
    mask_1
)


# -----------------------------
# Sample 3 - flip height
# -----------------------------

image_2 = np.flip(image, axis=1).copy()
mask_2 = np.flip(mask, axis=1).copy()

np.save(
    OUTPUT_DIR / "image_2.npy",
    image_2
)

np.save(
    OUTPUT_DIR / "mask_2.npy",
    mask_2
)


print()
print("Created 3 samples.")

for i in range(3):

    sample_image = np.load(
        OUTPUT_DIR / f"image_{i}.npy"
    )

    sample_mask = np.load(
        OUTPUT_DIR / f"mask_{i}.npy"
    )

    print(
        f"Sample {i}:",
        sample_image.shape,
        sample_mask.shape
    )


print("=" * 50)
print("SAMPLE CREATION COMPLETE!")
print("=" * 50)