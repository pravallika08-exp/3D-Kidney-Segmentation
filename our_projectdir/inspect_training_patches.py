from pathlib import Path
import numpy as np

root = Path("data/real/patch_dataset")

for patient in ("case00000", "case00001"):
    folder = root / patient
    print(f"\n{patient}")

    for i in (0, 1, 5, 9):
        image = np.load(folder / f"image_{i:02d}.npy")
        mask = np.load(folder / f"mask_{i:02d}.npy")

        print(
            f"Patch {i:02d}: shape={image.shape}, "
            f"image_range=({image.min():.3f}, {image.max():.3f}), "
            f"kidney_voxels={int((mask > 0.5).sum())}"
        )
