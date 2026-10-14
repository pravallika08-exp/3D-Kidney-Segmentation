from pathlib import Path
import numpy as np


output_dir = Path("data/real/training_reference_query")
output_dir.mkdir(parents=True, exist_ok=True)

patients = [
    "case00000",
    "case00001",
]

pair_id = 0

for patient in patients:

    patch_dir = Path(
        f"data/real/patch_dataset/{patient}"
    )

    for i in range(10):

        image_path = patch_dir / f"image_{i:02d}.npy"
        mask_path = patch_dir / f"mask_{i:02d}.npy"

        image = np.load(image_path)
        mask = np.load(mask_path)

        pair_dir = output_dir / f"pair_{pair_id:02d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # For this prototype, each patch acts as
        # both the reference source and query source.
        np.save(
            pair_dir / "reference_image.npy",
            image
        )

        np.save(
            pair_dir / "reference_mask.npy",
            mask
        )

        np.save(
            pair_dir / "query_image.npy",
            image
        )

        np.save(
            pair_dir / "query_mask.npy",
            mask
        )

        print(
            f"Pair {pair_id:02d} <- {patient} patch {i:02d}"
        )

        pair_id += 1


print()
print("Total training pairs:", pair_id)
print("Saved to:", output_dir)