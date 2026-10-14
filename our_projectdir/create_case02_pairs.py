from pathlib import Path
import numpy as np


reference_dir = Path("data/real/patch_dataset/case00000")
query_dir = Path("data/real/patch_dataset/case00002")

output_dir = Path("data/real/reference_query_case02")
output_dir.mkdir(parents=True, exist_ok=True)


for i in range(10):

    pair_dir = output_dir / f"pair_{i:02d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    reference_image = np.load(
        reference_dir / f"image_{i:02d}.npy"
    )

    reference_mask = np.load(
        reference_dir / f"mask_{i:02d}.npy"
    )

    query_image = np.load(
        query_dir / f"image_{i:02d}.npy"
    )

    query_mask = np.load(
        query_dir / f"mask_{i:02d}.npy"
    )

    np.save(pair_dir / "reference_image.npy", reference_image)
    np.save(pair_dir / "reference_mask.npy", reference_mask)
    np.save(pair_dir / "query_image.npy", query_image)
    np.save(pair_dir / "query_mask.npy", query_mask)

    print(
        f"Pair {i:02d}:",
        "reference", reference_image.shape,
        "query", query_image.shape
    )

print("Saved Case 00002 pairs to:", output_dir)