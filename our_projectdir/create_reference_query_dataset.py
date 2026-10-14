import numpy as np
from pathlib import Path

root = Path("data/real/patch_dataset")

reference_dir = root / "case00000"
query_dir = root / "case00001"

output_dir = Path("data/real/reference_query_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

num_pairs = 10

for i in range(num_pairs):

    # Reference patient: Case 00000
    reference_image = np.load(
        reference_dir / f"image_{i:02d}.npy"
    )

    reference_mask = np.load(
        reference_dir / f"mask_{i:02d}.npy"
    )

    # Query patient: Case 00001
    query_image = np.load(
        query_dir / f"image_{i:02d}.npy"
    )

    query_mask = np.load(
        query_dir / f"mask_{i:02d}.npy"
    )

    pair_dir = output_dir / f"pair_{i:02d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    np.save(pair_dir / "reference_image.npy", reference_image)
    np.save(pair_dir / "reference_mask.npy", reference_mask)
    np.save(pair_dir / "query_image.npy", query_image)
    np.save(pair_dir / "query_mask.npy", query_mask)

    print(
        f"Pair {i:02d}: "
        f"reference={reference_image.shape}, "
        f"query={query_image.shape}"
    )

print("\nREFERENCE/QUERY DATASET CREATED")
print("Pairs:", num_pairs)
print("Saved to:", output_dir)