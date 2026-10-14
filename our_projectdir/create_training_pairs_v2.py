from pathlib import Path
import numpy as np

root = Path("data/real")
patch_root = root / "patch_dataset"
output = root / "training_reference_query_v2"
cases = ["case00000", "case00001"]

if output.exists() and any(output.iterdir()):
    raise FileExistsError(f"{output} is not empty; refusing to overwrite it.")

def load_patch(case, i):
    folder = patch_root / case
    image = np.load(folder / f"image_{i:02d}.npy").astype(np.float32)
    mask = np.load(folder / f"mask_{i:02d}.npy").astype(np.float32)
    if image.shape != mask.shape or image.shape != (64, 64, 64):
        raise ValueError(f"Unexpected patch shape: {case}, {i}")
    return image, mask

def save_pair(pair_id, rc, ri, qc, qi):
    if rc == qc and ri == qi:
        raise ValueError("Identical reference/query patch rejected.")
    ref_img, ref_mask = load_patch(rc, ri)
    qry_img, qry_mask = load_patch(qc, qi)
    folder = output / f"pair_{pair_id:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    for name, array in [
        ("reference_image", ref_img),
        ("reference_mask", ref_mask),
        ("query_image", qry_img),
        ("query_mask", qry_mask),
    ]:
        np.save(folder / f"{name}.npy", array)

output.mkdir(parents=True, exist_ok=True)
pair_id = 0

# Cross-patient pairs in both directions
for i in range(10):
    save_pair(pair_id, cases[0], i, cases[1], i)
    pair_id += 1
    save_pair(pair_id, cases[1], i, cases[0], i)
    pair_id += 1

# Distinct intra-patient pairs
for case in cases:
    for i in range(15):
        ref_i = i % 10
        query_i = (ref_i + 1 + i // 10) % 10
        save_pair(pair_id, case, ref_i, case, query_i)
        pair_id += 1

print(f"Created {pair_id} distinct pairs in {output}")
print("Note: distinct patches are not necessarily anatomically aligned.")
