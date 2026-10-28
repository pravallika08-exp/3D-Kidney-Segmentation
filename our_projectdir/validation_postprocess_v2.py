
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import ndimage

from mass_model_v2 import MASSStyleModelV2


DATA_DIR = Path("data/real")
PAIR_DIR = DATA_DIR / "training_reference_query_v2"
CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"

OUTPUT_CSV = DATA_DIR / "validation_postprocess_v2.csv"
OUTPUT_TXT = DATA_DIR / "validation_postprocess_v2.txt"

SEED = 42
PREDICTION_THRESHOLD = 0.60
DEVICE = torch.device("cpu")

METRICS = ["dice", "iou", "precision", "recall", "specificity"]


class ValidationPairs(Dataset):
    def __init__(self, root):
        self.pairs = sorted(
            p for p in Path(root).glob("pair_*") if p.is_dir()
        )
        if not self.pairs:
            raise RuntimeError(f"No pair folders found in {root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        folder = self.pairs[index]
        names = [
            "reference_image.npy",
            "reference_mask.npy",
            "query_image.npy",
            "query_mask.npy",
        ]

        arrays = []
        for name in names:
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(path)

            arr = np.load(path, allow_pickle=False).astype(np.float32)

            if arr.shape != (64, 64, 64):
                raise ValueError(f"Unexpected shape in {path}: {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values in {path}")

            arrays.append(torch.from_numpy(arr.copy()).unsqueeze(0))

        return tuple(arrays)


def metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()

    return {
        "dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "iou": float(tp / max(tp + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def largest_component(mask):
    # 26-neighbour connectivity in 3D.
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(mask, structure=structure)

    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # Background is not a component.
    largest_label = int(counts.argmax())

    return labels == largest_label


def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    model = MASSStyleModelV2().to(DEVICE)

    try:
        state = torch.load(
            CHECKPOINT, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        state = torch.load(CHECKPOINT, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def main():
    dataset = ValidationPairs(PAIR_DIR)

    if len(dataset) < 10:
        raise RuntimeError(f"Expected at least 10 pairs, found {len(dataset)}")

    n_val = max(5, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val

    # Reproduce the training script's pair-level validation split.
    _, val_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    loader = DataLoader(
        val_subset, batch_size=1, shuffle=False, num_workers=0
    )

    model = load_model()
    rows = []

    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            ref_image, ref_mask, query_image, query_mask = batch

            logits = model(
                ref_image.to(DEVICE),
                ref_mask.to(DEVICE),
                query_image.to(DEVICE),
            )

            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
            target = query_mask[0, 0].numpy() > 0.5

            raw_prediction = probability >= PREDICTION_THRESHOLD
            filtered_prediction = largest_component(raw_prediction)

            for method, prediction in [
                ("raw", raw_prediction),
                ("largest_component", filtered_prediction),
            ]:
                result = metrics(prediction, target)
                rows.append({
                    "validation_index": idx,
                    "method": method,
                    "threshold": PREDICTION_THRESHOLD,
                    **result,
                    "predicted_voxels": int(prediction.sum()),
                    "target_voxels": int(target.sum()),
                })

            print(
                f"Validation item {idx:02d}: "
                f"raw Dice={rows[-2]['dice']:.4f}, "
                f"largest-component Dice={rows[-1]['dice']:.4f}"
            )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "V2 validation post-processing comparison",
        f"Validation pairs: {len(val_subset)}",
        f"Random split seed: {SEED}",
        f"Prediction threshold: {PREDICTION_THRESHOLD:.2f}",
        "",
        "Mean metrics across validation pairs:",
    ]

    for method in ["raw", "largest_component"]:
        method_rows = [r for r in rows if r["method"] == method]
        lines.append(f"\nMethod: {method}")

        for metric in METRICS:
            values = [r[metric] for r in method_rows]
            lines.append(f"Mean {metric}: {np.mean(values):.4f}")

    lines.extend([
        "",
        "Limitations:",
        "- Validation uses a pair-level split, not an independent patient-level split.",
        "- Keeping the largest component can remove true disconnected foreground regions.",
        "- Results must be checked before deciding whether to use post-processing.",
        "- The held-out case00002 test data was not used in this comparison.",
        "- The model checkpoint was not changed.",
    ])

    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()

