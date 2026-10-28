
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from mass_model_v2 import MASSStyleModelV2


# Run from the DL project root.
DATA_DIR = Path("data/real")
PAIR_DIR = DATA_DIR / "training_reference_query_v2"
CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"

OUTPUT_CSV = DATA_DIR / "validation_threshold_analysis_v2.csv"
OUTPUT_TXT = DATA_DIR / "validation_threshold_analysis_v2.txt"

SEED = 42
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
DEVICE = torch.device("cpu")

METRICS = ["dice", "iou", "precision", "recall", "specificity"]


class ValidationPairs(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.pairs = sorted(
            p for p in self.root.glob("pair_*") if p.is_dir()
        )

        if not self.pairs:
            raise RuntimeError(f"No pair folders found in {self.root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        folder = self.pairs[index]

        arrays = []
        filenames = [
            "reference_image.npy",
            "reference_mask.npy",
            "query_image.npy",
            "query_mask.npy",
        ]

        for filename in filenames:
            path = folder / filename

            if not path.is_file():
                raise FileNotFoundError(f"Missing file: {path}")

            arr = np.load(path, allow_pickle=False).astype(np.float32)

            if arr.shape != (64, 64, 64):
                raise ValueError(
                    f"{path} has shape {arr.shape}; expected (64,64,64)"
                )

            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values found in {path}")

            arrays.append(torch.from_numpy(arr.copy()).unsqueeze(0))

        return tuple(arrays)


def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = (prediction & target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    tn = (~prediction & ~target).sum().item()

    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    dataset = ValidationPairs(PAIR_DIR)

    if len(dataset) < 10:
        raise RuntimeError(
            f"Expected at least 10 pairs, found {len(dataset)}"
        )

    n_val = max(5, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val

    # Reproduce the training script's seed and pair-level split.
    _, validation_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    loader = DataLoader(
        validation_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = MASSStyleModelV2().to(DEVICE)

    try:
        state = torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(CHECKPOINT, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    # Run inference only once per validation pair.
    predictions = []

    with torch.inference_mode():
        for ref_image, ref_mask, query_image, query_mask in loader:
            logits = model(
                ref_image.to(DEVICE),
                ref_mask.to(DEVICE),
                query_image.to(DEVICE),
            )

            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
            target = query_mask[0, 0].numpy() > 0.5

            predictions.append((probabilities, target))

    results = []

    for threshold in THRESHOLDS:
        per_pair = []

        for probabilities, target in predictions:
            prediction = probabilities >= threshold
            per_pair.append(calculate_metrics(prediction, target))

        row = {
            "threshold": threshold,
            "validation_pairs": len(per_pair),
        }

        for metric in METRICS:
            values = [item[metric] for item in per_pair]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1 else 0.0
            )

        results.append(row)

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = max(results, key=lambda row: row["mean_dice"])

    lines = [
        "V2 validation threshold analysis",
        f"Training pair folder: {PAIR_DIR}",
        f"Total pairs: {len(dataset)}",
        f"Validation pairs: {len(validation_subset)}",
        f"Random split seed: {SEED}",
        "",
        "Threshold results (mean metrics across validation pairs):",
    ]

    for row in results:
        lines.append(
            f"threshold={row['threshold']:.2f}: "
            + ", ".join(
                f"{metric}={row[f'mean_{metric}']:.4f}"
                for metric in METRICS
            )
        )

    lines.extend([
        "",
        f"Highest validation mean Dice threshold: {best['threshold']:.2f}",
        f"Highest validation mean Dice: {best['mean_dice']:.4f}",
        "",
        "Important limitations:",
        "- This is a pair-level validation split, not patient-level validation.",
        "- The chosen threshold must be treated as a candidate, not proof of improvement.",
        "- Do not select a threshold using the held-out case00002 test results.",
        "- The model checkpoint was not retrained or modified.",
    ])

    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
