
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import ndimage

from mass_model_v2 import MASSStyleModelV2


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/real")
PAIR_DIR = DATA_DIR / "training_reference_query_v2"
CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"

OUTPUT_CSV = DATA_DIR / "validation_postprocess_6conn_v2.csv"
OUTPUT_TXT = DATA_DIR / "validation_postprocess_6conn_v2.txt"

SEED = 42
PREDICTION_THRESHOLD = 0.60
DEVICE = torch.device("cpu")

METRICS = ["dice", "iou", "precision", "recall", "specificity"]


# --------------------------------------------------
# Load validation pairs
# --------------------------------------------------

class ValidationPairs(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.pairs = sorted(
            p for p in root.glob("pair_*") if p.is_dir()
        )

        if not self.pairs:
            raise RuntimeError(f"No pair folders found in {root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        folder = self.pairs[index]

        filenames = [
            "reference_image.npy",
            "reference_mask.npy",
            "query_image.npy",
            "query_mask.npy",
        ]

        arrays = []

        for filename in filenames:
            path = folder / filename

            if not path.is_file():
                raise FileNotFoundError(f"Missing file: {path}")

            array = np.load(
                path, allow_pickle=False
            ).astype(np.float32)

            if array.shape != (64, 64, 64):
                raise ValueError(
                    f"Unexpected shape in {path}: {array.shape}"
                )

            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite values found in {path}")

            arrays.append(
                torch.from_numpy(array.copy()).unsqueeze(0)
            )

        return tuple(arrays)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(prediction, target):
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


# --------------------------------------------------
# Keep the largest component using 6-neighbour
# connectivity: adjacent voxels connect through faces,
# not only through edges or corners.
# --------------------------------------------------

def largest_component_6conn(mask):
    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )

    labels, count = ndimage.label(
        mask.astype(bool),
        structure=structure,
    )

    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest_label = int(sizes.argmax())

    return labels == largest_label


# --------------------------------------------------
# Load existing V2 checkpoint
# --------------------------------------------------

def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model = MASSStyleModelV2().to(DEVICE)

    try:
        state = torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            CHECKPOINT,
            map_location=DEVICE,
        )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model


# --------------------------------------------------
# Validation experiment
# --------------------------------------------------

def main():
    if not PAIR_DIR.is_dir():
        raise FileNotFoundError(
            f"Training pair directory not found: {PAIR_DIR}"
        )

    dataset = ValidationPairs(PAIR_DIR)

    if len(dataset) < 10:
        raise RuntimeError(
            f"Expected at least 10 pairs, found {len(dataset)}"
        )

    n_val = max(5, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val

    # Reproduce the original training script's validation split.
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

    model = load_model()
    rows = []

    print("V2 validation: 6-neighbour post-processing")
    print(f"Total pairs: {len(dataset)}")
    print(f"Validation pairs: {len(validation_subset)}")
    print(f"Split seed: {SEED}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD:.2f}")
    print("Connectivity: 6-neighbour in 3D\n")

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            ref_image, ref_mask, query_image, query_mask = batch

            logits = model(
                ref_image.to(DEVICE),
                ref_mask.to(DEVICE),
                query_image.to(DEVICE),
            )

            probabilities = (
                torch.sigmoid(logits)[0, 0].cpu().numpy()
            )
            target = query_mask[0, 0].numpy() > 0.5

            raw_prediction = (
                probabilities >= PREDICTION_THRESHOLD
            )

            filtered_prediction = largest_component_6conn(
                raw_prediction
            )

            raw_scores = calculate_metrics(
                raw_prediction,
                target,
            )

            filtered_scores = calculate_metrics(
                filtered_prediction,
                target,
            )

            for method, scores in [
                ("raw", raw_scores),
                ("largest_component_6conn", filtered_scores),
            ]:
                rows.append({
                    "validation_index": index,
                    "method": method,
                    "threshold": PREDICTION_THRESHOLD,
                    **scores,
                    "predicted_voxels": int(
                        np.asarray(
                            raw_prediction
                            if method == "raw"
                            else filtered_prediction
                        ).sum()
                    ),
                    "target_voxels": int(target.sum()),
                })

            print(
                f"Validation item {index:02d}: "
                f"raw Dice={raw_scores['dice']:.4f}, "
                f"6-connected Dice={filtered_scores['dice']:.4f}"
            )

    # Save individual validation-pair measurements.
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # Summarize each method.
    lines = [
        "V2 validation: 6-neighbour post-processing",
        f"Total pairs: {len(dataset)}",
        f"Validation pairs: {len(validation_subset)}",
        f"Random split seed: {SEED}",
        f"Prediction threshold: {PREDICTION_THRESHOLD:.2f}",
        "",
        "Mean metrics across validation pairs:",
    ]

    summaries = {}

    for method in ["raw", "largest_component_6conn"]:
        method_rows = [
            row for row in rows if row["method"] == method
        ]

        summaries[method] = {}

        lines.append(f"\nMethod: {method}")

        for metric in METRICS:
            mean_value = float(
                np.mean([row[metric] for row in method_rows])
            )
            summaries[method][metric] = mean_value
            lines.append(f"Mean {metric}: {mean_value:.4f}")

    lines.append("\nChange (6-connected minus raw):")

    for metric in METRICS:
        delta = (
            summaries["largest_component_6conn"][metric]
            - summaries["raw"][metric]
        )
        lines.append(f"{metric}: {delta:+.4f}")

    lines.extend([
        "",
        "Limitations:",
        "- This is a pair-level validation split, not patient-level validation.",
        "- Strict connectivity may split a kidney mask into separate components.",
        "- Validation improvement alone does not prove better generalization.",
        "- Held-out case00002 was not used in this experiment.",
        "- The existing model checkpoint was not modified.",
    ])

    OUTPUT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(lines))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
