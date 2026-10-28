
from pathlib import Path
import csv

import numpy as np
import torch
from scipy import ndimage

from real_mass_dataset import RealMASSDataset
from mass_model_v2 import MASSStyleModelV2


DATA_DIR = Path("data/real")
TEST_DIR = DATA_DIR / "reference_query_case02"
CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"

OUTPUT_CSV = DATA_DIR / "case02_v2_postprocess_comparison.csv"
OUTPUT_TXT = DATA_DIR / "case02_v2_postprocess_comparison.txt"

THRESHOLD = 0.60
DEVICE = torch.device("cpu")

METRICS = ["dice", "iou", "precision", "recall", "specificity"]


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


def largest_component(mask):
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(mask, structure=structure)

    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label_id = int(sizes.argmax())

    return labels == label_id


def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

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
    return model


def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    dataset = RealMASSDataset(TEST_DIR)
    if len(dataset) == 0:
        raise RuntimeError("No test pairs found.")

    model = load_model()
    rows = []

    print("Held-out case00002: V2 post-processing comparison")
    print(f"Test pairs: {len(dataset)}")
    print(f"Prediction threshold: {THRESHOLD:.2f}\n")

    with torch.inference_mode():
        for index in range(len(dataset)):
            ref_image, ref_mask, query_image, query_mask = dataset[index]

            ref_image = ref_image.unsqueeze(0).to(DEVICE)
            ref_mask = ref_mask.unsqueeze(0).to(DEVICE)
            query_image = query_image.unsqueeze(0).to(DEVICE)

            target = query_mask.squeeze(0).numpy() > 0.5

            logits = model(ref_image, ref_mask, query_image)
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

            raw_prediction = probabilities >= THRESHOLD
            filtered_prediction = largest_component(raw_prediction)

            raw_metrics = calculate_metrics(raw_prediction, target)
            filtered_metrics = calculate_metrics(filtered_prediction, target)

            for method, values in [
                ("raw", raw_metrics),
                ("largest_component", filtered_metrics),
            ]:
                rows.append({
                    "pair": f"pair_{index:02d}",
                    "method": method,
                    "threshold": THRESHOLD,
                    **values,
                })

            print(
                f"pair_{index:02d}: "
                f"raw Dice={raw_metrics['dice']:.4f}, "
                f"filtered Dice={filtered_metrics['dice']:.4f}"
            )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["pair", "method", "threshold"] + METRICS
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "V2 post-processing comparison on held-out case00002",
        f"Number of test pairs: {len(dataset)}",
        f"Prediction threshold: {THRESHOLD:.2f}",
        "",
        "Mean metrics across test patches:",
    ]

    summaries = {}

    for method in ["raw", "largest_component"]:
        method_rows = [row for row in rows if row["method"] == method]
        summaries[method] = {}

        lines.append(f"\nMethod: {method}")

        for metric in METRICS:
            mean_value = float(np.mean([row[metric] for row in method_rows]))
            summaries[method][metric] = mean_value
            lines.append(f"Mean {metric}: {mean_value:.4f}")

    lines.append("\nChange from post-processing (filtered minus raw):")

    for metric in METRICS:
        delta = (
            summaries["largest_component"][metric]
            - summaries["raw"][metric]
        )
        lines.append(f"{metric}: {delta:+.4f}")

    lines.extend([
        "",
        "Limitations:",
        "- All test patches are from one held-out patient.",
        "- The test set was not used to select the threshold or filter.",
        "- Results are preliminary patch-level measurements.",
        "- The model checkpoint was not modified.",
    ])

    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
