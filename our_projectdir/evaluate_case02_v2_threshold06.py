from pathlib import Path
import csv
import torch

from real_mass_dataset import RealMASSDataset
from mass_model_v2 import MASSStyleModelV2


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/real")
TEST_DIR = DATA_DIR / "reference_query_case02"

CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"

# Separate output files preserve the original evaluation.
RESULTS_CSV = DATA_DIR / "case02_evaluation_v2_threshold06.csv"
RESULTS_TXT = DATA_DIR / "case02_evaluation_v2_threshold06.txt"

THRESHOLD = 0.60
DEVICE = torch.device("cpu")

METRIC_NAMES = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
]


# --------------------------------------------------
# Calculate segmentation metrics
# --------------------------------------------------

def calculate_metrics(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    tp = (prediction & target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    tn = (~prediction & ~target).sum().item()

    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


# --------------------------------------------------
# Load the trained V2 model
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
# Evaluate the held-out test patches
# --------------------------------------------------

def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DIR}"
        )

    dataset = RealMASSDataset(TEST_DIR)

    if len(dataset) == 0:
        raise RuntimeError("No test pairs were found.")

    model = load_model()

    print("V2 evaluation on held-out case00002")
    print(f"Number of test pairs: {len(dataset)}")
    print(f"Prediction threshold: {THRESHOLD:.2f}")
    print()

    rows = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            (
                reference_image,
                reference_mask,
                query_image,
                query_mask,
            ) = dataset[index]

            reference_image = reference_image.unsqueeze(0).to(DEVICE)
            reference_mask = reference_mask.unsqueeze(0).to(DEVICE)
            query_image = query_image.unsqueeze(0).to(DEVICE)

            # Keep the ground-truth threshold unchanged.
            target = query_mask.unsqueeze(0).to(DEVICE) > 0.5

            logits = model(
                reference_image,
                reference_mask,
                query_image,
            )

            # Only the prediction threshold changes to 0.60.
            prediction = torch.sigmoid(logits) >= THRESHOLD

            metrics = calculate_metrics(prediction, target)

            row = {
                "pair": f"pair_{index:02d}",
                **metrics,
            }
            rows.append(row)

            print(
                f"{row['pair']}: "
                + " ".join(
                    f"{name}={metrics[name]:.4f}"
                    for name in METRIC_NAMES
                )
            )

    # Calculate average metrics across test patches.
    averages = {
        name: sum(row[name] for row in rows) / len(rows)
        for name in METRIC_NAMES
    }

    # Save per-pair metrics and the mean row.
    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["pair"] + METRIC_NAMES,
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "pair": "MEAN",
            **averages,
        })

    lines = [
        "V2 evaluation on held-out case00002",
        f"Number of test pairs: {len(rows)}",
        f"Prediction threshold: {THRESHOLD:.2f}",
        "",
    ]

    for name, value in averages.items():
        lines.append(f"Mean {name}: {value:.4f}")

    lines.extend([
        "",
        "Important limitations:",
        "- All test patches are from one held-out patient.",
        "- These are patch-level results, not patient-level generalization results.",
        "- The model checkpoint was not retrained or modified.",
        "- Threshold 0.60 was selected using validation data.",
    ])

    RESULTS_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n=== V2 TEST RESULTS AT THRESHOLD 0.60 ===")

    for name, value in averages.items():
        print(f"Mean {name}: {value:.4f}")

    print(f"\nSaved CSV: {RESULTS_CSV}")
    print(f"Saved summary: {RESULTS_TXT}")


if __name__ == "__main__":
    main()