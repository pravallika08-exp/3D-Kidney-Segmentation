from pathlib import Path
import csv
import torch

from real_mass_dataset import RealMASSDataset
from mass_model_v2 import MASSStyleModelV2

DATA_DIR = Path("data/real")
TEST_DIR = DATA_DIR / "reference_query_case02"
CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"
RESULTS_CSV = DATA_DIR / "case02_evaluation_v2.csv"
RESULTS_TXT = DATA_DIR / "case02_evaluation_v2.txt"

def calculate_metrics(pred, target):
    pred = pred.bool()
    target = target.bool()

    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()

    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
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

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

    dataset = RealMASSDataset(TEST_DIR)
    if len(dataset) == 0:
        raise RuntimeError("No test pairs found.")

    model = MASSStyleModelV2()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rows = []
    metric_names = ["dice", "iou", "precision", "recall", "specificity"]

    with torch.no_grad():
        for i in range(len(dataset)):
            reference_image, reference_mask, query_image, query_mask = dataset[i]

            logits = model(
                reference_image.unsqueeze(0),
                reference_mask.unsqueeze(0),
                query_image.unsqueeze(0),
            )
            prediction = torch.sigmoid(logits) > 0.5
            target = query_mask.unsqueeze(0) > 0.5

            metrics = calculate_metrics(prediction, target)
            rows.append({"pair": i, **metrics})

            print(
                f"Pair {i:02d}: "
                + " ".join(f"{key}={metrics[key]:.4f}" for key in metric_names)
            )

    averages = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in metric_names
    }

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"pair": "MEAN", **averages})

    lines = [
        "V2 final evaluation on held-out case00002",
        f"Number of test pairs: {len(rows)}",
        "Prediction threshold: 0.5",
        *[f"Mean {key}: {value:.4f}" for key, value in averages.items()],
        "",
        "Note: pairs from one held-out patient are not independent patients.",
    ]
    RESULTS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== V2 HELD-OUT TEST RESULTS ===")
    for key, value in averages.items():
        print(f"Mean {key}: {value:.4f}")
    print(f"\nSaved: {RESULTS_CSV}")
    print(f"Saved: {RESULTS_TXT}")

if __name__ == "__main__":
    main()
