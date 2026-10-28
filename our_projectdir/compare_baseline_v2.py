
import csv
from pathlib import Path

import torch

from real_mass_dataset import RealMASSDataset
from mass_model import MASSStyleModel
from mass_model_v2 import MASSStyleModelV2


DATA_DIR = Path("data/real")
TEST_DIR = DATA_DIR / "reference_query_case02"

BASELINE_CHECKPOINT = DATA_DIR / "mass_kits23_model.pth"
V2_CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"

OUTPUT_CSV = DATA_DIR / "baseline_v2_comparison_case02.csv"
OUTPUT_TXT = DATA_DIR / "baseline_v2_comparison_case02.txt"

THRESHOLD = 0.5
DEVICE = torch.device("cpu")

METRICS = ["dice", "iou", "precision", "recall", "specificity"]


def calculate_metrics(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

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


def load_model(model, checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        state = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def evaluate(model, dataset, model_name):
    rows = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            ref_image, ref_mask, query_image, query_mask = dataset[index]

            ref_image = ref_image.unsqueeze(0).to(DEVICE)
            ref_mask = ref_mask.unsqueeze(0).to(DEVICE)
            query_image = query_image.unsqueeze(0).to(DEVICE)

            target = query_mask.unsqueeze(0).to(DEVICE) > 0.5

            logits = model(ref_image, ref_mask, query_image)
            prediction = torch.sigmoid(logits) >= THRESHOLD

            metrics = calculate_metrics(prediction, target)

            rows.append({"pair": f"pair_{index:02d}", **metrics})

            print(
                f"{model_name} pair_{index:02d}: "
                + " ".join(
                    f"{name}={metrics[name]:.4f}"
                    for name in METRICS
                )
            )

    return rows


def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    dataset = RealMASSDataset(TEST_DIR)

    if len(dataset) == 0:
        raise RuntimeError("No test pairs were found.")

    baseline = load_model(
        MASSStyleModel(),
        BASELINE_CHECKPOINT,
    )

    v2 = load_model(
        MASSStyleModelV2(),
        V2_CHECKPOINT,
    )

    print(f"Test patches: {len(dataset)}")
    print(f"Prediction threshold: {THRESHOLD}\n")

    baseline_rows = evaluate(baseline, dataset, "BASELINE")
    print()
    v2_rows = evaluate(v2, dataset, "V2")

    results = []

    for base, improved in zip(baseline_rows, v2_rows):
        row = {"pair": base["pair"]}

        for metric in METRICS:
            row[f"baseline_{metric}"] = base[metric]
            row[f"v2_{metric}"] = improved[metric]
            row[f"delta_{metric}"] = improved[metric] - base[metric]

        results.append(row)

    columns = ["pair"]
    for metric in METRICS:
        columns.extend([
            f"baseline_{metric}",
            f"v2_{metric}",
            f"delta_{metric}",
        ])

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "Baseline versus V2: held-out case00002",
        f"Number of test patches: {len(results)}",
        f"Prediction threshold: {THRESHOLD}",
        "",
        "Mean metrics across patches:",
    ]

    for metric in METRICS:
        base_mean = sum(r[f"baseline_{metric}"] for r in results) / len(results)
        v2_mean = sum(r[f"v2_{metric}"] for r in results) / len(results)
        delta = v2_mean - base_mean

        lines.append(
            f"{metric}: baseline={base_mean:.4f}, "
            f"V2={v2_mean:.4f}, change={delta:+.4f}"
        )

    lines.extend([
        "",
        "Limitations:",
        "- All test patches are from one held-out patient.",
        "- Patch-level results do not establish patient-level generalization.",
        "- This is a comparison of our two models, not the official MASS model.",
    ])

    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
