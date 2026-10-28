
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage

from mass_model_v2 import MASSStyleModelV2


# Paths: run this script from the DL project root.
DATA_DIR = Path("data/real")
PAIR_DIR = DATA_DIR / "reference_query_case02" / "pair_00"
CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"
OUTPUT_DIR = DATA_DIR / "project_figures"
OUTPUT_FILE = OUTPUT_DIR / "case02_pair00_final_prediction.png"

THRESHOLD = 0.60
SLICE_INDEX = 32
DEVICE = torch.device("cpu")


def load_array(filename):
    path = PAIR_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    arr = np.load(path, allow_pickle=False).astype(np.float32)

    if arr.shape == (1, 64, 64, 64):
        arr = arr[0]

    if arr.shape != (64, 64, 64):
        raise ValueError(
            f"{path} has shape {arr.shape}; expected (64,64,64)"
        )

    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path}")

    return arr


def dice_score(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    denominator = prediction.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return float(
        2 * np.logical_and(prediction, target).sum() / denominator
    )


def keep_largest_component(mask):
    # Use 26-neighbour connectivity in 3D.
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(mask, structure=structure)

    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest_label = int(sizes.argmax())
    return labels == largest_label


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT}"
        )

    query = load_array("query_image.npy")
    target = load_array("query_mask.npy") > 0.5
    reference = load_array("reference_image.npy")
    reference_mask = load_array("reference_mask.npy")

    if not np.isin(reference_mask, [0, 1]).all():
        raise ValueError("Reference mask must contain only 0 and 1.")

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

    def to_tensor(array):
        return (
            torch.from_numpy(array.copy())
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

    with torch.inference_mode():
        logits = model(
            to_tensor(reference),
            to_tensor(reference_mask.astype(np.float32)),
            to_tensor(query),
        )

        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    raw_prediction = probabilities >= THRESHOLD
    filtered_prediction = keep_largest_component(raw_prediction)

    raw_dice = dice_score(raw_prediction, target)
    filtered_dice = dice_score(filtered_prediction, target)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ct_slice = query[SLICE_INDEX]
    target_slice = target[SLICE_INDEX]
    raw_slice = raw_prediction[SLICE_INDEX]
    filtered_slice = filtered_prediction[SLICE_INDEX]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))

    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("Query CT")

    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(
        np.ma.masked_where(~target_slice, target_slice),
        cmap="Greens",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground-truth kidney mask")

    axes[2].imshow(ct_slice, cmap="gray")
    axes[2].imshow(
        np.ma.masked_where(~raw_slice, raw_slice),
        cmap="Reds",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"V2 raw prediction\nThreshold {THRESHOLD:.2f}"
        f"\n3D Dice: {raw_dice:.4f}"
    )

    axes[3].imshow(ct_slice, cmap="gray")
    axes[3].imshow(
        np.ma.masked_where(~filtered_slice, filtered_slice),
        cmap="Blues",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[3].set_title(
        f"Largest 3D component\nThreshold {THRESHOLD:.2f}"
        f"\n3D Dice: {filtered_dice:.4f}"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Held-out case00002 — Pair 00 — Slice {SLICE_INDEX}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Raw prediction Dice: {raw_dice:.4f}")
    print(f"Filtered prediction Dice: {filtered_dice:.4f}")
    print(f"Saved visualization: {OUTPUT_FILE}")
    print("Note: Dice values above are for this entire 3D patch.")
    print("The figure displays one slice from that patch.")


if __name__ == "__main__":
    main()
