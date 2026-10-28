
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from mass_model_v2 import MASSStyleModelV2


DATA_DIR = Path("data/real")
PAIR_DIR = DATA_DIR / "reference_query_case02" / "pair_00"
CHECKPOINT = DATA_DIR / "best_mass_kits23_model_v2.pth"

THRESHOLD = 0.60
DEVICE = torch.device("cpu")


def load_array(filename):
    path = PAIR_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)

    array = np.load(path, allow_pickle=False).astype(np.float32)

    if array.shape == (1, 64, 64, 64):
        array = array[0]

    if array.shape != (64, 64, 64):
        raise ValueError(f"Unexpected shape for {path}: {array.shape}")

    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {path}")

    return array


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    reference_image = load_array("reference_image.npy")
    reference_mask = load_array("reference_mask.npy")
    query_image = load_array("query_image.npy")

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
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

    with torch.inference_mode():
        logits = model(
            to_tensor(reference_image),
            to_tensor(reference_mask),
            to_tensor(query_image),
        )
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    raw_mask = probabilities >= THRESHOLD

    # Same 26-neighbour 3D connectivity used by our filtering script.
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(raw_mask, structure=structure)

    print("3D connected-component analysis")
    print(f"Pair: case00002, pair_00")
    print(f"Prediction threshold: {THRESHOLD:.2f}")
    print(f"Predicted foreground voxels: {int(raw_mask.sum())}")
    print(f"Number of 3D connected components: {count}")

    if count == 0:
        print("No foreground components were predicted.")
        return

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    print("\nComponents, ordered largest to smallest:")
    sorted_labels = np.argsort(sizes)[::-1]

    for rank, label_id in enumerate(sorted_labels, start=1):
        size = int(sizes[label_id])
        if size == 0:
            continue

        coords = np.where(labels == label_id)

        z_min, z_max = int(coords[0].min()), int(coords[0].max())
        y_min, y_max = int(coords[1].min()), int(coords[1].max())
        x_min, x_max = int(coords[2].min()), int(coords[2].max())

        print(
            f"Component {label_id}: "
            f"voxels={size}, "
            f"z-range={z_min}-{z_max}, "
            f"y-range={y_min}-{y_max}, "
            f"x-range={x_min}-{x_max}"
        )

    display_slice = 32
    labels_in_slice = np.unique(labels[display_slice])
    foreground_labels = [int(v) for v in labels_in_slice if v != 0]

    print(f"\nComponents visible on slice {display_slice}:")
    if not foreground_labels:
        print("No foreground components on this slice.")
    else:
        for label_id in foreground_labels:
            slice_voxels = int(
                np.count_nonzero(labels[display_slice] == label_id)
            )
            print(
                f"Component {label_id}: "
                f"{slice_voxels} voxels on slice {display_slice}; "
                f"total 3D size={int(sizes[label_id])}"
            )

    largest_id = int(sizes.argmax())
    print(f"\nLargest component label: {largest_id}")
    print(
        "Does the largest component occupy slice 32? ",
        bool(np.any(labels[display_slice] == largest_id)),
    )

    print(
        "\nInterpretation: if the kidney and extra region have the same "
        "component label, they connect somewhere in 3D. If they have "
        "different labels, the extra region is a separate component."
    )


if __name__ == "__main__":
    main()
