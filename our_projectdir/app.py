
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from scipy import ndimage

# Make the model module importable locally and in deployment.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mass_model_v2 import MASSStyleModelV2


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="3D Kidney Segmentation",
    layout="wide",
)

st.title("MASS-Style 3D Kidney Segmentation")

st.warning(
    "Research prototype only — not for clinical use. "
    "This is not an official reproduction of MASS. "
    "Upload preprocessed 64x64x64 NumPy (.npy) patches only."
)

CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "best_mass_kits23_model_v2.pth"
)

DEVICE = torch.device("cpu")
PREDICTION_THRESHOLD = 0.60


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"V2 checkpoint not found: {CHECKPOINT}"
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
# Input validation
# --------------------------------------------------

def read_patch(upload, label, binary=False):
    if upload is None:
        return None

    array = np.load(upload, allow_pickle=False)
    array = np.asarray(array)

    if array.shape == (1, 64, 64, 64):
        array = array[0]

    if array.shape != (64, 64, 64):
        raise ValueError(
            f"{label} must have shape (64,64,64); "
            f"received {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{label} contains NaN or infinite values."
        )

    if binary and not np.isin(array, [0, 1]).all():
        raise ValueError(
            f"{label} must contain only binary values 0 and 1."
        )

    return array.astype(np.float32)


def to_tensor(array):
    return (
        torch.from_numpy(array.copy())
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )


# --------------------------------------------------
# Final 3D post-processing
# --------------------------------------------------

def keep_largest_component(mask):
    # Matches the 26-neighbour connectivity tested previously.
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(
        np.asarray(mask, dtype=bool),
        structure=structure,
    )

    if count == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest_label = int(sizes.argmax())
    return (labels == largest_label).astype(np.uint8)


def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()

    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)

    return float(dice), float(iou)


# --------------------------------------------------
# Upload interface
# --------------------------------------------------

st.subheader("Upload preprocessed 3D patches")

col1, col2 = st.columns(2)

with col1:
    reference_file = st.file_uploader(
        "1. Reference CT patch (.npy)",
        type=["npy"],
        key="reference_ct",
    )

    reference_mask_file = st.file_uploader(
        "2. Reference kidney mask (.npy)",
        type=["npy"],
        key="reference_mask",
    )

with col2:
    query_file = st.file_uploader(
        "3. Query CT patch (.npy)",
        type=["npy"],
        key="query_ct",
    )

    target_file = st.file_uploader(
        "Optional: query ground-truth mask (.npy)",
        type=["npy"],
        key="target_mask",
    )

slice_index = st.slider(
    "CT slice to display",
    min_value=0,
    max_value=63,
    value=32,
)

st.info(
    "Final candidate pipeline: V2 model → threshold 0.60 "
    "→ largest connected 3D component."
)

# --------------------------------------------------
# Run prediction
# --------------------------------------------------

if st.button("Predict kidney mask", type="primary"):

    if (
        reference_file is None
        or reference_mask_file is None
        or query_file is None
    ):
        st.error("Please upload all three required patches.")

    else:
        try:
            reference_image = read_patch(
                reference_file,
                "Reference CT",
            )

            reference_mask = read_patch(
                reference_mask_file,
                "Reference kidney mask",
                binary=True,
            )

            query_image = read_patch(
                query_file,
                "Query CT",
            )

            with st.spinner(
                "Loading V2 model and predicting on CPU..."
            ):
                model = load_model()

                with torch.inference_mode():
                    logits = model(
                        to_tensor(reference_image),
                        to_tensor(reference_mask),
                        to_tensor(query_image),
                    )

                    probabilities = (
                        torch.sigmoid(logits)[0, 0]
                        .cpu()
                        .numpy()
                    )

            raw_prediction = (
                probabilities >= PREDICTION_THRESHOLD
            ).astype(np.uint8)

            prediction = keep_largest_component(raw_prediction)

            st.success("Final prediction completed.")

            st.write(
                f"Threshold: {PREDICTION_THRESHOLD:.2f}"
                f" | Raw foreground fraction: "
                f"{raw_prediction.mean():.4f}"
                f" | Final foreground fraction: "
                f"{prediction.mean():.4f}"
            )

            # Prepare the selected CT slice for display.
            ct_slice = query_image[slice_index]

            low, high = np.percentile(
                ct_slice,
                [1, 99],
            )

            if high <= low:
                high = low + 1.0

            ct_uint8 = np.clip(
                (ct_slice - low) / (high - low) * 255,
                0,
                255,
            ).astype(np.uint8)

            prediction_slice = prediction[slice_index] > 0

            overlay = np.repeat(
                ct_uint8[:, :, None],
                3,
                axis=2,
            )

            red = np.array([255, 40, 40], dtype=np.float32)
            overlay[prediction_slice] = (
                0.55 * overlay[prediction_slice]
                + 0.45 * red
            ).astype(np.uint8)

            st.subheader("Final segmentation")

            image_col, mask_col, overlay_col = st.columns(3)

            with image_col:
                st.image(
                    ct_uint8,
                    caption="Query CT",
                    use_container_width=True,
                )

            with mask_col:
                st.image(
                    prediction[slice_index] * 255,
                    caption="Final kidney mask",
                    use_container_width=True,
                )

            with overlay_col:
                st.image(
                    overlay,
                    caption="CT with final mask overlay",
                    use_container_width=True,
                )

            # Download the final, post-processed binary mask.
            output = io.BytesIO()
            np.save(output, prediction)

            st.download_button(
                label="Download final kidney mask (.npy)",
                data=output.getvalue(),
                file_name="predicted_kidney_mask_threshold06_filtered.npy",
                mime="application/octet-stream",
            )

            # Optional evaluation, only when matching ground truth is supplied.
            if target_file is not None:
                target = read_patch(
                    target_file,
                    "Query ground-truth mask",
                    binary=True,
                )

                dice, iou = calculate_metrics(
                    prediction,
                    target,
                )

                st.subheader("Optional patch-level evaluation")

                metric_col1, metric_col2 = st.columns(2)

                metric_col1.metric("Dice", f"{dice:.4f}")
                metric_col2.metric("IoU", f"{iou:.4f}")

                st.caption(
                    "These metrics are meaningful only if the supplied "
                    "ground-truth mask corresponds to this exact query "
                    "patch. They are not clinical performance estimates."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
