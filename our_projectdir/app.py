
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# Make the model module importable on local and cloud deployments.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mass_model_v2 import MASSStyleModelV2


# --------------------------------------------------
# App configuration
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


# --------------------------------------------------
# Load the trained model
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
# Read and validate uploaded patches
# --------------------------------------------------

def read_patch(upload, label, binary=False):
    if upload is None:
        return None

    arr = np.load(upload, allow_pickle=False)
    arr = np.asarray(arr)

    # Accept either (64,64,64) or (1,64,64,64).
    if arr.shape == (1, 64, 64, 64):
        arr = arr[0]

    if arr.shape != (64, 64, 64):
        raise ValueError(
            f"{label} must have shape (64,64,64). "
            f"Received: {arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            f"{label} contains NaN or infinite values."
        )

    if binary and not np.isin(arr, [0, 1]).all():
        raise ValueError(
            f"{label} must contain only binary values 0 and 1."
        )

    return arr.astype(np.float32)


def to_tensor(arr):
    return (
        torch.from_numpy(arr.copy())
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )


# --------------------------------------------------
# Upload fields
# --------------------------------------------------

st.subheader("Upload 3D patches")

ref_file = st.file_uploader(
    "1. Reference CT patch (.npy)",
    type=["npy"],
    key="reference_ct",
)

mask_file = st.file_uploader(
    "2. Reference kidney mask (.npy)",
    type=["npy"],
    key="reference_mask",
)

query_file = st.file_uploader(
    "3. Query CT patch (.npy)",
    type=["npy"],
    key="query_ct",
)

target_file = st.file_uploader(
    "Optional: query ground-truth kidney mask (.npy)",
    type=["npy"],
    key="target_mask",
)

slice_index = st.slider(
    "CT slice to display",
    min_value=0,
    max_value=63,
    value=32,
)


# --------------------------------------------------
# Predict
# --------------------------------------------------

if st.button("Predict kidney mask", type="primary"):

    if ref_file is None or mask_file is None or query_file is None:
        st.error("Please upload all three required patches.")

    else:
        try:
            ref = read_patch(
                ref_file,
                "Reference CT",
            )

            ref_mask = read_patch(
                mask_file,
                "Reference kidney mask",
                binary=True,
            )

            query = read_patch(
                query_file,
                "Query CT",
            )

            with st.spinner(
                "Loading V2 model and predicting on CPU..."
            ):
                model = load_model()

                with torch.inference_mode():
                    logits = model(
                        to_tensor(ref),
                        to_tensor(ref_mask),
                        to_tensor(query),
                    )

                    probabilities = (
                        torch.sigmoid(logits)[0, 0]
                        .cpu()
                        .numpy()
                    )

                    prediction = (
                        probabilities >= 0.5
                    ).astype(np.uint8)

            st.success("Prediction completed successfully.")

            st.write(
                "Predicted foreground fraction: "
                f"{prediction.mean():.4f}"
            )

            # ------------------------------------------
            # Build display images without Matplotlib
            # ------------------------------------------

            ct_slice = query[slice_index]

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

            pred_slice = prediction[slice_index]

            overlay = np.repeat(
                ct_uint8[:, :, None],
                3,
                axis=2,
            )

            foreground = pred_slice.astype(bool)
            red = np.array(
                [255, 40, 40],
                dtype=np.float32,
            )

            overlay[foreground] = (
                0.55 * overlay[foreground]
                + 0.45 * red
            ).astype(np.uint8)

            st.subheader("Prediction visualization")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    ct_uint8,
                    caption="Query CT",
                    use_container_width=True,
                )

            with col2:
                st.image(
                    pred_slice * 255,
                    caption="Predicted kidney mask",
                    use_container_width=True,
                )

            with col3:
                st.image(
                    overlay,
                    caption="Prediction overlay",
                    use_container_width=True,
                )

            # ------------------------------------------
            # Download predicted mask
            # ------------------------------------------

            output = io.BytesIO()
            np.save(output, prediction)

            st.download_button(
                label="Download predicted mask (.npy)",
                data=output.getvalue(),
                file_name="predicted_kidney_mask.npy",
                mime="application/octet-stream",
            )

            # ------------------------------------------
            # Optional evaluation against ground truth
            # ------------------------------------------

            if target_file is not None:
                target = read_patch(
                    target_file,
                    "Query ground-truth mask",
                    binary=True,
                )

                intersection = np.logical_and(
                    prediction,
                    target,
                ).sum()

                denominator = (
                    prediction.sum()
                    + target.sum()
                )

                dice = (
                    1.0
                    if denominator == 0
                    else 2.0 * intersection / denominator
                )

                union = np.logical_or(
                    prediction,
                    target,
                ).sum()

                iou = (
                    1.0
                    if union == 0
                    else intersection / union
                )

                st.subheader(
                    "Optional patch-level evaluation"
                )

                metric_col1, metric_col2 = st.columns(2)

                metric_col1.metric(
                    "Dice",
                    f"{dice:.4f}",
                )

                metric_col2.metric(
                    "IoU",
                    f"{iou:.4f}",
                )

                st.caption(
                    "These metrics are valid only when the "
                    "ground-truth mask belongs to this exact "
                    "query patch. They are not clinical "
                    "performance measurements."
                )

        except Exception as exc:
            st.error(
                f"Prediction failed: {exc}"
            )
