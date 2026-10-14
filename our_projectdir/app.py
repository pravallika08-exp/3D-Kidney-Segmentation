import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from mass_model_v2 import MASSStyleModelV2

st.set_page_config(page_title="3D Kidney Segmentation", layout="wide")
st.title("MASS-Style 3D Kidney Segmentation")
st.warning(
    "Research prototype only — not for clinical use. "
    "This is not an official reproduction of MASS. "
    "Upload preprocessed 64x64x64 NumPy (.npy) patches only."
)

CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "data" / "real" / "best_mass_kits23_model_v2.pth"
)
DEVICE = torch.device("cpu")


@st.cache_resource
def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = MASSStyleModelV2().to(DEVICE)
    try:
        state = torch.load(
            CHECKPOINT, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        state = torch.load(CHECKPOINT, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def read_patch(upload, label, binary=False):
    if upload is None:
        return None

    arr = np.load(upload, allow_pickle=False)
    arr = np.asarray(arr)

    # Accept (64,64,64) or a single leading singleton dimension.
    if arr.shape == (1, 64, 64, 64):
        arr = arr[0]

    if arr.shape != (64, 64, 64):
        raise ValueError(
            f"{label} must have shape (64,64,64); got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains NaN or infinite values.")

    if binary:
        if not np.isin(arr, [0, 1]).all():
            raise ValueError(
                f"{label} must be a binary mask containing only 0 and 1."
            )
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32)

    return arr


def to_tensor(arr):
    return torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0).to(DEVICE)


ref_file = st.file_uploader(
    "1. Reference CT patch (.npy)", type=["npy"], key="reference_ct"
)
mask_file = st.file_uploader(
    "2. Reference kidney mask (.npy)", type=["npy"], key="reference_mask"
)
query_file = st.file_uploader(
    "3. Query CT patch (.npy)", type=["npy"], key="query_ct"
)
target_file = st.file_uploader(
    "Optional: query ground-truth mask for evaluation (.npy)",
    type=["npy"], key="target_mask"
)

slice_index = st.slider("Slice to display", min_value=0, max_value=63, value=32)

if st.button("Predict kidney mask", type="primary"):
    if not (ref_file and mask_file and query_file):
        st.error("Please upload all three required patches.")
    else:
        try:
            ref = read_patch(ref_file, "Reference CT")
            ref_mask = read_patch(mask_file, "Reference mask", binary=True)
            query = read_patch(query_file, "Query CT")

            with st.spinner("Loading model and predicting on CPU..."):
                model = load_model()
                with torch.inference_mode():
                    logits = model(
                        to_tensor(ref),
                        to_tensor(ref_mask),
                        to_tensor(query),
                    )
                    probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
                    prediction = (probabilities >= 0.5).astype(np.uint8)

            st.success("Prediction completed.")
            st.write(
                f"Predicted foreground fraction: "
                f"{prediction.mean():.4f}"
            )

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].imshow(query[slice_index], cmap="gray")
            axes[0].set_title("Query CT")
            axes[1].imshow(prediction[slice_index], cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Predicted mask")
            axes[2].imshow(query[slice_index], cmap="gray")
            axes[2].imshow(
                prediction[slice_index], cmap="autumn", alpha=0.4,
                    vmin=0, vmax=1
            )
            axes[2].set_title("Prediction overlay")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            output = io.BytesIO()
            np.save(output, prediction)
            st.download_button(
                "Download predicted mask (.npy)",
                data=output.getvalue(),
                file_name="predicted_kidney_mask.npy",
                mime="application/octet-stream",
            )

            if target_file is not None:
                target = read_patch(
                    target_file, "Query ground-truth mask", binary=True
                )
                intersection = np.logical_and(prediction, target).sum()
                denominator = prediction.sum() + target.sum()
                dice = (
                    1.0 if denominator == 0
                    else 2.0 * intersection / denominator
                )
                union = np.logical_or(prediction, target).sum()
                iou = 1.0 if union == 0 else intersection / union
                st.subheader("Optional patch-level evaluation")
                st.metric("Dice", f"{dice:.4f}")
                st.metric("IoU", f"{iou:.4f}")
                st.caption(
                    "These metrics are meaningful only when the supplied "
                    "ground-truth mask matches this exact query patch."
                )

        except Exception as exc:
            st.error(f"Could not run prediction: {exc}")
