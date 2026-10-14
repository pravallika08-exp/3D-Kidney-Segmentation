import torch
import numpy as np
import matplotlib.pyplot as plt

from mass_dataset import MASSDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("SEGMENTATION OVERLAY")
print("=" * 50)


# -----------------------------
# Load sample
# -----------------------------

dataset = MASSDataset(
    "data/processed/samples"
)

sample = dataset[0]

reference_image = sample["reference_image"].unsqueeze(0)
reference_mask = sample["reference_mask"].unsqueeze(0)

query_image = sample["query_image"].unsqueeze(0)
query_mask = sample["query_mask"].unsqueeze(0)


# -----------------------------
# Load trained model
# -----------------------------

model = MASSStyleModel()

model.load_state_dict(
    torch.load(
        "data/processed/mass_reference_query_model.pth",
        map_location="cpu"
    )
)

model.eval()


# -----------------------------
# Prediction
# -----------------------------

with torch.no_grad():

    output = model(
        reference_image,
        reference_mask,
        query_image
    )

    probability = torch.sigmoid(output)

    prediction = (
        probability > 0.5
    ).float()


# -----------------------------
# Convert to NumPy
# -----------------------------

query = query_image[0, 0].numpy()
pred = prediction[0, 0].numpy()
ground_truth = query_mask[0, 0].numpy()


# -----------------------------
# Middle slice
# -----------------------------

slice_index = query.shape[0] // 2

mri_slice = query[slice_index]
pred_slice = pred[slice_index]
gt_slice = ground_truth[slice_index]


# -----------------------------
# Create overlay
# -----------------------------

plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)

plt.imshow(
    mri_slice,
    cmap="gray"
)

plt.title("Query MRI")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    mri_slice,
    cmap="gray"
)

plt.imshow(
    pred_slice,
    alpha=0.45
)

plt.title("Predicted Segmentation")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    mri_slice,
    cmap="gray"
)

plt.imshow(
    gt_slice,
    alpha=0.45
)

plt.title("Reference Mask")
plt.axis("off")


plt.tight_layout()


output_path = (
    "data/processed/"
    "mass_overlay.png"
)

plt.savefig(
    output_path,
    dpi=150
)

plt.close()


print("Overlay saved to:")
print(output_path)

print("=" * 50)
print("OVERLAY COMPLETE!")
print("=" * 50)