import torch
import numpy as np
import matplotlib.pyplot as plt

from mass_dataset import MASSDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("MASS-STYLE VISUALIZATION")
print("=" * 50)


# -----------------------------
# Load dataset
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
# Load model
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

ref_img = reference_image[0, 0].numpy()
ref_mask = reference_mask[0, 0].numpy()

query_img = query_image[0, 0].numpy()
query_mask_np = query_mask[0, 0].numpy()

prediction_np = prediction[0, 0].numpy()


# -----------------------------
# Select middle slice
# -----------------------------

slice_index = query_img.shape[0] // 2


# -----------------------------
# Create visualization
# -----------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(
    query_img[slice_index],
    cmap="gray"
)
plt.title("Query MRI")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(
    ref_mask[slice_index],
    cmap="gray"
)
plt.title("Reference Mask")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(
    prediction_np[slice_index],
    cmap="gray"
)
plt.title("Prediction")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(
    query_mask_np[slice_index],
    cmap="gray"
)
plt.title("Query Mask")
plt.axis("off")

plt.tight_layout()

output_path = (
    "data/processed/"
    "mass_segmentation_result.png"
)

plt.savefig(
    output_path,
    dpi=150
)

plt.close()

print("Visualization saved to:")
print(output_path)

print("=" * 50)
print("VISUALIZATION COMPLETE!")
print("=" * 50)