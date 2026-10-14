from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from mass_model import MASSStyleModel

pair_dir = Path("data/real/reference_query_case02/pair_00")
model_path = Path("data/real/mass_kits23_model.pth")
output_path = Path("data/real/case02_pair00_threshold01.png")

device = torch.device("cpu")

def load_array(name):
    array = np.load(pair_dir / name)
    return torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)

reference_image = load_array("reference_image.npy")
reference_mask = load_array("reference_mask.npy")
query_image = load_array("query_image.npy")
query_mask = load_array("query_mask.npy")

model = MASSStyleModel().to(device)
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    logits = model(
        reference_image.to(device),
        reference_mask.to(device),
        query_image.to(device),
    )
    prediction = (torch.sigmoid(logits) >= 0.1).float().squeeze().cpu().numpy()

image = query_image.squeeze().numpy()
truth = query_mask.squeeze().numpy()

# Show the central axial slice.
z = int(np.argmax(np.sum(truth > 0.5, axis=(1, 2))))
img_slice = image[z]
truth_slice = truth[z]
pred_slice = prediction[z]

dice = (
    2 * np.logical_and(pred_slice > 0, truth_slice > 0).sum()
    / max((pred_slice > 0).sum() + (truth_slice > 0).sum(), 1)
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img_slice, cmap="gray")
axes[0].set_title("Query CT")

axes[1].imshow(img_slice, cmap="gray")
axes[1].imshow(truth_slice, cmap="Greens", alpha=0.45)
axes[1].set_title("Ground Truth")

axes[2].imshow(img_slice, cmap="gray")
axes[2].imshow(pred_slice, cmap="Reds", alpha=0.45)
axes[2].set_title(f"Prediction (slice Dice: {dice:.3f})")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved visualization: {output_path}")
print(f"Central slice index: {z}")
print(f"Central slice Dice: {dice:.4f}")
