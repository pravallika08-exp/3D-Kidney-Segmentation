from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_mass_dataset import RealMASSDataset
from mass_model_v2 import MASSStyleModelV2

ROOT = Path("data/real")
OUT = ROOT / "project_figures"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Training curves
history_path = ROOT / "training_history_v2.csv"
with history_path.open(newline="", encoding="utf-8-sig") as f:
    history = list(csv.DictReader(f))

epochs = [int(r["epoch"]) for r in history]
train_loss = [float(r["train_loss"]) for r in history]
val_loss = [float(r["val_loss"]) for r in history]
val_dice = [float(r["val_dice"]) for r in history]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(epochs, train_loss, marker="o", label="Training loss")
axes[0].plot(epochs, val_loss, marker="o", label="Validation loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("V2 Training and Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, val_dice, marker="o", label="Validation Dice")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Dice")
axes[1].set_title("V2 Validation Dice")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "training_curves_v2.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 2. Prediction visualization for held-out case 00002, pair 00
dataset = RealMASSDataset(ROOT / "reference_query_case02")
if len(dataset) == 0:
    raise RuntimeError("No case00002 test pairs found.")

model = MASSStyleModelV2()
checkpoint = torch.load(
    ROOT / "best_mass_kits23_model_v2.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint)
model.eval()

ref_img, ref_mask, query_img, query_mask = dataset[0]

with torch.no_grad():
    logits = model(
        ref_img.unsqueeze(0),
        ref_mask.unsqueeze(0),
        query_img.unsqueeze(0),
    )
    pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()

def middle_slice(array):
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {array.shape}")
    return array[array.shape[0] // 2]

image_slice = middle_slice(query_img.numpy())
mask_slice = middle_slice(query_mask.numpy())
pred_slice = middle_slice(pred)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image_slice, cmap="gray")
axes[0].set_title("Query CT")
axes[1].imshow(image_slice, cmap="gray")
axes[1].imshow(mask_slice, cmap="Greens", alpha=0.45)
axes[1].set_title("Ground-truth mask")
axes[2].imshow(image_slice, cmap="gray")
axes[2].imshow(pred_slice, cmap="Reds", alpha=0.45)
axes[2].set_title("V2 prediction (threshold 0.5)")

for ax in axes:
    ax.axis("off")

fig.suptitle("Held-out case00002 — Pair 00")
fig.tight_layout()
fig.savefig(OUT / "case02_pair00_prediction_v2.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("FIGURES CREATED:")
print(OUT / "training_curves_v2.png")
print(OUT / "case02_pair00_prediction_v2.png")
print("Note: prediction visualization shows the middle slice of one test patch.")
