import torch
import numpy as np

from dataset import MedicalVolumeDataset
from model import Simple3DUNet


print("=" * 50)
print("BASELINE EVALUATION")
print("=" * 50)


# Load data
dataset = MedicalVolumeDataset(
    "data/processed/image.npy",
    "data/processed/mask.npy"
)

sample = dataset[0]

image = sample["image"].unsqueeze(0)
mask = sample["mask"].unsqueeze(0)


# Load trained model
model = Simple3DUNet()

model.load_state_dict(
    torch.load(
        "data/processed/baseline_model.pth",
        map_location="cpu"
    )
)

model.eval()


# Prediction
with torch.no_grad():
    output = model(image)
    probability = torch.sigmoid(output)

    prediction = (probability > 0.5).float()


# Dice score
intersection = (prediction * mask).sum()

dice = (
    (2.0 * intersection + 1e-8)
    /
    (prediction.sum() + mask.sum() + 1e-8)
)


print("Image shape:", image.shape)
print("Prediction shape:", prediction.shape)
print("Ground-truth mask shape:", mask.shape)

print("Predicted foreground voxels:", int(prediction.sum()))
print("Ground-truth foreground voxels:", int(mask.sum()))

print()
print(f"Dice score: {dice.item():.4f}")

print("=" * 50)
print("BASELINE EVALUATION COMPLETE!")
print("=" * 50)