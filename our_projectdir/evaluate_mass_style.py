import torch

from dataset import MedicalVolumeDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("MASS-STYLE MODEL EVALUATION")
print("=" * 50)


# -----------------------------
# 1. Load processed data
# -----------------------------

dataset = MedicalVolumeDataset(
    "data/processed/image.npy",
    "data/processed/mask.npy"
)

sample = dataset[0]

image = sample["image"].unsqueeze(0)
mask = sample["mask"].unsqueeze(0)


# -----------------------------
# 2. Reference + query
# -----------------------------

reference_image = image
reference_mask = mask
query_image = image


# -----------------------------
# 3. Load trained model
# -----------------------------

model = MASSStyleModel()

model.load_state_dict(
    torch.load(
        "data/processed/mass_style_model.pth",
        map_location="cpu"
    )
)

model.eval()


# -----------------------------
# 4. Prediction
# -----------------------------

with torch.no_grad():

    output = model(
        reference_image,
        reference_mask,
        query_image
    )

    probability = torch.sigmoid(output)

    prediction = (probability > 0.5).float()


# -----------------------------
# 5. Dice score
# -----------------------------

intersection = (prediction * mask).sum()

dice = (
    (2.0 * intersection + 1e-8)
    /
    (prediction.sum() + mask.sum() + 1e-8)
)


print("Prediction shape:", prediction.shape)
print("Predicted voxels:", int(prediction.sum()))
print("Reference voxels:", int(mask.sum()))

print()
print(f"Dice score: {dice.item():.4f}")

print("=" * 50)
print("MASS-STYLE EVALUATION COMPLETE!")
print("=" * 50)