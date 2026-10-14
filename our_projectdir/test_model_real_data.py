import torch

from dataset import MedicalVolumeDataset
from model import Simple3DUNet


print("=" * 50)
print("REAL DATA + 3D MODEL TEST")
print("=" * 50)


# Load our processed medical volume
dataset = MedicalVolumeDataset(
    "data/processed/image.npy",
    "data/processed/mask.npy"
)

sample = dataset[0]

image = sample["image"].unsqueeze(0)
mask = sample["mask"].unsqueeze(0)


print("Input image shape:", image.shape)
print("Target mask shape:", mask.shape)


# Create model
model = Simple3DUNet()


# Run the real volume through the model
with torch.no_grad():
    prediction = model(image)


print("Prediction shape:", prediction.shape)

print()
print("Input dtype:", image.dtype)
print("Prediction dtype:", prediction.dtype)

print("=" * 50)
print("REAL DATA MODEL TEST SUCCESSFUL!")
print("=" * 50)