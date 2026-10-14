import torch

from dataset import MedicalVolumeDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("MASS-STYLE REAL DATA TEST")
print("=" * 50)


# Load our processed medical data
dataset = MedicalVolumeDataset(
    "data/processed/image.npy",
    "data/processed/mask.npy"
)

sample = dataset[0]

# Add batch dimension
image = sample["image"].unsqueeze(0)
mask = sample["mask"].unsqueeze(0)


# For this first experiment:
# use the same sample as reference and query
reference_image = image
reference_mask = mask
query_image = image


print("Reference image:", reference_image.shape)
print("Reference mask :", reference_mask.shape)
print("Query image    :", query_image.shape)


# Create MASS-style model
model = MASSStyleModel()
model.eval()


# Run inference
with torch.no_grad():

    output = model(
        reference_image,
        reference_mask,
        query_image
    )


print("Prediction:", output.shape)

print("=" * 50)
print("MASS-STYLE REAL DATA TEST SUCCESSFUL!")
print("=" * 50)