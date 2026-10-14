from pathlib import Path
import torch

from real_mass_dataset import RealMASSDataset
from mass_model import MASSStyleModel


# Dataset
dataset = RealMASSDataset(
    Path("data/real/reference_query_dataset")
)

sample = dataset[0]

reference_image = sample[0].unsqueeze(0)
reference_mask = sample[1].unsqueeze(0)
query_image = sample[2].unsqueeze(0)
query_mask = sample[3].unsqueeze(0)

print("Reference image:", reference_image.shape)
print("Reference mask :", reference_mask.shape)
print("Query image    :", query_image.shape)
print("Query mask     :", query_mask.shape)


# Model
model = MASSStyleModel()

with torch.no_grad():
    prediction = model(
        reference_image,
        reference_mask,
        query_image
    )

print("Prediction:", prediction.shape)

print("Expected:", query_mask.shape)

print(
    "Shape match:",
    prediction.shape == query_mask.shape
)

if prediction.shape == query_mask.shape:
    print("\nREAL MASS MODEL TEST SUCCESSFUL!")