import torch
from torch.utils.data import DataLoader

from dataset import MedicalVolumeDataset


dataset = MedicalVolumeDataset(
    "data/processed/image.npy",
    "data/processed/mask.npy"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

print("=" * 50)
print("DATALOADER TEST")
print("=" * 50)

for batch in loader:

    image = batch["image"]
    mask = batch["mask"]

    print("Batch image shape:", image.shape)
    print("Batch mask shape :", mask.shape)

    print("Image dtype:", image.dtype)
    print("Mask dtype :", mask.dtype)

    break

print("=" * 50)
print("DATALOADER TEST SUCCESSFUL!")
print("=" * 50)