from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from real_mass_dataset import RealMASSDataset
from mass_model import MASSStyleModel


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (probabilities * targets).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        probabilities.sum() + targets.sum() + smooth
    )

    return 1.0 - dice


def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(logits, targets)

    return bce + dice


# Dataset
dataset = RealMASSDataset(
    Path("data/real/training_reference_query")
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

print("Dataset size:", len(dataset))


# Model
model = MASSStyleModel()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# Training
epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for reference_image, reference_mask, query_image, query_mask in loader:

        optimizer.zero_grad()

        logits = model(
            reference_image,
            reference_mask,
            query_image
        )

        loss = combined_loss(
            logits,
            query_mask
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {average_loss:.6f}"
    )


# Save model
output_path = Path(
    "data/real/mass_kits23_model.pth"
)

torch.save(
    model.state_dict(),
    output_path
)

print("Model saved to:", output_path)