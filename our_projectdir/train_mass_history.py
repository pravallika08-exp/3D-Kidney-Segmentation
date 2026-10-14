import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mass_dataset import MASSDataset
from validation_dataset import ValidationDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("MASS TRAINING WITH HISTORY")
print("=" * 50)


# -----------------------------
# 1. Load training data
# -----------------------------

train_dataset = MASSDataset(
    "data/processed/train"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)


# -----------------------------
# 2. Load validation data
# -----------------------------

val_dataset = ValidationDataset(
    "data/processed/val"
)


# -----------------------------
# 3. Model
# -----------------------------

model = MASSStyleModel()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 4. History
# -----------------------------

history = []

epochs = 10


# -----------------------------
# 5. Training
# -----------------------------

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        reference_image = batch["reference_image"]
        reference_mask = batch["reference_mask"]

        query_image = batch["query_image"]
        query_mask = batch["query_mask"]

        optimizer.zero_grad()

        prediction = model(
            reference_image,
            reference_mask,
            query_image
        )

        loss = criterion(
            prediction,
            query_mask
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    train_loss = (
        total_loss /
        len(train_loader)
    )


    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    with torch.no_grad():

        val_sample = val_dataset[0]

        val_image = (
            val_sample["image"]
            .unsqueeze(0)
        )

        val_mask = (
            val_sample["mask"]
            .unsqueeze(0)
        )

        reference = train_dataset[0]

        reference_image = (
            reference["reference_image"]
            .unsqueeze(0)
        )

        reference_mask = (
            reference["reference_mask"]
            .unsqueeze(0)
        )

        output = model(
            reference_image,
            reference_mask,
            val_image
        )

        probability = torch.sigmoid(output)

        prediction = (
            probability > 0.5
        ).float()

        intersection = (
            prediction * val_mask
        ).sum()

        val_dice = (
            (2.0 * intersection + 1e-8)
            /
            (
                prediction.sum()
                + val_mask.sum()
                + 1e-8
            )
        )

        val_dice = val_dice.item()


    # -----------------------------
    # Store history
    # -----------------------------

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_dice": val_dice
    })


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {train_loss:.6f} "
        f"Val Dice: {val_dice:.4f}"
    )


# -----------------------------
# 6. Save model
# -----------------------------

model_path = Path(
    "data/processed/"
    "mass_history_model.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# -----------------------------
# 7. Save CSV
# -----------------------------

csv_path = Path(
    "data/processed/"
    "training_history.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_dice"
        ]
    )

    writer.writeheader()

    writer.writerows(history)


print()
print("Model saved:", model_path)
print("History saved:", csv_path)

print("=" * 50)
print("TRAINING HISTORY COMPLETE!")
print("=" * 50)