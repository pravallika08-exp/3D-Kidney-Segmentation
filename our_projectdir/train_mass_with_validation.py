import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mass_dataset import MASSDataset
from validation_dataset import ValidationDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("MASS TRAINING + VALIDATION")
print("=" * 50)


# -----------------------------
# Training data
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
# Validation data
# -----------------------------

val_dataset = ValidationDataset(
    "data/processed/val"
)


# -----------------------------
# Model
# -----------------------------

model = MASSStyleModel()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# Training
# -----------------------------

epochs = 10

for epoch in range(epochs):

    # ===== TRAIN =====

    model.train()

    total_train_loss = 0.0

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

        total_train_loss += loss.item()


    train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # ===== VALIDATION =====

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

        # For this prototype, use the
        # first training sample as reference.
        reference = train_dataset[0]

        reference_image = (
            reference["reference_image"]
            .unsqueeze(0)
        )

        reference_mask = (
            reference["reference_mask"]
            .unsqueeze(0)
        )

        val_output = model(
            reference_image,
            reference_mask,
            val_image
        )

        val_probability = torch.sigmoid(
            val_output
        )

        val_prediction = (
            val_probability > 0.5
        ).float()

        intersection = (
            val_prediction * val_mask
        ).sum()

        val_dice = (
            (2.0 * intersection + 1e-8)
            /
            (
                val_prediction.sum()
                + val_mask.sum()
                + 1e-8
            )
        )


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Dice: {val_dice.item():.4f}"
    )


# -----------------------------
# Save final model
# -----------------------------

torch.save(
    model.state_dict(),
    "data/processed/"
    "mass_final_model.pth"
)

print()
print("Saved model:")
print(
    "data/processed/mass_final_model.pth"
)

print("=" * 50)
print("TRAINING + VALIDATION COMPLETE!")
print("=" * 50)