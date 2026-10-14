import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mass_dataset import MASSDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("MASS-STYLE REFERENCE/QUERY TRAINING")
print("=" * 50)


# -----------------------------
# 1. Dataset
# -----------------------------

dataset = MASSDataset(
    "data/processed/samples"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)


# -----------------------------
# 2. Model
# -----------------------------

model = MASSStyleModel()


# -----------------------------
# 3. Loss + optimizer
# -----------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 4. Training
# -----------------------------

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch in loader:

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

    average_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.6f}"
    )


# -----------------------------
# 5. Save
# -----------------------------

torch.save(
    model.state_dict(),
    "data/processed/mass_reference_query_model.pth"
)

print()
print("Model saved to:")
print(
    "data/processed/"
    "mass_reference_query_model.pth"
)

print("=" * 50)
print("REFERENCE/QUERY TRAINING COMPLETE!")
print("=" * 50)