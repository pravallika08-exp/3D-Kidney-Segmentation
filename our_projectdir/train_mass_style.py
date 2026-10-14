import torch
import torch.nn as nn

from dataset import MedicalVolumeDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("MASS-STYLE TRAINING")
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
# 3. Create model
# -----------------------------

model = MASSStyleModel()


# -----------------------------
# 4. Loss and optimizer
# -----------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 5. Training
# -----------------------------

epochs = 10

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    prediction = model(
        reference_image,
        reference_mask,
        query_image
    )

    loss = criterion(
        prediction,
        mask
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {loss.item():.6f}"
    )


# -----------------------------
# 6. Save model
# -----------------------------

torch.save(
    model.state_dict(),
    "data/processed/mass_style_model.pth"
)

print()
print("Model saved to:")
print("data/processed/mass_style_model.pth")

print("=" * 50)
print("MASS-STYLE TRAINING COMPLETE!")
print("=" * 50)