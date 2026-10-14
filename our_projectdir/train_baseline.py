import torch
import torch.nn as nn

from dataset import MedicalVolumeDataset
from model import Simple3DUNet


print("=" * 50)
print("3D BASELINE TRAINING")
print("=" * 50)


# -----------------------------
# 1. Load data
# -----------------------------

dataset = MedicalVolumeDataset(
    "data/processed/image.npy",
    "data/processed/mask.npy"
)

sample = dataset[0]

image = sample["image"].unsqueeze(0)
mask = sample["mask"].unsqueeze(0)


# -----------------------------
# 2. Create model
# -----------------------------

model = Simple3DUNet()


# -----------------------------
# 3. Loss + optimizer
# -----------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 4. Training loop
# -----------------------------

epochs = 10

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    prediction = model(image)

    loss = criterion(prediction, mask)

    loss.backward()

    optimizer.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {loss.item():.6f}"
    )


# -----------------------------
# 5. Save model
# -----------------------------

torch.save(
    model.state_dict(),
    "data/processed/baseline_model.pth"
)

print()
print("Model saved to:")
print("data/processed/baseline_model.pth")

print("=" * 50)
print("BASELINE TRAINING COMPLETE!")
print("=" * 50)