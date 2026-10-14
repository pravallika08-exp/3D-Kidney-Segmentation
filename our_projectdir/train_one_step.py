import torch
import torch.nn as nn

from dataset import MedicalVolumeDataset
from model import Simple3DUNet


print("=" * 50)
print("ONE TRAINING STEP")
print("=" * 50)


# -----------------------------
# 1. Load dataset
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
# 3. Loss function
# -----------------------------

criterion = nn.BCEWithLogitsLoss()


# -----------------------------
# 4. Optimizer
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 5. Forward pass
# -----------------------------

prediction = model(image)

loss = criterion(prediction, mask)

print("Initial loss:", loss.item())


# -----------------------------
# 6. Backpropagation
# -----------------------------

optimizer.zero_grad()

loss.backward()

optimizer.step()


print("Training step completed.")

print("=" * 50)
print("ONE TRAINING STEP SUCCESSFUL!")
print("=" * 50)