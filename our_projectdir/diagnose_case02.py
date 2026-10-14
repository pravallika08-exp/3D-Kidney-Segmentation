from pathlib import Path
import numpy as np
import torch
from mass_model import MASSStyleModel

root = Path("data/real/reference_query_case02/pair_00")
device = torch.device("cpu")

def load_tensor(name):
    a = np.load(root / name).astype(np.float32)
    return torch.from_numpy(a).unsqueeze(0).unsqueeze(0)

ref_img = load_tensor("reference_image.npy")
ref_mask = load_tensor("reference_mask.npy")
query_img = load_tensor("query_image.npy")
target = load_tensor("query_mask.npy")

model = MASSStyleModel().to(device)
state = torch.load(
    "data/real/mass_kits23_model.pth",
    map_location=device,
    weights_only=True
)
if "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    logits = model(ref_img, ref_mask, query_img)
    probs = torch.sigmoid(logits)

p = probs.squeeze().numpy()
y = target.squeeze().numpy()

print("Query shape:", tuple(query_img.shape))
print("Target positive voxels:", int((y > 0.5).sum()))
print("Probability min:", float(p.min()))
print("Probability max:", float(p.max()))
print("Probability mean:", float(p.mean()))

for threshold in (0.1, 0.2, 0.3, 0.5):
    pred = p >= threshold
    positives = int(pred.sum())
    dice = 2 * (pred & (y > 0.5)).sum() / max(
        positives + int((y > 0.5).sum()), 1
    )
    print(
        f"Threshold {threshold:.1f}: "
        f"predicted voxels={positives}, 3D Dice={dice:.4f}"
    )
