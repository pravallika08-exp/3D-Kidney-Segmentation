from pathlib import Path
import numpy as np
import torch
from mass_model import MASSStyleModel

root = Path("data/real/reference_query_case02")
model_path = Path("data/real/mass_kits23_model.pth")
thresholds = (0.1, 0.2, 0.3, 0.5)
device = torch.device("cpu")

pairs = sorted(p for p in root.iterdir()
               if p.is_dir() and p.name.startswith("pair_"))

if not pairs:
    raise RuntimeError(f"No pair directories found in {root}")

def load_tensor(path):
    arr = np.load(path).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

model = MASSStyleModel().to(device)
state = torch.load(model_path, map_location=device, weights_only=True)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
model.eval()

scores = {t: [] for t in thresholds}

for pair in pairs:
    ref_img = load_tensor(pair / "reference_image.npy")
    ref_mask = load_tensor(pair / "reference_mask.npy")
    query_img = load_tensor(pair / "query_image.npy")
    truth = np.load(pair / "query_mask.npy") > 0.5

    with torch.no_grad():
        logits = model(ref_img, ref_mask, query_img)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    print(f"\n{pair.name}:")
    for t in thresholds:
        pred = probs >= t
        denominator = int(pred.sum()) + int(truth.sum())
        dice = (
            2.0 * np.logical_and(pred, truth).sum() / denominator
            if denominator else 1.0
        )
        scores[t].append(float(dice))
        print(f"  Threshold {t:.1f}: Dice = {dice:.4f}")

print("\n=== AVERAGE 3D DICE ACROSS CASE 00002 ===")
for t in thresholds:
    print(f"Threshold {t:.1f}: Mean Dice = {np.mean(scores[t]):.4f}")
