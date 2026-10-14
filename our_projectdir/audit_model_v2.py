from pathlib import Path
import torch
from real_mass_dataset import RealMASSDataset
from mass_model_v2 import MASSStyleModelV2

torch.set_num_threads(2)
root = Path("data/real")

print("===== CHECKPOINT LOAD =====")
model = MASSStyleModelV2()
state = torch.load(
    root / "best_mass_kits23_model_v2.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state)
model.eval()
print("PASS: V2 checkpoint loaded")

print("\n===== DATASET INTEGRITY =====")
for folder, expected in [
    (root / "training_reference_query_v2", 50),
    (root / "reference_query_case02", 10),
]:
    ds = RealMASSDataset(folder)
    assert len(ds) == expected, f"{folder}: expected {expected}, got {len(ds)}"

    for i in range(len(ds)):
        ref_img, ref_mask, query_img, query_mask = ds[i]
        for name, tensor in [
            ("reference image", ref_img),
            ("reference mask", ref_mask),
            ("query image", query_img),
            ("query mask", query_mask),
        ]:
            assert torch.isfinite(tensor).all(), f"{folder} pair {i}: non-finite {name}"
            assert tensor.shape == (1, 64, 64, 64), (
                f"{folder} pair {i}: unexpected {name} shape {tensor.shape}"
            )

        for name, mask in [("reference mask", ref_mask), ("query mask", query_mask)]:
            assert ((mask == 0) | (mask == 1)).all(), (
                f"{folder} pair {i}: non-binary {name}"
            )

    print(f"PASS: {folder} — {len(ds)} pairs, valid shapes and binary masks")

print("\n===== INFERENCE SMOKE TEST =====")
test_ds = RealMASSDataset(root / "reference_query_case02")
ref_img, ref_mask, query_img, query_mask = test_ds[0]

with torch.no_grad():
    logits = model(
        ref_img.unsqueeze(0),
        ref_mask.unsqueeze(0),
        query_img.unsqueeze(0),
    )
    probabilities = torch.sigmoid(logits)

assert logits.shape == (1, 1, 64, 64, 64), f"Unexpected output shape: {logits.shape}"
assert torch.isfinite(probabilities).all(), "Non-finite model output"
assert probabilities.min().item() >= 0 and probabilities.max().item() <= 1

prediction = probabilities > 0.5
print("PASS: inference completed")
print("Output shape:", tuple(logits.shape))
print(f"Prediction foreground fraction: {prediction.float().mean().item():.4f}")
print("\nFINAL CHECKS: PASS")
