from pathlib import Path
import torch
from real_mass_dataset import RealMASSDataset
from mass_model import MASSStyleModel

dataset = RealMASSDataset(
    Path("data/real/reference_query_dataset")
)

reference_image, reference_mask, query_image, query_mask = dataset[0]

model = MASSStyleModel()

model.load_state_dict(
    torch.load(
        "data/real/mass_kits23_model.pth",
        map_location="cpu"
    )
)

model.eval()

with torch.no_grad():
    logits = model(
        reference_image.unsqueeze(0),
        reference_mask.unsqueeze(0),
        query_image.unsqueeze(0)
    )

    probabilities = torch.sigmoid(logits)

print("Logit min:", logits.min().item())
print("Logit max:", logits.max().item())
print("Probability min:", probabilities.min().item())
print("Probability max:", probabilities.max().item())
print("Probability mean:", probabilities.mean().item())
print("Predicted positive voxels:", (probabilities > 0.5).sum().item())
print("Ground-truth positive voxels:", query_mask.sum().item())