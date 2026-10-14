from pathlib import Path

import torch

from real_mass_dataset import RealMASSDataset
from mass_model import MASSStyleModel


def dice_score(prediction, target, smooth=1.0):
    prediction = prediction.float()
    target = target.float()

    intersection = (prediction * target).sum()

    return (
        2.0 * intersection + smooth
    ) / (
        prediction.sum() + target.sum() + smooth
    )


dataset = RealMASSDataset(
    Path("data/real/reference_query_case02")
)

model = MASSStyleModel()

model.load_state_dict(
    torch.load(
        "data/real/mass_kits23_model.pth",
        map_location="cpu"
    )
)

model.eval()

scores = []

with torch.no_grad():

    for i in range(len(dataset)):

        reference_image, reference_mask, query_image, query_mask = dataset[i]

        logits = model(
            reference_image.unsqueeze(0),
            reference_mask.unsqueeze(0),
            query_image.unsqueeze(0)
        )

        probabilities = torch.sigmoid(logits)

        prediction = (probabilities > 0.5).float()

        dice = dice_score(
            prediction,
            query_mask.unsqueeze(0)
        )

        scores.append(dice.item())

        print(
            f"Case 00002 Pair {i:02d} - "
            f"Dice: {dice.item():.4f}"
        )


average_dice = sum(scores) / len(scores)

print()
print("=== UNSEEN PATIENT TEST ===")
print(f"Average Dice: {average_dice:.4f}")