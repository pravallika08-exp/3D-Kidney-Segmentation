from pathlib import Path

import torch
from torch.utils.data import DataLoader

from real_mass_dataset import RealMASSDataset
from mass_model import MASSStyleModel


DATASET_PATH = Path(
    "data/real/reference_query_dataset"
)

MODEL_PATH = Path(
    "data/real/mass_kits23_model.pth"
)


def dice_score(prediction, target):

    prediction = (prediction > 0.5).float()

    intersection = (
        prediction * target
    ).sum()

    denominator = (
        prediction.sum() +
        target.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection / denominator
    ).item()


dataset = RealMASSDataset(DATASET_PATH)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

model = MASSStyleModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

scores = []

with torch.no_grad():

    for i, (
        reference_image,
        reference_mask,
        query_image,
        query_mask
    ) in enumerate(loader):

        prediction = model(
            reference_image,
            reference_mask,
            query_image
        )

        dice = dice_score(
            torch.sigmoid(prediction),
            query_mask
        )

        scores.append(dice)

        print(
            f"Pair {i:02d} - Dice: {dice:.4f}"
        )


average_dice = sum(scores) / len(scores)

print("\n=== REAL KiTS23 MASS-STYLE EVALUATION ===")
print("Average Dice:", round(average_dice, 4))