import torch
from torch.utils.data import DataLoader

from mass_dataset import MASSDataset
from mass_model import MASSStyleModel


print("=" * 50)
print("REFERENCE/QUERY MODEL EVALUATION")
print("=" * 50)


# -----------------------------
# 1. Load dataset
# -----------------------------

dataset = MASSDataset(
    "data/processed/samples"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)


# -----------------------------
# 2. Load trained model
# -----------------------------

model = MASSStyleModel()

model.load_state_dict(
    torch.load(
        "data/processed/mass_reference_query_model.pth",
        map_location="cpu"
    )
)

model.eval()


# -----------------------------
# 3. Evaluate
# -----------------------------

total_dice = 0.0

with torch.no_grad():

    for i, batch in enumerate(loader):

        reference_image = batch["reference_image"]
        reference_mask = batch["reference_mask"]

        query_image = batch["query_image"]
        query_mask = batch["query_mask"]

        output = model(
            reference_image,
            reference_mask,
            query_image
        )

        probability = torch.sigmoid(output)

        prediction = (
            probability > 0.5
        ).float()

        intersection = (
            prediction * query_mask
        ).sum()

        dice = (
            (2.0 * intersection + 1e-8)
            /
            (
                prediction.sum()
                + query_mask.sum()
                + 1e-8
            )
        )

        total_dice += dice.item()

        print(
            f"Sample {i + 1} "
            f"Dice: {dice.item():.4f}"
        )


# -----------------------------
# 4. Average Dice
# -----------------------------

average_dice = total_dice / len(loader)

print()
print(
    f"Average Dice: {average_dice:.4f}"
)

print("=" * 50)
print("REFERENCE/QUERY EVALUATION COMPLETE!")
print("=" * 50)