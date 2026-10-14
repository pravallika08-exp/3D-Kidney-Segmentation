import csv
from pathlib import Path


OUTPUT = Path("data/processed/training_history.csv")


# Example history file structure.
# The values will be replaced by the actual
# training results when we connect this
# with the final training script.

epochs = list(range(1, 11))

rows = []

for epoch in epochs:
    rows.append({
        "epoch": epoch,
        "train_loss": "",
        "val_dice": ""
    })


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_dice"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print("=" * 50)
print("TRAINING HISTORY FILE CREATED")
print("=" * 50)

print("Saved to:")
print(OUTPUT)

print("=" * 50)