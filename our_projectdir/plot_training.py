import csv
import matplotlib.pyplot as plt

CSV_PATH = "data/processed/training_history.csv"

epochs = []
train_loss = []
val_dice = []

with open(CSV_PATH, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_dice.append(float(row["val_dice"]))


# -----------------------------
# Training loss
# -----------------------------

plt.figure()

plt.plot(
    epochs,
    train_loss,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("MASS-Style Training Loss")

plt.grid(True)

plt.savefig(
    "data/processed/training_loss.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# -----------------------------
# Validation Dice
# -----------------------------

plt.figure()

plt.plot(
    epochs,
    val_dice,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("MASS-Style Validation Dice")

plt.grid(True)

plt.savefig(
    "data/processed/validation_dice.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("=" * 50)
print("TRAINING CURVES CREATED")
print("=" * 50)

print("Saved:")
print("data/processed/training_loss.png")
print("data/processed/validation_dice.png")

print("=" * 50)