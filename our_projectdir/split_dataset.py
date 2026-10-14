from pathlib import Path
import shutil


SOURCE = Path("data/processed/samples")

TRAIN_DIR = Path("data/processed/train")
VAL_DIR = Path("data/processed/val")

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 50)
print("CREATING TRAIN / VALIDATION SPLIT")
print("=" * 50)


# We currently have 3 augmented samples.
# Use samples 0 and 1 for training,
# sample 2 for validation.

for index in [0, 1]:

    shutil.copy(
        SOURCE / f"image_{index}.npy",
        TRAIN_DIR / f"image_{index}.npy"
    )

    shutil.copy(
        SOURCE / f"mask_{index}.npy",
        TRAIN_DIR / f"mask_{index}.npy"
    )


# Validation sample

index = 2

shutil.copy(
    SOURCE / f"image_{index}.npy",
    VAL_DIR / f"image_{index}.npy"
)

shutil.copy(
    SOURCE / f"mask_{index}.npy",
    VAL_DIR / f"mask_{index}.npy"
)


print("Training samples: 2")
print("Validation samples: 1")

print()
print("Training directory:")
print(TRAIN_DIR)

print()
print("Validation directory:")
print(VAL_DIR)

print("=" * 50)
print("DATASET SPLIT COMPLETE!")
print("=" * 50)