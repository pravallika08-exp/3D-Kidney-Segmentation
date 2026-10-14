from pathlib import Path
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from mass_model_v2 import MASSStyleModelV2

ROOT = Path("data/real/training_reference_query_v2")
OUT = Path("data/real")
SEED = 42
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")


class PairDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = Path(root)
        self.items = sorted(p for p in self.root.glob("pair_*") if p.is_dir())
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        arrays = [
            np.load(p / "reference_image.npy").astype(np.float32),
            np.load(p / "reference_mask.npy").astype(np.float32),
            np.load(p / "query_image.npy").astype(np.float32),
            np.load(p / "query_mask.npy").astype(np.float32),
        ]
        ri, rm, qi, qm = arrays

        # Apply spatial transforms consistently to each image/mask pair.
        if self.augment:
            for axis in (0, 1, 2):
                if random.random() < 0.5:
                    ri, rm = np.flip(ri, axis).copy(), np.flip(rm, axis).copy()
                    qi, qm = np.flip(qi, axis).copy(), np.flip(qm, axis).copy()
            k = random.randrange(4) if random.random() < 0.5 else 0
            if k:
                ri, rm = np.rot90(ri, k, axes=(1, 2)).copy(), np.rot90(rm, k, axes=(1, 2)).copy()
                qi, qm = np.rot90(qi, k, axes=(1, 2)).copy(), np.rot90(qm, k, axes=(1, 2)).copy()
            for image_name in ("ri", "qi"):
                scale = random.uniform(0.9, 1.1)
                shift = random.uniform(-0.05, 0.05)
                if image_name == "ri":
                    ri = np.clip(ri * scale + shift, 0, 1)
                else:
                    qi = np.clip(qi * scale + shift, 0, 1)

        tensors = [
            torch.from_numpy(a.copy()).float().unsqueeze(0)
            for a in (ri, rm, qi, qm)
        ]
        return tuple(tensors)


def combined_loss(logits, target):
    target = target.float()
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = prob * target + (1 - prob) * (1 - target)
    focal = (((1 - pt) ** 2) * bce).mean()
    dims = (1, 2, 3, 4)
    intersection = (prob * target).sum(dims)
    dice = (2 * intersection + 1) / (prob.sum(dims) + target.sum(dims) + 1)
    return focal + (1 - dice).mean()


def evaluate(model, loader):
    model.eval()
    losses, dices = [], []
    with torch.no_grad():
        for ri, rm, qi, qm in loader:
            logits = model(ri, rm, qi)
            losses.append(combined_loss(logits, qm).item())
            pred = (torch.sigmoid(logits) >= 0.5).float()
            dims = (1, 2, 3, 4)
            inter = (pred * qm).sum(dims)
            score = (2 * inter + 1) / (pred.sum(dims) + qm.sum(dims) + 1)
            dices.extend(score.cpu().tolist())
    return float(np.mean(losses)), float(np.mean(dices))


def main():
    full = PairDataset(ROOT, augment=False)
    if len(full) < 10:
        raise RuntimeError(f"Expected at least 10 pairs; found {len(full)}")

    n_val = max(5, int(0.2 * len(full)))
    n_train = len(full) - n_val
    train_subset, val_subset = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_ds = PairDataset(ROOT, augment=True)
    # Subsets preserve the indices chosen by random_split.
    train_subset.dataset = train_ds

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MASSStyleModelV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    history_path = OUT / "training_history_v2.csv"
    best_path = OUT / "best_mass_kits23_model_v2.pth"
    best_dice = -1.0

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "lr"])

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_losses = []
            for ri, rm, qi, qm in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(ri, rm, qi)
                loss = combined_loss(logits, qm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            val_loss, val_dice = evaluate(model, val_loader)
            scheduler.step()
            train_loss = float(np.mean(train_losses))
            lr_now = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, train_loss, val_loss, val_dice, lr_now])
            f.flush()

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_path)

            print(
                f"Epoch {epoch:02d}/{EPOCHS} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_dice={val_dice:.4f} lr={lr_now:.6g}"
            )

    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"History CSV: {history_path}")
    print("Validation is a pair-level split, not an independent patient-level split.")


if __name__ == "__main__":
    main()
