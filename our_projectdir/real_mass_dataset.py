from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class RealMASSDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.pairs = sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("pair_")
        )
        print(f"Found {len(self.pairs)} pairs in {self.root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair_dir = self.root / self.pairs[index]
        names = (
            "reference_image",
            "reference_mask",
            "query_image",
            "query_mask",
        )
        tensors = []
        for name in names:
            array = np.load(pair_dir / f"{name}.npy")
            tensors.append(
                torch.from_numpy(array).float().unsqueeze(0)
            )
        return tuple(tensors)
