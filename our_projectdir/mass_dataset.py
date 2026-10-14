import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MASSDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

        self.images = sorted(
            self.data_dir.glob("image_*.npy")
        )

        self.masks = sorted(
            self.data_dir.glob("mask_*.npy")
        )

        assert len(self.images) == len(self.masks)
        assert len(self.images) >= 2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # Current sample becomes query
        query_image = np.load(
            self.images[index]
        ).astype(np.float32)

        query_mask = np.load(
            self.masks[index]
        ).astype(np.float32)

        # Choose a different sample as reference
        reference_index = (index + 1) % len(self.images)

        reference_image = np.load(
            self.images[reference_index]
        ).astype(np.float32)

        reference_mask = np.load(
            self.masks[reference_index]
        ).astype(np.float32)

        # Convert to tensors
        reference_image = torch.from_numpy(
            reference_image
        ).unsqueeze(0)

        reference_mask = torch.from_numpy(
            reference_mask
        ).unsqueeze(0)

        query_image = torch.from_numpy(
            query_image
        ).unsqueeze(0)

        query_mask = torch.from_numpy(
            query_mask
        ).unsqueeze(0)

        return {
            "reference_image": reference_image,
            "reference_mask": reference_mask,
            "query_image": query_image,
            "query_mask": query_mask
        }


if __name__ == "__main__":

    dataset = MASSDataset(
        "data/processed/samples"
    )

    print("=" * 50)
    print("MASS DATASET TEST")
    print("=" * 50)

    print("Number of samples:", len(dataset))

    sample = dataset[0]

    print(
        "Reference image:",
        sample["reference_image"].shape
    )

    print(
        "Reference mask:",
        sample["reference_mask"].shape
    )

    print(
        "Query image:",
        sample["query_image"].shape
    )

    print(
        "Query mask:",
        sample["query_mask"].shape
    )

    print("=" * 50)
    print("MASS DATASET TEST SUCCESSFUL!")
    print("=" * 50)