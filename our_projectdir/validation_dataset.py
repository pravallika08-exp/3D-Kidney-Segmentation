import numpy as np
import torch
from torch.utils.data import Dataset


class ValidationDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = data_dir

        self.image_path = f"{data_dir}/image_2.npy"
        self.mask_path = f"{data_dir}/mask_2.npy"

        self.image = np.load(
            self.image_path
        ).astype(np.float32)

        self.mask = np.load(
            self.mask_path
        ).astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):

        image = torch.from_numpy(
            self.image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            self.mask
        ).unsqueeze(0)

        return {
            "image": image,
            "mask": mask
        }


if __name__ == "__main__":

    dataset = ValidationDataset(
        "data/processed/val"
    )

    sample = dataset[0]

    print("=" * 50)
    print("VALIDATION DATASET TEST")
    print("=" * 50)

    print("Number of samples:", len(dataset))

    print(
        "Validation image:",
        sample["image"].shape
    )

    print(
        "Validation mask:",
        sample["mask"].shape
    )

    print(
        "Image dtype:",
        sample["image"].dtype
    )

    print(
        "Mask dtype:",
        sample["mask"].dtype
    )

    print("=" * 50)
    print("VALIDATION DATASET TEST SUCCESSFUL!")
    print("=" * 50)