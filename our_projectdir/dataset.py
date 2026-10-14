import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalVolumeDataset(Dataset):

    def __init__(self, image_path, mask_path):
        self.image = np.load(image_path).astype(np.float32)
        self.mask = np.load(mask_path).astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):

        image = torch.from_numpy(self.image)
        mask = torch.from_numpy(self.mask)

        # Add channel dimension
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask
        }


if __name__ == "__main__":

    dataset = MedicalVolumeDataset(
        "data/processed/image.npy",
        "data/processed/mask.npy"
    )

    sample = dataset[0]

    print("=" * 50)
    print("DATASET TEST")
    print("=" * 50)

    print("Dataset length:", len(dataset))
    print("Image shape:", sample["image"].shape)
    print("Mask shape:", sample["mask"].shape)
    print("Image dtype:", sample["image"].dtype)
    print("Mask dtype:", sample["mask"].dtype)

    print("=" * 50)
    print("DATASET TEST SUCCESSFUL!")
    print("=" * 50)