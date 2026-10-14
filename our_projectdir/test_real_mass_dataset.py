from pathlib import Path
from torch.utils.data import DataLoader

from real_mass_dataset import RealMASSDataset


root = Path(
    "data/real/reference_query_dataset"
)

dataset = RealMASSDataset(root)

print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

reference_image, reference_mask, query_image, query_mask = next(
    iter(loader)
)

print("Reference image:", reference_image.shape)
print("Reference mask :", reference_mask.shape)
print("Query image    :", query_image.shape)
print("Query mask     :", query_mask.shape)