import torch
import torch.nn as nn


class MASSStyleModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Reference-mask encoder
        self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Query segmentation decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(48, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(8, 1, kernel_size=1)
        )

    def forward(
        self,
        reference_image,
        reference_mask,
        query_image
    ):

        # Encode reference image
        reference_features = self.image_encoder(
            reference_image
        )

        # Encode reference mask
        mask_features = self.mask_encoder(
            reference_mask
        )

        # Combine reference information
        reference_task = torch.cat(
            [
                reference_features,
                mask_features
            ],
            dim=1
        )

        # Encode query image
        query_features = self.image_encoder(
            query_image
        )

        # Combine query + reference task
        combined = torch.cat(
            [
                query_features,
                reference_task
            ],
            dim=1
        )

        # Predict query segmentation
        output = self.decoder(combined)

        return output


if __name__ == "__main__":

    print("=" * 50)
    print("MASS-STYLE MODEL TEST")
    print("=" * 50)

    model = MASSStyleModel()

    reference_image = torch.randn(
        1, 1, 32, 32, 32
    )

    reference_mask = torch.randint(
        0,
        2,
        (1, 1, 32, 32, 32)
    ).float()

    query_image = torch.randn(
        1, 1, 32, 32, 32
    )

    output = model(
        reference_image,
        reference_mask,
        query_image
    )

    print("Reference image:", reference_image.shape)
    print("Reference mask :", reference_mask.shape)
    print("Query image    :", query_image.shape)
    print("Output         :", output.shape)

    print("=" * 50)
    print("MASS-STYLE MODEL TEST SUCCESSFUL!")
    print("=" * 50)