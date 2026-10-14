import torch
import torch.nn as nn


class Simple3DUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Segmentation head
        self.decoder = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(8, 1, kernel_size=1)
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__ == "__main__":

    print("=" * 50)
    print("3D SEGMENTATION MODEL TEST")
    print("=" * 50)

    model = Simple3DUNet()

    # Dummy 3D volume
    x = torch.randn(1, 1, 32, 32, 32)

    output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    print("=" * 50)
    print("3D MODEL TEST SUCCESSFUL!")
    print("=" * 50)