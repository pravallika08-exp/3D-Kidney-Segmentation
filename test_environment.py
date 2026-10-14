import torch

print("=" * 50)
print("MASS PROJECT - ENVIRONMENT TEST")
print("=" * 50)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create a small dummy 3D medical volume
x = torch.randn(1, 1, 32, 32, 32)

print("Input shape:", x.shape)

# Simple 3D convolution
layer = torch.nn.Conv3d(
    in_channels=1,
    out_channels=4,
    kernel_size=3,
    padding=1
)

y = layer(x)

print("Output shape:", y.shape)

print("=" * 50)
print("Environment test successful!")
print("=" * 50)