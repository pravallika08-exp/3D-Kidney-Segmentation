import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        identity = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)


class MASSStyleModelV2(nn.Module):
    """
    Lightweight MASS-style prototype, not the official MASS/Iris model.
    Uses reference foreground pooling to condition query features.
    """
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock3D(1, 8, stride=2)    # 64 -> 32
        self.enc2 = ConvBlock3D(8, 16, stride=2)   # 32 -> 16
        self.fuse = ConvBlock3D(33, 16)
        self.decode = nn.Sequential(
            nn.Conv3d(16, 8, 3, padding=1),
            nn.InstanceNorm3d(8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8, 1, 1)
        )

    def encode(self, x):
        return self.enc2(self.enc1(x))

    def forward(self, reference_image, reference_mask, query_image):
        ref_feat = self.encode(reference_image)
        qry_feat = self.encode(query_image)

        mask_small = F.interpolate(
            reference_mask.float(), size=ref_feat.shape[2:], mode="nearest"
        )
        weights = mask_small.clamp(0, 1)
        prototype = (ref_feat * weights).sum(dim=(2, 3, 4), keepdim=True)
        prototype = prototype / weights.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)

        proto_map = prototype.expand_as(qry_feat)
        similarity = F.cosine_similarity(
            qry_feat, proto_map, dim=1, eps=1e-6
        ).unsqueeze(1)

        fused = torch.cat([qry_feat, proto_map, similarity], dim=1)
        logits_small = self.decode(self.fuse(fused))
        return F.interpolate(
            logits_small, size=query_image.shape[2:],
            mode="trilinear", align_corners=False
        )


if __name__ == "__main__":
    model = MASSStyleModelV2()
    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 1, 64, 64, 64)
        m = (torch.rand(1, 1, 64, 64, 64) > 0.7).float()
        y = model(x, m, x)
    print("Model test passed.")
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
