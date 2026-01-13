# rrcdnet.py
import torch
import torch.nn as nn
#from s4.models.s4.s4d import S4Block
from s4.models.s4.s4d import ResidualS4Block as S4Block

class ConvBlock(nn.Module):
    """Conv + BN + ReLU (optional BN/ReLU controlled by flags)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1, use_bn=True, use_relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn:
            self.bn = nn.BatchNorm1d(out_ch)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

class RightSubNet(nn.Module):
    """Right network: conventional conv stack, depth=17 as described."""
    def __init__(self, in_ch=1, base_ch=64, kernel=3):
        super().__init__()
        layers = []
        # Layer 1: 1 -> 64 conv + BN + ReLU
        layers.append(ConvBlock(in_ch, base_ch, kernel_size=kernel, padding=1, use_bn=True, use_relu=True))
        # Layers 2-16: 64 -> 64 Conv + BN + ReLU (except final of these has no ReLU if described)
        for _ in range(2, 17):
            # We'll keep layers 2-16 as Conv+BN+ReLU, and final layer (17th) separately
            layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1, use_bn=True, use_relu=True))
        # Replace last layer (17th) with Conv -> out 1, no BN/ReLU
        layers[-1] = ConvBlock(base_ch, 1, kernel_size=kernel, padding=1, use_bn=False, use_relu=False)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LeftSubNet(nn.Module):
    """Left network: uses dilated conv in many layers. Depth=17."""
    def __init__(self, in_ch=1, base_ch=64, kernel=3, dilation=2):
        super().__init__()
        layers = []
        # We'll follow the description: layers 1,9,16 are Conv+BN+ReLU; 2-8 and 10-15 are dilated Conv+ReLU
        # Use 1-indexing for clarity.
        for i in range(1, 18):  # 1..17
            if i == 1:
                layers.append(ConvBlock(in_ch, base_ch, kernel_size=kernel, padding=1, use_bn=True, use_relu=True))
            elif i in (9, 16):
                # non-dilated conv + BN + ReLU (description says these have BN+ReLU)
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1, use_bn=True, use_relu=True))
            elif 2 <= i <= 8 or 10 <= i <= 15:
                # dilated conv + ReLU (BN likely not used here per description; but they said BN layers exist to ensure consistency,
                # to keep faithful we will include BN if you prefer; here keep BN=False to match "Dilated conv + ReLU")
                layers.append(S4Block(d_model=base_ch))
            elif i == 17:
                # final layer only Conv (no BN/ReLU)
                layers.append(ConvBlock(base_ch, 1, kernel_size=kernel, padding=1, use_bn=False, use_relu=False))
            else:
                # default fallback
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1, use_bn=True, use_relu=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RRCDNet_S4(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.left = LeftSubNet(in_ch=in_ch, base_ch=base_ch)
        self.right = RightSubNet(in_ch=in_ch, base_ch=base_ch)
        # After concatenation, optionally add a small fusion conv to predict residual
        # The paper seems to concat outputs and then probably process; here we simply concat and fuse
        self.fuse = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, y):
        # y: (B, 1, L)
        out_l = self.left(y)    # (B,1,L)
        out_r = self.right(y)   # (B,1,L)
        out = torch.cat([out_l, out_r], dim=1)  # (B,2,L)
        residual = self.fuse(out)  # predict noise
        return residual
