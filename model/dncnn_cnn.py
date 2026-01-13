import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """基础卷积模块：Conv + BN + ReLU（可选）"""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, use_bn=True, use_relu=True):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_ch))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DnCNN_CNN(nn.Module):
    """
    与 LeftSubNet 结构完全对齐的纯卷积版 DnCNN。
    无 S4，无 dilated conv。
    """

    def __init__(self, in_ch=1, base_ch=64, kernel=3):
        super().__init__()
        layers = []

        for i in range(1, 18):  # 1..17
            if i == 1:
                # 输入层
                layers.append(ConvBlock(in_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))
            elif 2 <= i <= 8:
                # 原 S4Block 区域 → 普通 ConvBlock
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))
            elif i == 9:
                # 过渡层
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))
            elif 10 <= i <= 15:
                # 第二段原 S4Block 区域
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))
            elif i == 16:
                # 输出前层
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))
            elif i == 17:
                # 输出层（无 BN / ReLU）
                layers.append(ConvBlock(base_ch, 1, kernel_size=kernel, padding=1,
                                        use_bn=False, use_relu=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
