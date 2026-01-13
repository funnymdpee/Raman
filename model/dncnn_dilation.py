import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """可配置的卷积模块"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1,
                 use_bn=True, use_relu=True):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size,
                            padding=padding, dilation=dilation, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_ch))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DnCNN_Dilation(nn.Module):
    """纯 CNN 版本，与 LeftSubNet 结构完全对齐"""
    def __init__(self, in_ch=1, base_ch=64, kernel=3):
        super().__init__()
        layers = []

        for i in range(1, 18):  # 1~17 层
            if i == 1:
                # 输入层 Conv + BN + ReLU
                layers.append(ConvBlock(in_ch, base_ch, kernel_size=kernel,
                                        padding=1, use_bn=True, use_relu=True))
            elif 2 <= i <= 8:
                # Dilated Conv + ReLU（无BN）
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel,
                                        dilation=2, padding=2, use_bn=False, use_relu=True))
            elif i == 9:
                # 普通 Conv + BN + ReLU
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel,
                                        padding=1, use_bn=True, use_relu=True))
            elif 10 <= i <= 15:
                # 第二段 Dilated Conv + ReLU
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel,
                                        dilation=2, padding=2, use_bn=False, use_relu=True))
            elif i == 16:
                # Conv + BN + ReLU
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel,
                                        padding=1, use_bn=True, use_relu=True))
            elif i == 17:
                # 输出层 Conv（无 BN/ReLU）
                layers.append(ConvBlock(base_ch, 1, kernel_size=kernel,
                                        padding=1, use_bn=False, use_relu=False))
            else:
                # 安全兜底（不触发）
                layers.append(ConvBlock(base_ch, base_ch))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
