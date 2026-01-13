import torch
import torch.nn as nn

from s4.models.s4.s4d import S4Block
#from s4.models.s4.s4d import ResidualS4Block as S4Block


class ConvBlock(nn.Module):
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


class DnCNN_S4(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, kernel=3):
        super().__init__()
        layers = []

        for i in range(1, 18):  # 1..17 共17层
            if i == 1:
                # 输入层 Conv + BN + ReLU
                layers.append(ConvBlock(in_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))
            elif 2 <= i <= 8:
                # 使用 S4Block 替代 dilated conv
                layers.append(S4Block(d_model=base_ch))
            elif i == 9:
                # Conv + BN + ReLU
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))
            elif 10 <= i <= 15:
                # 第二段 S4Block
                layers.append(S4Block(d_model=base_ch))
            elif i == 16:
                # Conv + BN + ReLU
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))
            elif i == 17:
                # 最后一层输出层（仅 Conv）
                layers.append(ConvBlock(base_ch, 1, kernel_size=kernel, padding=1,
                                        use_bn=False, use_relu=False))
            else:
                # 兜底（其实不会进入）
                layers.append(ConvBlock(base_ch, base_ch, kernel_size=kernel, padding=1,
                                        use_bn=True, use_relu=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

