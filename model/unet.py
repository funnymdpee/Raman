import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- 基本卷积块 ----
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ---- 下采样块 ----
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down  # 返回当前层输出 + 下采样结果


# ---- 上采样块 ----
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch, kernel_size)

    def forward(self, x, skip):
        x = self.up(x)
        # 如果长度不一致，进行裁剪
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, diff))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---- U-Net 主体 ----
class UNet1D_Denoise(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, kernel_size=3):
        super().__init__()
        # 编码器
        self.input = ConvBlock(in_ch, base_ch, kernel_size)
        self.down1 = DownBlock(base_ch, base_ch * 2, kernel_size)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, kernel_size)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8, kernel_size)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8, kernel_size)

        # 解码器
        self.up1 = UpBlock(base_ch * 8 + base_ch * 8, base_ch * 4, kernel_size)
        self.up2 = UpBlock(base_ch * 4 + base_ch * 4, base_ch * 2, kernel_size)
        self.up3 = UpBlock(base_ch * 2 + base_ch * 2, base_ch, kernel_size)

        # 输出层
        self.output = nn.Conv1d(base_ch, in_ch, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        # 编码
        x1 = self.input(x)
        x2, x_down1 = self.down1(x1)
        x3, x_down2 = self.down2(x_down1)
        x4, x_down3 = self.down3(x_down2)

        # Bottleneck
        x_bottleneck = self.bottleneck(x_down3)

        # 解码
        x = self.up1(x_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        # 输出残差形式
        out = self.output(x)
        return out  # 残差去噪思想（输出为去噪信号）
        # 若希望直接输出噪声估计，可改为 return out


# ---- 测试代码 ----
if __name__ == "__main__":
    model = UNet1D_Denoise(in_ch=1, base_ch=64, kernel_size=3)
    x = torch.randn(2, 1, 1024)  # (batch, channel, length)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
