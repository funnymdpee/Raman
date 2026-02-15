import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# ==========================================
# 1. 通道注意力机制 (Channel Attention)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 利用1x1卷积代替全连接层以减少参数
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# ==========================================
# 2. 空间注意力机制 (Spatial Attention 1D版)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上做 max 和 avg
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ==========================================
# 3. CBAM 模块 (结合通道+空间)
# ==========================================
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7, enable=True):
        super().__init__()
        self.enable = enable
        if enable:
            self.ca = ChannelAttention(in_planes, ratio)
            self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        if not self.enable:
            return x
        out = x * self.ca(x)
        return out * self.sa(out)



# ==========================================
# 4. 改进的卷积块：ResBlock + CBAM
# ==========================================
class ResCBAMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2

        # 两个卷积层
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # 注意力模块
        self.cbam = CBAM(out_ch)

        # 如果输入输出通道不一致，需要用1x1卷积调整shortcut
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 加入注意力机制
        out = self.cbam(out)

        # 残差连接
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ==========================================
# 5. ASPP 模块 (用于 Bottleneck，扩大感受野)
# ==========================================
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 不同的空洞率
        dilations = [1, 2, 4, 8]
        self.aspp_blocks = nn.ModuleList()

        for d in dilations:
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ))

        # 全局池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

        # 融合层
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp_blocks[0](x)
        x2 = self.aspp_blocks[1](x)
        x3 = self.aspp_blocks[2](x)
        x4 = self.aspp_blocks[3](x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x.size(2), mode='linear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv_fusion(x)
        return x


# ==========================================
# 6. 最终的增强版 UNet (Res-Attention-UNet)
# ==========================================

# 下采样辅助类
class DownBlock_ResCBAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ResCBAMBlock(in_ch, out_ch)
        self.pool = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down


# 上采样辅助类
class UpBlock_ResCBAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        # 注意：这里输入通道减半处理逻辑在 forward 的 cat 之后
        # 但为了保持和 ResBlock 兼容，我们在 Conv 内部处理通道融合
        self.conv = ResCBAMBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 自动裁剪 padding
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, diff))

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RAPUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        # 它可以学习如何对原始输入进行缩放或线性变换
        self.global_shortcut = nn.Conv1d(in_ch, in_ch, kernel_size=1, bias=False)

        # 编码器
        self.inc = ResCBAMBlock(in_ch, base_ch)
        self.down1 = DownBlock_ResCBAM(base_ch, base_ch * 2)
        self.down2 = DownBlock_ResCBAM(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock_ResCBAM(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.bottleneck = ASPP(base_ch * 8, base_ch * 16)

        # 解码器
        self.up1 = UpBlock_ResCBAM(base_ch * 16 + base_ch * 8, base_ch * 8)
        self.up2 = UpBlock_ResCBAM(base_ch * 8 + base_ch * 4, base_ch * 4)
        # 修改处：up3 的输出通道直接调整为 base_ch，准备输出
        self.up3 = UpBlock_ResCBAM(base_ch * 4 + base_ch * 2, base_ch)

        # 删除 self.up4

        # 输出层
        self.outc = nn.Conv1d(base_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)  # [B, 32, L]
        x2, x1_down = self.down1(x1)  # x2:[B, 64, L], pool:[B, 64, L/2]
        x3, x2_down = self.down2(x1_down)  # x3:[B, 128, L/2], pool:[B, 128, L/4]
        x4, x3_down = self.down3(x2_down)  # x4:[B, 256, L/4], pool:[B, 256, L/8]

        x_mid = self.bottleneck(x3_down)  # [B, 512, L/8]

        x = self.up1(x_mid, x4)  # -> [B, 256, L/4]
        x = self.up2(x, x3)  # -> [B, 128, L/2]
        x = self.up3(x, x2)  # -> [B, 32, L]  (这里融合了 x2)


        residual = self.outc(x)
        final_out = self.global_shortcut(x_in) - residual
        return final_out


# 简单测试代码
if __name__ == '__main__':
    model = RAPUNet(in_ch=1, base_ch=32)
    summary(model, input_size=(1, 1, 10000))
    x = torch.randn(2, 1, 10000).to(device='cuda')
    y = model(x)

