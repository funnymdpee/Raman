import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath
from torchinfo import summary


# ================= 基础构建模块 =================

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
    """卷积 + BN + ReLU"""
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
    """卷积 + BN（无激活）"""
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm1d(out_channels)
    )


# ================= 核心模块 =================

class LargeKernelConv(nn.Module):
    """大卷积核深度卷积模块（Depthwise）"""

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.depthwise_conv = conv_bn(channels, channels, kernel_size, groups=channels)

    def forward(self, x):
        return self.depthwise_conv(x)


class RepLKBlock(nn.Module):
    """RepLK 基础块：PW → DW(大核) → PW + 残差"""

    def __init__(self, in_channels, expand_ratio, kernel_size, drop_path=0.):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)

        self.norm = nn.BatchNorm1d(in_channels)
        self.pw_expand = conv_bn_relu(in_channels, hidden_channels, kernel_size=1)
        self.large_kernel = LargeKernelConv(hidden_channels, kernel_size)
        self.activation = nn.ReLU(inplace=True)
        self.pw_project = conv_bn(hidden_channels, in_channels, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.large_kernel(x)
        x = self.activation(x)
        x = self.pw_project(x)
        return shortcut + self.drop_path(x)


class ConvFFN(nn.Module):
    """卷积前馈网络（类似 Transformer FFN）"""

    def __init__(self, in_channels, ffn_ratio=4, drop_path=0.):
        super().__init__()
        hidden_channels = int(in_channels * ffn_ratio)

        self.norm = nn.BatchNorm1d(in_channels)
        self.pw1 = conv_bn(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.pw2 = conv_bn(hidden_channels, in_channels, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.pw1(x)
        x = self.activation(x)
        x = self.pw2(x)
        return shortcut + self.drop_path(x)


class Stage(nn.Module):
    """网络阶段：多个 RepLKBlock + FFN 的堆叠"""

    def __init__(self, channels, num_blocks, kernel_size, drop_path_rates,
                 expand_ratio=1.0, ffn_ratio=4.0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        blocks = []
        for i in range(num_blocks):
            dp = drop_path_rates[i] if isinstance(drop_path_rates, list) else drop_path_rates
            blocks.append(RepLKBlock(channels, expand_ratio, kernel_size, dp))
            blocks.append(ConvFFN(channels, ffn_ratio, dp))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


# ================= 主网络 =================

class LKDNet(nn.Module):
    """大卷积核降噪网络（简化版）"""

    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            base_channels=64,
            num_stages=2,
            stage_blocks=[2, 2],
            kernel_sizes=[51, 47],
            channels=[64, 128],
            expand_ratio=1.0,
            ffn_ratio=4.0,
            drop_path_rate=0.3,
            use_checkpoint=False
    ):
        super().__init__()
        assert num_stages == len(stage_blocks) == len(kernel_sizes) == len(channels)

        # Stem: 初始特征提取
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, base_channels, 3, stride=2, padding=1),
            conv_bn_relu(base_channels, base_channels, 3, padding=1, groups=base_channels),
            conv_bn_relu(base_channels, base_channels, 1),
            conv_bn_relu(base_channels, base_channels, 3, padding=1, groups=base_channels)
        )

        # 计算每个block的DropPath rate（线性递增）
        total_blocks = sum(stage_blocks)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # 构建各个Stage
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        current_dp_idx = 0
        for i in range(num_stages):
            # Stage
            stage = Stage(
                channels[i],
                stage_blocks[i],
                kernel_sizes[i],
                dp_rates[current_dp_idx:current_dp_idx + stage_blocks[i]],
                expand_ratio,
                ffn_ratio,
                use_checkpoint
            )
            self.stages.append(stage)
            current_dp_idx += stage_blocks[i]

            # Transition（Stage间的通道变换，除最后一个Stage）
            if i < num_stages - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[i], channels[i + 1], 1),
                    conv_bn_relu(channels[i + 1], channels[i + 1], 3, groups=channels[i + 1])
                )
                self.transitions.append(transition)

        # 输出头
        self.output_conv = conv_bn(channels[-1], out_channels, 1)
        self.upsample = nn.ConvTranspose1d(
            out_channels, out_channels,
            kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        # Output
        x = self.output_conv(x)
        x = self.upsample(x)
        return x


# ================= 工厂函数 =================

def create_LKDNet(drop_path_rate=0.3, use_checkpoint=False):
    """创建标准 LKDNet 模型"""
    return LKDNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        num_stages=2,
        stage_blocks=[5, 5],
        kernel_sizes=[3, 3],
        channels=[64, 64],
        expand_ratio=1.0,
        ffn_ratio=4.0,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint
    )


# ================= 测试代码 =================

if __name__ == "__main__":
    model = create_LKDNet()
    model.eval()

    print("=" * 60)
    print("模型结构：")
    print("=" * 60)
    print(model)

    # 测试前向传播
    x = torch.randn(2, 1, 10000)
    with torch.no_grad():
        y = model(x)

    print("\n" + "=" * 60)
    print(f"输入维度: {x.shape}")
    print(f"输出维度: {y.shape}")
    print("=" * 60)

    # 详细结构信息
    summary(model, input_size=(1, 1, 10000))
