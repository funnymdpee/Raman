import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath


# ------------------- 基础构件 -------------------

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    """1D卷积构建函数"""
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=bias)


use_sync_bn = False


def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True


def get_bn1d(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm1d(channels)


def conv_bn1d(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
    result.add_module('bn', get_bn1d(out_channels))
    return result


def conv_bn_relu1d(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    result = conv_bn1d(in_channels, out_channels, kernel_size, stride, padding, groups, dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result


def fuse_bn(conv, bn):
    """Conv + BN 融合"""
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


# ------------------- 模块定义 -------------------

class ReparamLargeKernelConv1d(nn.Module):
    """大卷积核 + 小卷积核 reparam 结构 (1D版)"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, small_kernel, small_kernel_merged=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2

        if small_kernel_merged:
            self.lkb_reparam = get_conv1d(in_channels, out_channels, kernel_size, stride, padding, 1, groups, bias=True)
        else:
            self.lkb_origin = conv_bn1d(in_channels, out_channels, kernel_size, stride, padding, groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size
                self.small_conv = conv_bn1d(in_channels, out_channels, small_kernel,
                                            stride, small_kernel // 2, groups, 1)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):
            return self.lkb_reparam(x)
        out = self.lkb_origin(x)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(x)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            # 修复：1D卷积只需2个pad参数
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 2)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv1d(self.lkb_origin.conv.in_channels,
                                      self.lkb_origin.conv.out_channels,
                                      self.lkb_origin.conv.kernel_size,
                                      self.lkb_origin.conv.stride,
                                      self.lkb_origin.conv.padding,
                                      self.lkb_origin.conv.dilation,
                                      self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        del self.lkb_origin
        if hasattr(self, 'small_conv'):
            del self.small_conv


class ConvFFN1d(nn.Module):
    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.preffn_bn = get_bn1d(in_channels)
        self.pw1 = conv_bn1d(in_channels, internal_channels, 1, 1, 0, 1)
        self.pw2 = conv_bn1d(internal_channels, out_channels, 1, 1, 0, 1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKBlock1d(nn.Module):
    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super().__init__()
        self.pw1 = conv_bn_relu1d(in_channels, dw_channels, 1, 1, 0, 1)
        self.large_kernel = ReparamLargeKernelConv1d(dw_channels, dw_channels, block_lk_size, 1, dw_channels,
                                                     small_kernel, small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.pw2 = conv_bn1d(dw_channels, in_channels, 1, 1, 0, 1)
        self.prelkb_bn = get_bn1d(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKNetStage1d(nn.Module):
    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 small_kernel, dw_ratio=1, ffn_ratio=4,
                 use_checkpoint=False, small_kernel_merged=False,
                 norm_intermediate_features=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []
        for i in range(num_blocks):
            dp_rate = drop_path[i] if isinstance(drop_path, list) else drop_path
            replk = RepLKBlock1d(channels, int(channels * dw_ratio), stage_lk_size, small_kernel, dp_rate, small_kernel_merged)
            ffn = ConvFFN1d(channels, int(channels * ffn_ratio), channels, dp_rate)
            blks.extend([replk, ffn])
        self.blocks = nn.ModuleList(blks)
        self.norm = get_bn1d(channels) if norm_intermediate_features else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x


# ------------------- BigNet 主体 -------------------

class BigNet(nn.Module):
    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=1, out_channels=10000,num_classes=None, out_indices=None,
                 use_checkpoint=False, small_kernel_merged=False, use_sync_bn=True,
                 norm_intermediate_features=False):
        super().__init__()

        if use_sync_bn:
            enable_sync_bn()

        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        base_width = channels[0]
        self.num_stages = len(layers)

        # Stem
        self.stem = nn.ModuleList([
            conv_bn_relu1d(in_channels, base_width, 3, 2, 1, 1),
            conv_bn_relu1d(base_width, base_width, 3, 1, 1, base_width),
            conv_bn_relu1d(base_width, base_width, 1, 1, 0, 1),
            conv_bn_relu1d(base_width, base_width, 3, 2, 1, base_width)
        ])

        # DropPath设置
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        # Stages
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i in range(self.num_stages):
            stage = RepLKNetStage1d(channels[i], layers[i], large_kernel_sizes[i],
                                    dpr[sum(layers[:i]):sum(layers[:i + 1])],
                                    small_kernel, dw_ratio, ffn_ratio,
                                    use_checkpoint, small_kernel_merged,
                                    norm_intermediate_features)
            self.stages.append(stage)
            if i < self.num_stages - 1:
                transition = nn.Sequential(
                    conv_bn_relu1d(channels[i], channels[i + 1], 1, 1, 0, 1),
                    conv_bn_relu1d(channels[i + 1], channels[i + 1], 3, 1, 1, channels[i + 1])
                )
                self.transitions.append(transition)

        self.norm = get_bn1d(channels[-1])

        # ✅ 分类任务：平均池化 + 线性层
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(channels[-1], num_classes)
            self.output_conv = None
        # ✅ 序列重建任务：输出卷积层
        else:
            self.output_conv = nn.Conv1d(channels[-1], in_channels, kernel_size=1)
            self.avgpool = None
            self.head = None

    def forward_features(self, x):
        for layer in self.stem:
            x = checkpoint.checkpoint(layer, x) if self.use_checkpoint else layer(x)

        if self.out_indices is None:
            for i in range(self.num_stages):
                x = self.stages[i](x)
                if i < self.num_stages - 1:
                    x = self.transitions[i](x)
            return x
        else:
            outs = []
            for i in range(self.num_stages):
                x = self.stages[i](x)
                if i in self.out_indices:
                    outs.append(self.stages[i].norm(x))
                if i < self.num_stages - 1:
                    x = self.transitions[i](x)
            return outs
        
    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        # ✅ 分类模式
        if self.head is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
        # ✅ 序列重建模式
        elif self.output_conv is not None:
            x = self.output_conv(x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    def deep_fuse_BN(self):
        """修正后的BN融合 (使用BatchNorm1d)"""
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if len(m) not in [2, 3]:
                continue
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm1d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = get_conv1d(conv.in_channels, conv.out_channels, conv.kernel_size,
                                        conv.stride, conv.padding, conv.dilation, conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()


# ------------------- 工厂函数 -------------------

def create_BigNet(drop_path_rate=0.3, use_checkpoint=None, small_kernel_merged=False):
    return BigNet(large_kernel_sizes=[31, 29, 27, 13],
                  layers=[2, 2, 18, 2],
                  channels=[64, 64, 64, 64],
                  drop_path_rate=drop_path_rate,
                  small_kernel=5,
                  use_checkpoint=use_checkpoint,
                  small_kernel_merged=small_kernel_merged)


# ------------------- 测试 -------------------
if __name__ == "__main__":
    model = create_BigNet()
    model.eval()
    print('------------------- training-time model -------------')
    print(model)
    x = torch.randn(2, 3, 10000)
    origin_y = model(x)
    model.structural_reparam()
    print('------------------- after re-param -------------')
    print(model)
    reparam_y = model(x)
    print('------------------- the difference is ------------------------')
    print((origin_y - reparam_y).abs().sum())
