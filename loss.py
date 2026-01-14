import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    """
    多分辨率 STFT 损失函数
    计算预测信号和真实信号在多个不同 FFT 窗口下的 
    1. 谱收敛损失 (Spectral Convergence)
    2. 对数幅度损失 (Log Magnitude)
    """
    def __init__(self, 
                 fft_sizes=[1024, 2048, 512], 
                 hop_sizes=[120, 240, 50], 
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window

    def stft(self, x, fft_size, hop_size, win_length):
        """
        计算 STFT。
        x 的形状通常是 [Batch, Channel, Length] 或 [Batch, Length]
        我们需要将其统一为 [Batch, Length] 才能送入 torch.stft
        """
        # 如果输入是 [Batch, 1, Length]，先去掉通道维度
        if x.dim() == 3:
            x = x.squeeze(1) 

        # 定义窗函数 (汉宁窗)
        window = torch.hann_window(win_length).to(x.device)
        
        # 执行短时傅里叶变换
        # return_complex=True 是 PyTorch 1.7+ 的推荐写法
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        
        # 获取幅度谱 (Magnitude)
        # clamp 避免 log(0)
        mag = torch.abs(x_stft)
        return mag

    def forward(self, x_pred, x_true):
        """
        x_pred: 网络预测的去噪信号
        x_true: 干净的 Ground Truth 信号
        """
        loss_spectral = 0.0

        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_pred_mag = self.stft(x_pred, fft_size, hop_size, win_length)
            x_true_mag = self.stft(x_true, fft_size, hop_size, win_length)
            
            # 1. 谱收敛损失 (Spectral Convergence Loss)
            # 衡量整体频谱能量分布的差异
            sc_loss = torch.norm(x_true_mag - x_pred_mag, p="fro") / (torch.norm(x_true_mag, p="fro") + 1e-7)
            
            # 2. 对数幅度损失 (Log Magnitude Loss)
            # 衡量细节差异，取 log 后更关注低能量区域（这对于去噪很重要）
            log_mag_pred = torch.log(x_pred_mag + 1e-7)
            log_mag_true = torch.log(x_true_mag + 1e-7)
            mag_loss = F.l1_loss(log_mag_pred, log_mag_true)
            
            loss_spectral += sc_loss + mag_loss

        # 取平均
        return loss_spectral / len(self.fft_sizes)


class HybridLoss(nn.Module):
    """
    混合损失：时域 L1 + 频域 STFT
    """
    def __init__(self, alpha=1.0):
        super(HybridLoss, self).__init__()
        self.l1 = nn.L1Loss() # 或者 nn.SmoothL1Loss()
        self.stft = MultiResolutionSTFTLoss()
        self.alpha = alpha # 用于平衡两项损失的权重

    def forward(self, pred, target):
        loss_time = self.l1(pred, target)
        loss_freq = self.stft(pred, target)
        # 这里的 alpha 需要根据实际 loss 数值大小调整，通常 STFT loss 会比 L1 大一些
        return loss_time + self.alpha * loss_freq
