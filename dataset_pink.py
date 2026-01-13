import numpy as np
from torch.utils.data import Dataset
# 假设 utils 中还有 minmax_scale 等其他必要函数，保留引用
from utils import minmax_scale, apply_scale_with_params,generate_clean_signal


def power_law_noise(n_samples, alpha=1.0):
    """
    生成具有 1/f^alpha 功率谱密度的有色噪声。
    alpha = 1.0 -> 粉红噪声 (Pink Noise), 适合测试 S4 的长距离捕捉能力
    alpha = 0.0 -> 白噪声 (White Noise)
    """
    # 1. 生成频域成分
    frequencies = np.fft.rfftfreq(n_samples)

    # 避免直流分量 (f=0) 除以 0
    if frequencies[0] == 0:
        frequencies[0] = 1e-10

    # 2. 生成幅度谱 (1/f^alpha) 和 随机相位
    scale = np.sqrt(np.abs(frequencies) ** (-alpha))
    phase = np.random.uniform(0, 2 * np.pi, len(frequencies))

    # 3. 合成复数频谱并转换回时域
    spectrum = scale * np.exp(1j * phase)
    noise = np.fft.irfft(spectrum, n=n_samples)

    # 4. 标准化 (0均值, 1标准差)，保证后续混合比例准确
    noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-10)
    return noise.astype(np.float32)


def generate_noised_signal(clean,length=10000,pink_noise_ratio = 1,snr_low=20,snr_high=27):

    # 2. 生成混合噪声基底 (Unit Variance)
    # 生成粉红噪声 (长距离相关)
    noise_pink = power_law_noise(length, alpha=1.0)
    # 生成白噪声 (局部随机)
    noise_white = np.random.randn(length).astype(np.float32)

    # 混合两者:
    # 简单的加权求和会导致方差变化，这里不做严格功率归一化，
    # 只要混合后整体再次标准化即可，方便后续计算 SNR。
    mixed_noise = pink_noise_ratio * noise_pink + (1 - pink_noise_ratio) * noise_white
    # 再次强制归一化混合后的噪声 (mean=0, std=1)
    mixed_noise = (mixed_noise - np.mean(mixed_noise)) / (np.std(mixed_noise) + 1e-10)

    # 3. 根据 SNR 计算需要的噪声幅度并叠加
    target_snr = np.random.uniform(snr_low, snr_high)

    # 计算信号功率 Ps
    signal_power = np.mean(clean ** 2)
    # 根据 SNR 公式: SNR = 10 * log10(Ps / Pn) -> Pn = Ps / 10^(SNR/10)
    noise_power_target = signal_power / (10 ** (target_snr / 10))
    # 因为 mixed_noise 的方差已经是 1 (即功率为 1)，我们只需要乘以 sqrt(目标功率)
    noise_scale = np.sqrt(noise_power_target)

    # 最终的含噪信号
    noise = clean + mixed_noise * noise_scale

    return noise


class RamanSynthDataset(Dataset):
    def __init__(self, n_samples=5000, length=10000, snr_range=(20, 37),
                 pink_noise_ratio=0.7, train=True):
        """
        参数:
        - pink_noise_ratio: 混合噪声中粉红噪声的占比 (0~1)。
          0.7 表示 70% 的能量来自粉红噪声，30% 来自白噪声。
          建议设高一点 (0.5 - 0.8) 以突出 S4 优势。
        """
        self.n = n_samples
        self.length = length
        self.snr_low, self.snr_high = snr_range
        self.data = []

        print(f"Generating Dataset: {n_samples} samples, Pink/White Ratio: {pink_noise_ratio}")

        # 预生成数据集
        for _ in range(n_samples):
            # 1. 生成干净信号
            clean = generate_clean_signal(length=length)

            noisy = generate_noised_signal(clean)

            # 4. 归一化处理 (保持你原有的逻辑)
            # Normalize noisy to [0,1] and scale clean using same params
            noisy_scaled, mn, mx = minmax_scale(noisy)
            clean_scaled = apply_scale_with_params(clean, mn, mx)

            self.data.append((noisy_scaled.astype(np.float32), clean_scaled.astype(np.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy, clean = self.data[idx]
        # convert to shape (1, L) for 1D conv
        return noisy[np.newaxis, :], clean[np.newaxis, :]
