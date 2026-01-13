# utils.py
import numpy as np
import torch
import random
import pywt

### 小波分解
def wd(signal, wavelet='db4', level=3):

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 估计噪声σ：使用最细尺度系数
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # 通用阈值 VisuShrink
    thr = sigma * np.sqrt(2 * np.log(len(signal)))

    # 阈值处理细节系数
    coeffs[1:] = [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]

    # 重构
    return pywt.waverec(coeffs, wavelet)
    # ------------------- WD3 / WD4 示例 -------------------

    # # 创建一个简单的信号
    # x = np.linspace(0, 1, 512)
    # signal = np.sin(2 * np.pi * 7 * x) + 0.5 * np.sin(2 * np.pi * 70 * x)
    #
    # # WD3 (Daubechies-3)
    # denoised_wd3 = wd(signal, wavelet='db3', level=3)
    #
    # # WD4 (Daubechies-4)
    # denoised_wd4 = wd(signal, wavelet='db4', level=4)

def set_seed(seed=42):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_clean_signal(length=10000,maxlen = 51):
    """
    Generate synthetic 'clean' signal per description:
    random amplitude in [0,1], random block length 1..50 repeated to fill 'length',
    simulating decay + variable temperature region.
    """
    signal = np.zeros(length, dtype=np.float32)
    pos = 0
    while pos < length:
        amp = np.random.rand()  # [0,1]
        seg_len = np.random.randint(1, maxlen)  # 1..50
        end = min(length, pos + seg_len)
        signal[pos:end] = amp
        pos = end
    # Optionally apply a smooth decay or convolution to be more realistic
    # e.g., convolve with small kernel:
    kernel = np.exp(-np.arange(0,20)/5.0)
    signal = np.convolve(signal, kernel, mode='same').astype(np.float32)
    return signal

def add_gaussian_noise_to_snr(clean, snr_db):
    """
    clean: numpy array signal (N,)
    return noisy signal at desired SNR in dB
    SNR dB = 10*log10( P_signal / P_noise )
    """
    power_signal = np.mean(clean**2)
    snr_linear = 10**(snr_db / 10.0)
    power_noise = power_signal / snr_linear
    noise = np.random.normal(scale=np.sqrt(power_noise), size=clean.shape)
    return clean + noise

def minmax_scale(sample):
    """Scale sample to [0,1]. sample is numpy array"""
    mn = sample.min()
    mx = sample.max()
    if mx - mn < 1e-8:
        return np.zeros_like(sample), mn, mx
    scaled = (sample - mn) / (mx - mn)
    return scaled, mn, mx

def apply_scale_with_params(sample, mn, mx):
    if mx - mn < 1e-8:
        return np.zeros_like(sample)
    return (sample - mn) / (mx - mn)

def average_noise(v_prime, l):
    v_prime = np.asarray(v_prime)
    L = len(v_prime)
    N = L - l + 1
    if N <= 0:
        raise ValueError("窗口长度 l 必须小于或等于数据长度")

    # 计算每个窗口的均值：卷积实现 (1/l)
    window = np.ones(l) / l
    window_means = np.convolve(v_prime, window, mode='valid')

    # 套用公式：平均绝对值
    avg_noise = np.mean(np.abs(window_means))
    return avg_noise

def mse(raw,pre):
    return np.mean((raw - pre)**2)



def compute_snr(clean, denoised):
    """
    计算降噪信号的信噪比 (SNR)
    :param clean: ndarray, 干净信号
    :param denoised: ndarray, 降噪后的信号
    :return: float, SNR in dB
    """
    clean = np.asarray(clean)
    denoised = np.asarray(denoised)

    # 保证信号长度一致
    if clean.shape != denoised.shape:
        raise ValueError("clean 和 denoised 信号长度不一致！")

    noise = clean - denoised  # 残余噪声
    snr = 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))
    return snr


