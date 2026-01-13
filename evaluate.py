import torch
import sys
sys.path.append("D:\\Code\\Raman_scatter\\RRCDNet\\s4")
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from utils import *
from dataset_pink import generate_noised_signal

from model.dncnn import DnCNN
from model.dncnn_s4 import DnCNN_S4
from model.dncnn_cnn import DnCNN_CNN
from model.dncnn_dilation import DnCNN_Dilation

from model.rrcdnet import RRCDNet
from model.rrcdnet_left import RRCDNet_Left

from model.rrcdnet_s4 import RRCDNet_S4
from model.rrcdnet_s4_left import RRCDNet_S4_Left

from model.fbnet import BigNet,create_BigNet
from model.lkdnet import create_LKDNet, LKDNet
from model.unet import UNet1D_Denoise

from skimage.metrics import structural_similarity as ssim


#这个脚本的目的，是输出模型在各个评价指标下的参数，因此，需要让这个脚本输出所有的指标的值


def plot_psd_three_signals(x_clean, x_noisy, x_denoised, fs=100e6, nperseg=2048):
    """
    绘制干净信号、加噪信号、降噪信号的功率谱密度（PSD）图。

    参数：
    ----------
    x_clean : array_like
        干净信号
    x_noisy : array_like
        加噪信号
    x_denoised : array_like
        降噪信号
    fs : float, optional
        采样率 (Hz)，默认 1000 Hz
    nperseg : int, optional
        Welch方法中每段的长度，默认 1024

    输出：
    ----------
    绘制 PSD 图（Matplotlib 图像）
    """

    # === 计算功率谱密度 ===
    f_clean, Pxx_clean = signal.welch(x_clean, fs, nperseg=nperseg)
    f_noisy, Pxx_noisy = signal.welch(x_noisy, fs, nperseg=nperseg)
    f_denoised, Pxx_denoised = signal.welch(x_denoised, fs, nperseg=nperseg)

    # === 绘图 ===
    plt.figure(figsize=(10,6))
    plt.semilogy(f_clean, Pxx_clean, label='clean')
    plt.semilogy(f_noisy, Pxx_noisy, label='noisy')
    plt.semilogy(f_denoised, Pxx_denoised, label='denoise')

    plt.title('PSD')
    plt.xlabel('Hz')
    plt.ylabel('V²/Hz')
    plt.legend()
    #plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()




def evaluate(clean,noisy,denoised):

    print("")
    print("average noise:")
    print(average_noise(denoised, 50))

    print("")
    print("mse:")
    print(f"noisy_mse:{mse(noisy, clean)}")
    print(f"denoised_mse:{mse(denoised, clean)}")
    print(f"rmse:{np.sqrt(mse(denoised, clean))}")

    print("")
    print(f"ssim:{ssim(clean, denoised, data_range=1)}")

    print("")
    print("snr:")
    print(f"noise_snr:{compute_snr(clean, noisy)}")
    print(f"denoise_snr:{compute_snr(clean, denoised)}")
    print(f"isnr:{compute_snr(clean, denoised)-compute_snr(clean, noisy)}")

    print("")
    print(f"ρ:{np.corrcoef(clean, denoised)[0,1]}")

    plot_psd_three_signals(clean, noisy, denoised)

    plt.figure(figsize=(4, 3))
    plt.plot(clean, label='clean', alpha=0.7)
    plt.plot(noisy, label='noisy', alpha=0.5)
    plt.plot(denoised, label='denoised (rescaled)', alpha=0.8)

    # 130-160
    plt.xlim(1200, 1400)
    plt.legend()
    plt.show()


def main():
    ckpt = torch.load("checkpoints/UNet1D_Denoise_2026-01-12_17-52-31/best.pt", map_location='cpu')
    #D:\Code\Raman_scatter\RRCDNet\checkpoints\DnCNN_2026-01-11_09-38-15
    #D:\Code\Raman_scatter\RRCDNet\checkpoints\LKDNet_2026-01-11_11-53-50
    model = UNet1D_Denoise().eval()
    model.load_state_dict(ckpt['model_state'])


    set_seed()

    #需要一条干净的数据和加噪后的数据
    clean = generate_clean_signal(maxlen=51)

    #加高斯噪声
    gaussian_noisy = add_gaussian_noise_to_snr(clean, 25)
    gaussian_noisy_scaled, gaussian_mn, gaussian_mx = minmax_scale(gaussian_noisy)
    gaussian_noisy_tensor = torch.tensor(gaussian_noisy_scaled[np.newaxis, np.newaxis, :], dtype=torch.float32)



    with torch.no_grad():
        residual = model(gaussian_noisy_tensor)
        gaussian_denoised = gaussian_noisy_tensor - residual
        gaussian_denoised = gaussian_denoised.cpu().numpy().squeeze()
        gaussian_rescale = gaussian_denoised * (gaussian_mx - gaussian_mn) + gaussian_mn

    evaluate(clean,gaussian_noisy,gaussian_rescale)

    # #加粉色噪声
    # pink_noisy = generate_noised_signal(clean)
    # pink_noisy_scaled, pink_mn, pink_mx = minmax_scale(pink_noisy)
    # pink_noisy_tensor = torch.tensor(pink_noisy_scaled[np.newaxis, np.newaxis, :], dtype=torch.float32)



    # with torch.no_grad():
    #     residual = model(pink_noisy_tensor)
    #     pink_denoised = pink_noisy_tensor - residual
    #     pink_denoised = pink_denoised.cpu().numpy().squeeze()
    #     pink_rescale = pink_denoised * (pink_mx - pink_mn) + pink_mn
    #
    #
    # evaluate(clean,pink_noisy,pink_rescale)




if __name__ == '__main__':
    main()