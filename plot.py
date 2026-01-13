import torch
import sys
sys.path.append("D:\\Code\\Raman_scatter\\RRCDNet\\s4")
import matplotlib.pyplot as plt
import numpy as np
from dataset import generate_clean_signal
from utils import *

from model.dncnn import DnCNN
from model.dncnn_s4 import DnCNN_S4
from model.dncnn_cnn import DnCNN_CNN
from model.dncnn_dilation import DnCNN_Dilation

from model.rrcdnet import RRCDNet
from model.rrcdnet_left import RRCDNet_Left

from model.rrcdnet_s4 import RRCDNet_S4
from model.rrcdnet_s4_left import RRCDNet_S4_Left


set_seed(42)

ckpt = torch.load("checkpoints/dncnn/rrcd_best.pt", map_location='cpu')
model = DnCNN().eval()
model.load_state_dict(ckpt['model_state'])

clean = generate_clean_signal()
noisy = add_gaussian_noise_to_snr(clean, 25)
noisy_scaled, mn, mx = minmax_scale(noisy)
noisy_tensor = torch.tensor(noisy_scaled[np.newaxis, np.newaxis, :], dtype=torch.float32)

with torch.no_grad():
    residual = model(noisy_tensor)
    denoised = noisy_tensor - residual
denoised = denoised.cpu().numpy().squeeze()

model2 = DnCNN_S4().eval()
ckpt = torch.load("checkpoints/dncnn_s4/rrcd_best.pt", map_location='cpu')
model2.load_state_dict(ckpt['model_state'])
with torch.no_grad():
    residual2 = model2(noisy_tensor)
    denoised2 = noisy_tensor - residual2
denoised2 = denoised2.cpu().numpy().squeeze()

print("average noise:")
print(average_noise(denoised,50))
print(average_noise(denoised2,50))

print("mse:")
print(mse(denoised,clean))
print(mse(denoised2,clean))



# Undo scaling to compare to original clean if desired:
# (If scaling was per-sample min-max, you'd unscale similarly)
plt.figure(figsize=(4,3))
plt.plot(clean, label='clean', alpha=0.7)
plt.plot(noisy, label='noisy', alpha=0.5)
plt.plot(denoised * (mx-mn) + mn, label='denoised (rescaled)', alpha=0.8)
plt.plot(denoised2 * (mx-mn) + mn, label='denoised2 (rescaled)', alpha=0.8)

#130-160
plt.xlim(350, 370)
plt.legend(); plt.show()
