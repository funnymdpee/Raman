# dataset.py
import numpy as np
from torch.utils.data import Dataset
from utils import add_gaussian_noise_to_snr, minmax_scale, apply_scale_with_params,generate_clean_signal



class RamanSynthDataset(Dataset):
    def __init__(self, n_samples=5000, length=10000, snr_range=(20,37), train=True,maxlen = 51):
        self.n = n_samples
        self.length = length
        self.snr_low, self.snr_high = snr_range
        self.data = []
        # Pre-generate dataset to make training deterministic / faster
        for _ in range(n_samples):
            clean = generate_clean_signal(length=length,maxlen=maxlen)
            snr = np.random.uniform(self.snr_low, self.snr_high)
            noisy = add_gaussian_noise_to_snr(clean, snr)
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
