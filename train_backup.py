# train.py
import os
import sys
sys.path.append("D:\\Code\\Raman_scatter\\RRCDNet\\s4")
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from model.rrcdnet_right import RRCDNet
from model.rrcdnet_left import RRCDNet_Left

from model.rrcdnet_s4 import RRCDNet_S4
from model.rrcdnet_s4_left import RRCDNet_S4_Left

from model.dncnn import DnCNN
from model.dncnn_cnn import DnCNN_CNN
from model.dncnn_dilation import DnCNN_Dilation

from model.fbnet import BigNet,create_BigNet
from model.dncnn_s4 import  DnCNN_S4
from dataset import RamanSynthDataset
from utils import set_seed


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    mse = nn.MSELoss()
    for y_batch, x_batch in tqdm(loader, desc="train batches"):
        y = y_batch.to(device)  # (B,1,L)
        x = x_batch.to(device)
        opt.zero_grad()
        residual = model(y)  # predicted noise
        pred = y - residual
        loss = mse(pred, x)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    mse = nn.MSELoss()
    with torch.no_grad():
        for y_batch, x_batch in loader:
            y = y_batch.to(device)
            x = x_batch.to(device)
            residual = model(y)
            pred = y - residual
            loss = mse(pred, x)
            total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparams from paper
    lr = 3e-4
    batch_size = 32
    epochs = 200

    # Dataset (4500 train, 500 test)
    train_ds = RamanSynthDataset(n_samples=4500)
    val_ds = RamanSynthDataset(n_samples=500)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    #model = RRCDNet(in_ch=1, base_ch=64).to(device)
    model = DnCNN_S4().to(device)

    # Kaiming init for conv layers
    def init_weights(m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    opt = Adam(model.parameters(), lr=lr)
    best_val = 1e9
    ckpt_dir = "checkpoints/dncnn_s4"
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_path = ckpt_dir+"/rrcd_epoch100.pt"  # <-- 如果不恢复就设 None
    #resume_path = None
    start_epoch = 1

    if resume_path is not None and os.path.isfile(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["val_loss"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_val={best_val:.6f}")
    else:
        print("Training from scratch.")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device)
        val_loss = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"rrcd_epoch{epoch:03d}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(ckpt_dir, f"rrcd_best.pt"))

def custom(model,model_dir,epochs = 100):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparams from paper
    lr = 3e-4
    batch_size = 8

    # Dataset (4500 train, 500 test)
    train_ds = RamanSynthDataset(n_samples=4500,maxlen=251)
    val_ds = RamanSynthDataset(n_samples=500,maxlen=251)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    #model = RRCDNet_S4(in_ch=1, base_ch=64).to(device)

    # Kaiming init for conv layers
    def init_weights(m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    opt = Adam(model.parameters(), lr=lr)
    best_val = 1e9
    ckpt_dir = "checkpoints/"+model_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_path = ckpt_dir+"/rrcd_epoch100.pt"  # <-- 如果不恢复就设 None
    resume_path = None
    start_epoch = 1

    if resume_path is not None and os.path.isfile(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["val_loss"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_val={best_val:.6f}")
    else:
        print("Training from scratch.")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device)
        val_loss = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"rrcd_epoch{epoch:03d}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(ckpt_dir, f"rrcd_best.pt"))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_BigNet().to(device)
    custom(model=model,model_dir="bignet_l251")
