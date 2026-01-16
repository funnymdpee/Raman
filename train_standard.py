import os
import sys
import datetime
sys.path.append(".\\s4")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from loss import HybridLoss

# ==== Imports ====
# from model.rrcdnet_right import RRCDNet
# from model.rrcdnet_left import RRCDNet_Left
# from model.dncnn import DnCNN
# from model.dncnn_cnn import DnCNN_CNN
# from model.dncnn_dilation import DnCNN_Dilation
from model.unet import UNet1D_Denoise
from model.rapunet import RAPUNet

from dataset import RamanSynthDataset
from utils import set_seed


# =========================================================
# 🔧 基础训练逻辑
# =========================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    # criterion = nn.MSELoss()
    criterion = HybridLoss(grad_weight=0).to(device)

    for y_batch, x_batch in tqdm(loader, desc="Train", leave=False):
        y, x = y_batch.to(device), x_batch.to(device)
        optimizer.zero_grad()
        pred = model(y)
        loss = criterion(pred, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    criterion = HybridLoss(grad_weight=0).to(device)

    with torch.no_grad():
        for y_batch, x_batch in loader:
            y, x = y_batch.to(device), x_batch.to(device)
            pred = model(y)
            loss = criterion(pred, x)
            total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


# =========================================================
# 🧩 工具函数
# =========================================================
def init_weights(m):
    """Kaiming initialization for Conv1d layers."""
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_dataloaders(batch_size=32, train_size=10000, val_size=1000,maxlen = 251):
    train_ds = RamanSynthDataset(n_samples=train_size,maxlen=maxlen)
    val_ds = RamanSynthDataset(n_samples=val_size,maxlen=maxlen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def auto_ckpt_dir(base_dir="checkpoints", model_name=None):
    """自动生成 checkpoint 目录: checkpoints/<ModelName>_YYYY-MM-DD_HH-MM-SS"""
    if model_name is None:
        model_name = "unnamed_model"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def plot_loss_curve(train_losses, val_losses, ckpt_dir):
    """绘制并保存训练/验证损失曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(ckpt_dir, 'loss_curve.png')
    plt.savefig(save_path)
    plt.close()

    # 同时保存 CSV
    df = pd.DataFrame({'epoch': range(1, len(train_losses) + 1),
                       'train_loss': train_losses,
                       'val_loss': val_losses})
    df.to_csv(os.path.join(ckpt_dir, 'loss_log.csv'), index=False)


# =========================================================
# 🚀 通用训练流程
# =========================================================
def train_model(
    model,
    ckpt_dir=None,
    epochs=200,
    lr=3e-4,
    batch_size=32,
    resume_path=None,
    maxlen=251
):
    """
    通用训练接口。
    Args:
        model: torch.nn.Module
        ckpt_dir: 保存路径（若为None则自动生成）
        epochs: 训练轮数
        lr: 学习率
        batch_size: batch大小
        resume_path: 断点恢复路径
        scheduler_cfg: 例 {'step_size':200, 'gamma':0.1}
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.apply(init_weights)

    # === 自动命名路径 ===
    if ckpt_dir is None:
        model_name = model.__class__.__name__
        ckpt_dir = auto_ckpt_dir(model_name=model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(batch_size=batch_size, maxlen=maxlen)
    optimizer = Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    start_epoch = 1
    train_losses, val_losses = [], []

    # === Resume ===
    if resume_path and os.path.isfile(resume_path):
        print(f"🔄 Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device)

        # === 恢复模型 & 优化器 ===
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        best_val = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1

        # === 只恢复当前学习率，不加载旧 scheduler 状态 ===
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        print(f"✅ Restored optimizer (lr={current_lr:.6e})")
    else:
        print(f"🚀 Training from scratch. Saving to: {ckpt_dir}")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


    # === Main Loop ===
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, device)
        if scheduler:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.6e}")

        # === Save Checkpoints ===
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch{epoch:03d}.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))

    # === 保存损失曲线 ===
    plot_loss_curve(train_losses, val_losses, ckpt_dir)

    print(f"✅ Training completed. Best val_loss={best_val:.6f}")
    print(f"📁 Logs & models saved at: {ckpt_dir}")



# =========================================================
# 🧠 两个入口函数
# =========================================================
def main():
    """默认训练入口（论文复现）"""
    model = UNet1D_Denoise()
    train_model(
        model=model,
        epochs=200,
        lr=3e-4,
        batch_size=32
    )


def custom(model, epochs=100):
    """自定义模型训练，仅需换模型即可"""
    train_model(
        model=model,
        epochs=epochs,
        lr=1e-4,
        batch_size=16,
        #resume_path="checkpoints/UNet1D_Denoise_2026-01-13_06-27-21/epoch076.pt",
        ckpt_dir=None  # 自动生成目录
    )


# =========================================================
# ✅ Entry
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAPUNet().to(device)
    custom(model=model, epochs=200)
