import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, grad_weight=0.1):
        """
        Args:
            grad_weight: 梯度损失的权重。
                         对于光纤信号，通常 MSE 值很小，梯度值也很小，
                         建议从 0.1 或 1.0 开始尝试，观察量级是否平衡。
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.grad_weight = grad_weight

    def forward(self, pred, target):
        # 1. 基础 MSE Loss (保证数值准确)
        loss_mse = self.mse(pred, target)
        
        if self.grad_weight!=0:
            # 2. Gradient Loss (保证形状/边缘准确)
            # 计算沿最后一个维度 (L维度) 的一阶差分
            # pred shape: [Batch, Channel, Length]
            pred_grad = pred[..., 1:] - pred[..., :-1]
            target_grad = target[..., 1:] - target[..., :-1]
            
            # 使用 L1 Loss 计算梯度差异，对边缘更敏感
            loss_grad = F.l1_loss(pred_grad, target_grad)
            
            # 3. 总损失
            total_loss = loss_mse + self.grad_weight * loss_grad
        else:
            total_loss = loss_mse
        
        return total_loss
