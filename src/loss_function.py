import torch
import torch.nn as nn
from torch.nn import functional as F


class DiceCoeff(nn.Module):
    def __init__(self, ignore_index: int = None, smooth: float = 1e-7):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: (N, C, ...) — raw logits
        targets: (N, ...) — class indices
        """
        inputs = F.softmax(inputs, dim=1)  # Convert to probabilities
        N, C = inputs.shape[:2]
        spatial_dims = inputs.shape[2:]  # Arbitrary spatial shape

        # One-hot encode targets to (N, C, ...)
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0, -1, *range(1, targets.ndim)).float()

        # Flatten all dimensions except batch and channel
        inputs_flat = inputs.view(N, C, -1)
        targets_flat = targets_onehot.view(N, C, -1)

        # Optional: handle ignore index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index  # shape (N, ...)
            mask = mask.view(N, -1).unsqueeze(1)  # (N, 1, num_voxels)
            inputs_flat = inputs_flat * mask
            targets_flat = targets_flat * mask

        intersection = (inputs_flat * targets_flat).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (
            inputs_flat.sum(dim=2) + targets_flat.sum(dim=2) + self.smooth
        )
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        """
        Multi-class focal loss implementation.
        Args:
            alpha (float): balancing factor for classes.
            gamma (float): focusing parameter.
            reduction (str): 'mean', 'sum' or 'none'.
            ignore_index (int, optional): class index to ignore in loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)  # pt = softmax prob of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
class LaplacianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a 3x3 Laplacian kernel
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def forward(self, pred: torch.Tensor):
        """
        Args:
            pred: (N, 1, H, W) tensor on any device
        Returns:
            Laplacian smoothness loss (scalar)
        """
        # Move kernel to the same device as pred
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        kernel = self.kernel.to(pred.device)
        pred_lap = F.conv2d(pred, kernel, padding=1)
        return torch.mean(torch.abs(pred_lap))

class _NPBranchLoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(_NPBranchLoss, self).__init__()
        self.dice_coeff = DiceCoeff()
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets)
        dice_loss = 1 - self.dice_coeff(logits, targets)
        return self.alpha * ce_loss + self.beta * dice_loss


class _HVBranchLoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.laplacian = LaplacianLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, h_grads: torch.Tensor, v_grads: torch.Tensor):
        hl = logits[:, 0, :, :]
        vl = logits[:, 1, :, :]

        mse_loss = F.mse_loss(hl, h_grads) + F.mse_loss(vl, v_grads)
        laplacian_loss = self.laplacian(hl) + self.laplacian(vl)

        return self.alpha * mse_loss + self.beta * laplacian_loss

class _GCBranchLoss(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1):
        super(_GCBranchLoss, self).__init__()
        self.dice_coeff = DiceCoeff()
        self.focal_loss = FocalLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets)
        dice_loss = 1 - self.dice_coeff(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        return self.alpha * ce_loss + self.beta * dice_loss + self.gamma * focal_loss
        
class HoverLoss(nn.Module):
    def __init__(self):
        super(HoverLoss, self).__init__()
        self.np_loss = _NPBranchLoss(0.75, 0.25)
        self.hv_loss = _HVBranchLoss(0.5, 0.5)
        self.nc_loss = _NPBranchLoss(0.75, 0.25)
        self.gc_loss = _GCBranchLoss(1, 1, 1)

    def forward(self, np_logits, np_targets,
                hv_logits, h_grads, v_grads,
                nc_logits, nc_targets,
                gc_logits=None, gc_targets=None,
                weights=(1, 1, 1, 1)) -> torch.Tensor:

        loss_np = self.np_loss(np_logits, np_targets) * weights[0]
        loss_hv = self.hv_loss(hv_logits, h_grads, v_grads) * weights[1]
        loss_nc = self.nc_loss(nc_logits, nc_targets) * weights[2]

        if gc_logits is not None and gc_targets is not None and gc_logits.numel() > 0:
            loss_gc = self.gc_loss(gc_logits, gc_targets) * weights[3]
        else:
            loss_gc = torch.tensor(0.0, device=np_logits.device)

        return loss_np + loss_hv + loss_nc + loss_gc