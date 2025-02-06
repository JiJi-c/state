import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

class WassersteinLoss(nn.Module):
    """
    Implements Wasserstein distance loss for distributions represented by logits.
    This implementation supports both 1D and 2D Wasserstein distance calculations.
    """
    def __init__(self, p=1, reduction='mean'):
        """
        Args:
            p (int): Order of Wasserstein distance (1 or 2)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        """
        Compute Wasserstein distance between predicted and target distributions.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes)
            target (torch.Tensor): Target probabilities of shape (batch_size, num_classes)
                                 or class indices of shape (batch_size,)

        Returns:
            torch.Tensor: Computed Wasserstein distance
        """

        target = torch.nan_to_num(target, nan=0.0)
        # Convert logits to probabilities
        pred_probs = F.softmax(input, dim=-1)
        target = F.softmax(target, dim=-1)

        # Compute cumulative distribution functions (CDFs)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target, dim=-1)

        max_len = max(pred_cdf.size(1), target_cdf.size(1))
        if pred_cdf.size(1) < max_len:
            pred_cdf = F.pad(pred_cdf, (0, max_len - pred_cdf.size(1)), 'constant', 0)
        if target_cdf.size(1) < max_len:
            target_cdf = F.pad(target_cdf, (0, max_len - target_cdf.size(1)), 'constant', 0)

        # Compute Wasserstein distance
        wasserstein_dist = torch.abs(pred_cdf - target_cdf).pow(self.p)
        wasserstein_dist = wasserstein_dist.sum(dim=-1)

        # Apply reduction if specified
        if self.reduction == 'mean':
            return wasserstein_dist.mean()
        elif self.reduction == 'sum':
            return wasserstein_dist.sum()
        return wasserstein_dist

class KLDivergenceLoss(nn.Module):
    def __init__(self, apply_normalization=False, epsilon=1e-10):
        super().__init__()
        self.apply_normalization = apply_normalization
        self.epsilon = epsilon

    def forward(self, input, target):
        target = torch.nan_to_num(target, nan=0.0)

        max_len = max(input.size(1), target.size(1))
        if input.size(1) < max_len:
            input = F.pad(input, (0, max_len - input.size(1)), 'constant', 0)
        if target.size(1) < max_len:
            target = F.pad(target, (0, max_len - target.size(1)), 'constant', 0)

        if self.apply_normalization:
            p = F.softmax(input, dim=-1)
            q = F.softmax(target, dim=-1)
        else:
            p = input
            q = target

        return torch.sum(p * torch.log(p / q))

class MMDLoss(nn.Module):
    def __init__(self, kernel="energy", blur=0.05, scaling=0.5):
        super().__init__()
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)

    def forward(self, input, target):
        return self.mmd_loss(input, target)