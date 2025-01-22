import torch
import torch.nn as nn
import torch.nn.functional as F

class WassersteinLoss(nn.Module):
    """
    Implements Wasserstein distance loss for distributions represented by logits.
    This implementation supports both 1D and 2D Wasserstein distance calculations.
    """
    def __init__(self, num_classes, p=1, reduction='mean'):
        """
        Args:
            num_classes (int): Number of classes in the distribution
            p (int): Order of Wasserstein distance (1 or 2)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.reduction = reduction

        # Pre-compute the cost matrix for efficiency
        self.register_buffer('cost_matrix', self._create_cost_matrix())

    def _create_cost_matrix(self):
        """Creates the cost matrix for computing Wasserstein distance."""
        indices = torch.arange(self.num_classes)
        cost_matrix = torch.abs(indices.view(-1, 1) - indices.view(1, -1))
        if self.p == 2:
            cost_matrix = cost_matrix.pow(2)
        return cost_matrix

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
        # Convert logits to probabilities
        pred_probs = F.softmax(input, dim=-1)
        target = F.softmax(target, dim=-1)
        # target = torch.log(target)

        # Compute cumulative distribution functions (CDFs)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target, dim=-1)

        # Compute Wasserstein distance
        wasserstein_dist = torch.abs(pred_cdf - target_cdf).pow(self.p)
        wasserstein_dist = wasserstein_dist.sum(dim=-1)

        # Apply reduction if specified
        if self.reduction == 'mean':
            return wasserstein_dist.mean()
        elif self.reduction == 'sum':
            return wasserstein_dist.sum()
        return wasserstein_dist
