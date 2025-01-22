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

# Example usage
def example_usage():
    # Create sample data
    batch_size = 32
    num_classes = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize loss function
    criterion = WassersteinLoss(num_classes=num_classes, p=2).to(device)

    # Generate random logits and targets
    logits = torch.randn(batch_size, num_classes).to(device)

    # Case 1: Target as class indices
    target_indices = torch.randint(0, num_classes, (batch_size,)).to(device)
    loss1 = criterion(logits, target_indices)

    # Case 2: Target as probability distribution
    target_probs = torch.rand(batch_size, num_classes).to(device)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
    loss2 = criterion(logits, target_probs)

    return loss1, loss2

if __name__ == "__main__":
    loss1, loss2 = example_usage()
    print(f"Loss with target indices: {loss1.item():.4f}")
    print(f"Loss with target probabilities: {loss2.item():.4f}")


# import torch

# from scipy.stats import wasserstein_distance


# def wasserstein_loss(x, y, p=2, eps=1e-3, max_iters=100):
#     tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
#     tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
#     cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
#     cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

#     if p == 1:
#         cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
#     elif p == 2:
#         cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
#     else:
#         cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

#     cdf_loss = cdf_distance.mean()
#     return cdf_loss

#     # n, d = x.shape
#     # m, _ = y.shape

#     # # Initialize cost matrix
#     # C = torch.cdist(x, y, p=p)

#     # # Initialize potentials
#     # u = torch.zeros(n, 1)
#     # v = torch.zeros(m, 1)

#     # for _ in range(max_iters):
#     #     u = torch.logsumexp(-C/eps + v.t(), dim=1, keepdim=True)
#     #     v = torch.logsumexp(-C/eps + u, dim=0, keepdim=True)

#     # # Compute the distance
#     # return torch.sum(u * torch.exp(-C/eps + v.t()))
