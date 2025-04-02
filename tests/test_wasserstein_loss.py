import os
import sys
import logging
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vci.loss import wasserstein_loss


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tests")


def test_identical_distributions():
    y_hat = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 3.0])

    loss = wasserstein_loss(y_hat, y, p=1)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    dist = wasserstein_distance(y_hat.numpy(), y.numpy())
    dist = torch.tensor(dist, dtype=torch.float)

    assert torch.isclose(loss, dist, atol=1e-6)



def test_identical_random():
    y_hat = torch.randn(4)
    y = torch.randn(4)

    loss = wasserstein_loss(y_hat, y, p=1)

    dist = wasserstein_distance(y_hat.numpy(), y.numpy())
    dist = torch.tensor(dist, dtype=torch.float)

    assert torch.isclose(loss, dist, atol=1e-6)