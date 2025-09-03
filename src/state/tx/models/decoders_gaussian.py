import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple
from .utils import build_mlp

class GaussianDecoder(nn.Module):
    """
    A simple MLP-based decoder that maps a latent embedding to the parameters
    of a Gaussian distribution (mean and log-variance) over gene expression.
    """

    def __init__(
        self, latent_dim: int = 128, gene_dim: int = 200, hidden_dim: int = 512
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, gene_dim)
        self.logsig_head = nn.Linear(hidden_dim, gene_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the decoder.

        Args:
            x: Input tensor of shape [B, latent_dim].

        Returns:
            A tuple containing:
                - mu: The mean of the Gaussian distribution [B, gene_dim].
                - log_sigma: The log standard deviation of the Gaussian distribution [B, gene_dim].
        """
        h = self.backbone(x)
        mu = self.mu_head(h)
        # Clamp for numerical stability
        log_sigma = self.logsig_head(h).clamp(-5.0, 2.0)
        return mu, log_sigma

    @staticmethod
    def sample(
        mu: torch.Tensor, log_sigma: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples k candidates from N(mu, sigma) using the reparameterization trick.

        Args:
            mu: Mean of the Gaussian distribution [B, G].
            log_sigma: Log standard deviation of the Gaussian distribution [B, G].
            k: Number of samples to draw.

        Returns:
            A tuple containing:
                - y_samples: Sampled candidates of shape [k, B, G].
                - log_p: Log probability of each sample, summed over the gene dimension [k, B].
        """
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)

        # rsample uses the reparameterization trick
        y_samples = dist.rsample(sample_shape=(k,))  # Shape: [k, B, G]

        # Calculate log probability, summing over the gene dimension
        log_p = dist.log_prob(y_samples).sum(dim=-1)  # Shape: [k, B]

        return y_samples, log_p




class GaussianDecoder_v2(nn.Module):
    """
    A simple MLP-based decoder that maps a latent embedding to the parameters
    of a Gaussian distribution (mean and log-variance) over gene expression.
    using st's project out layer and final down then up layer
    """

    def __init__(
        self, 
        hidden_dim: int = 512,
        output_dim: int = 512,
        n_decoder_layers: int = 2,
        dropout: float = 0.0,
        activation_class: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.activation_class = activation_class

        self.project_out = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_decoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )

        self.mean_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 8),
            nn.GELU(),
            nn.Linear(self.output_dim // 8, self.output_dim),
        )

        self.logsig_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 8),
            nn.GELU(),
            nn.Linear(self.output_dim // 8, self.output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.project_out(x)
        mu = self.mean_head(h)
        log_sigma = self.logsig_head(h)
        return mu, log_sigma

    @staticmethod
    def sample(
        mu: torch.Tensor, log_sigma: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples k candidates from N(mu, sigma) using the reparameterization trick.
        """
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)

        # rsample uses the reparameterization trick
        y_samples = dist.rsample(sample_shape=(k,))  # Shape: [k, B, G]

        # Calculate log probability, summing over the gene dimension
        log_p = dist.log_prob(y_samples).sum(dim=-1)  # Shape: [k, B]

        return y_samples, log_p
