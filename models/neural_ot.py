# File: models/neural_ot.py
import torch

from collections import defaultdict
from geomloss import SamplesLoss
from typing import Optional

from models.base import PerturbationModel
from models.decoders import DecoderInterface
from models.utils import build_mlp, get_activation_class, get_transformer_backbone


class NeuralOTPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        output_space: str = "gene",
        decoder: Optional[DecoderInterface] = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        # Call the parent PerturbationModel constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            output_space=output_space,
            decoder=decoder,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.distributional_loss = distributional_loss

        # Build the distributional loss from geomloss
        self.loss_fn = SamplesLoss(loss=self.distributional_loss)
        # self.loss_fn = LearnableAlignmentLoss()

        # Build the underlying neural OT network
        self._build_networks()

        # For caching validation data across steps, if desired
        self.val_cache = defaultdict(list)

    def _build_networks(self):
        """
        Here we instantiate the actual GPT2-based model or any neuralOT translator
        via your old get_model(model_key, model_kwargs) approach.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Map the input embedding to the hidden space
        self.basal_encoder = build_mlp(
            in_dim=self.input_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        self.convolve = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        print(self)

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Return the latent perturbed state given the perturbation and basal state.
        """
        pert_embedding = self.encode_perturbation(pert).unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
        control_cells = self.encode_basal_expression(basal).unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
        cls_input = torch.zeros_like(pert_embedding)  # shape: [batch_size, 1, hidden_dim]
        seq_input = torch.cat([pert_embedding, control_cells, cls_input], dim=1)  # shape: [batch_size, 3, hidden_dim]

        # forward pass + extract CLS last hidden state
        prediction = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state[:, -1]

        # add to basal if predicting residual
        if self.predict_residual:
            # treat the actual prediction as a residual sum to basal
            return prediction + control_cells.squeeze(1)
        else:
            return prediction

    def forward(self, batch: dict) -> torch.Tensor:
        """
        The main forward call. The old code used (B, 2, N) tensors as
        input to the GPT2-backbone. Here we reshape to (1, B, 2N) to
        allow cells to attend to one another and learn a distributional
        set function.

        """
        pert_embedding = self.encode_perturbation(batch["pert"]).unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
        control_cells = self.encode_basal_expression(batch["basal"]).unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
        seq_input = torch.cat([pert_embedding, control_cells], dim=1)  # shape: [batch_size, 2, hidden_dim]
        seq_input = seq_input.permute(1, 0, 2).reshape(
            1, -1, 2 * self.hidden_dim
        )  # shape: [1, batch_size, 2 * hidden_dim]
        seq_input = self.convolve(seq_input)  # shape: [1, batch_size, hidden_dim]

        # forward pass + extract CLS last hidden state
        prediction = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state[0]

        # add to basal if predicting residual
        if self.predict_residual:
            # treat the actual prediction as a residual sum to basal
            return self.project_out(prediction + control_cells.squeeze(1))
        else:
            return self.project_out(prediction)

    def test_step(self, batch, batch_idx):
        """
        Same approach for test.
        """
        output_samples = self.forward(batch)
        target_samples = batch["X"]
        loss = self.loss_fn(output_samples, target_samples).mean()

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Return an optimizer for the internal model parameters.
        (Or you can do param re-grouping if needed.)
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Experiment with a learnable alignment loss function that lets
# the model pick the ground truth for a given cell:

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableAlignmentLoss(nn.Module):
    def __init__(self, hidden_dim=1280, num_heads=4):
        """
        Initialize the learnable alignment loss function.

        Args:
            hidden_dim (int): Dimension of the key/query projection space
            num_heads (int): Number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Learnable projection matrices for cross-attention
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)

    def forward(self, pred_samples, target_samples):
        """
        Compute the loss between predictions and targets using learnable alignment.

        Args:
            pred_samples (torch.Tensor): Predictions of shape (B, N)
            target_samples (torch.Tensor): Targets of shape (B, N)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Ensure inputs are the correct shape
        assert pred_samples.shape == target_samples.shape

        B, N = pred_samples.shape  # batch size, feature dimension

        # Project predictions and targets
        Q = self.Q(pred_samples)  # (B, hidden_dim)
        K = self.K(target_samples)  # (B, hidden_dim)

        # Compute attention scores
        # (B, B) = (B, hidden_dim) @ (hidden_dim, B)
        attention_scores = torch.mm(Q, K.transpose(-2, -1))

        # Apply row-wise softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, B)

        # Apply attention weights to targets
        aligned_targets = torch.mm(attention_weights, target_samples)  # (B, N)

        # Compute MSE loss between aligned targets and predictions
        loss = F.mse_loss(aligned_targets, pred_samples)

        return loss
