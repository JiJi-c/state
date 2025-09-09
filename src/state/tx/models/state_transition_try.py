import logging
from typing import Dict, Optional
import math
import anndata as ad
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from geomloss import SamplesLoss
from typing import Tuple

from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder
from .decoders_nb import NBDecoder, nb_nll
from .utils import build_mlp, get_activation_class, get_transformer_backbone
from .decoders_gaussian import GaussianDecoder, GaussianDecoder_v2
from ._reward import RewardEvaluator
#from ._reward_official import MetricsEvaluator
logger = logging.getLogger(__name__)

class CombinedLoss(nn.Module):
    """
    Combined Sinkhorn + Energy loss
    """
    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)
    
    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val

class ConfidenceToken(nn.Module):
    """
    Learnable confidence token that gets appended to the input sequence
    and learns to predict the expected loss value.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learnable confidence token embedding
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map confidence token output to scalar loss prediction
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),  # Ensure positive loss prediction
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand confidence token to batch size
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project confidence token output to scalar
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred


class StateTransitionPerturbationModel(PerturbationModel):
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
        batch_dim: int = None,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
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
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        # Save or store relevant hyperparams
        # decide whether to use GRPO mode
        self.grpo_mode = kwargs.get("grpo_mode", False)
       
        self.gene_list_path = kwargs.get("gene_list_path", "/yuchang/shangyue/wd/data_state/competition_support_set/gene_names.csv")
        
        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim

        self._setup_grpo_components(**kwargs)


        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)  # 1/100 = 0.01
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur)
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Build the underlying neural OT network
        self._build_networks()

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = torch.nn.ReLU()

        # initialize a confidence token
        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.confidence_loss_fn = nn.MSELoss()

        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", False)
        if self.freeze_pert_backbone:
            modules_to_freeze = [
                self.transformer_backbone,
                self.project_out,
            ]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            gene_names = []
            if output_space == "gene":
                # hvg's but for which dataset?
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5")
                    # gene_names = temp.var.index.values
            else:
                assert output_space == "all"
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5")
                    gene_names = temp.var.index.values

            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
                # latent_dim=self.output_dim + (self.batch_dim or 0),
            )

        print(self)


    def _setup_grpo_components(self, **kwargs):
        """GRPO mode setup"""
        grpo_config = kwargs.get("grpo_config", {})
        self.reward_weights = grpo_config.get("reward_weights", 
                                        kwargs.get("reward_weights", 
                                            {"pds": 1.0, "des": 1.0}))
        # add a gaussian decoder outputing mu and log_sigma
        if self.grpo_mode:
            self.gaussian_decoder = GaussianDecoder_v2(
                hidden_dim=self.hidden_dim,
                output_dim=self.gene_dim,
                n_decoder_layers=self.n_decoder_layers,
                dropout=self.dropout,
                activation_class=self.activation_class,
            )
            self.mu_old_buf = None
            self.logsig_old_buf = None


            # self.metrics_evaluator = MetricsEvaluator(
            #     gene_names=self.gene_list,
            #     reward_weights=self.reward_weights,
            # )

            self.k_samples = grpo_config.get("k_samples", kwargs.get("k_samples", 16))
            self.ppo_eps = grpo_config.get("ppo_eps", kwargs.get("ppo_eps", 0.2))
            self.alpha_kl = grpo_config.get("alpha_kl", kwargs.get("alpha_kl", 0.1))
            self.alpha_ent = grpo_config.get("alpha_ent", kwargs.get("alpha_ent", 0.01))
            self.alpha_sup = grpo_config.get("alpha_sup", kwargs.get("alpha_sup", 1.0))

            # Training schedule parameters
            self.stage0_steps = grpo_config.get("stage0_steps", 1000)
            self.stage1_steps = grpo_config.get("stage1_steps", 2000)  # steps for stage 1
            self.stage2_start_step = self.stage0_steps + self.stage1_steps
            
            # log_sigma annealing parameters
            self.initial_max_logsig = grpo_config.get("initial_max_logsig", 0.5)
            self.final_max_logsig = grpo_config.get("final_max_logsig", -0.2)
            
            # Target KL for adaptive alpha_kl
            self.target_kl = grpo_config.get("target_kl", 0.02)
       
        self.min_perts_for_pds = grpo_config.get("min_perts_for_pds", kwargs.get("min_perts_for_pds", 3))
        self.k_percentage = grpo_config.get("des_k_percentage", kwargs.get("des_k_percentage", 0.05))

        self.gene_list = pd.read_csv(self.gene_list_path, header=None)[0].tolist()

    
    
    def _build_networks(self):
        """
        Here we instantiate the actual GPT2-based model.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Simple linear layer that maintains the input dimension
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Project from input_dim to hidden_dim for transformer input
        # self.project_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)

        if self.grpo_mode == False:
            self.project_out = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_decoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )

            if self.output_space == 'all':
                self.final_down_then_up = nn.Sequential(
                    nn.Linear(self.output_dim, self.output_dim // 8),
                    nn.GELU(),
                    nn.Linear(self.output_dim // 8, self.output_dim),
                )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def forward(self, batch: dict, padded=True) -> torch.Tensor:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded: 
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        # Shape: [B, S, input_dim]
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        # Add encodings in input_dim space, then project to hidden_dim
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]

        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        confidence_pred = None
        if self.confidence_token is not None:
            # Append confidence token: [B, S, E] -> [B, S+1, E]
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device

            self.transformer_backbone._attn_implementation = "eager"

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device).view(1, seq_length, seq_length)

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            transformer_output = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state

        # Extract confidence prediction if confidence token was used
        if self.confidence_token is not None:
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
        else:
            res_pred = transformer_output

        # add to basal if predicting residual
        if self.grpo_mode:
            # GRPO mode, directly use transformer output
            mu, logsig = self.gaussian_decoder(res_pred)
            if self.confidence_token is not None:
                return mu, logsig, confidence_pred
            else:
                return mu, logsig
        else: 
            if self.predict_residual and self.output_space == "all":
                # Project control_cells to hidden_dim space to match res_pred
                # control_cells_hidden = self.project_to_hidden(control_cells)
                # treat the actual prediction as a residual sum to basal
                out_pred = self.project_out(res_pred) + basal
                out_pred = self.final_down_then_up(out_pred)
            elif self.predict_residual:
                out_pred = self.project_out(res_pred + control_cells)
            else:
                out_pred = self.project_out(res_pred)

            # apply relu if specified and we output to HVG space
            is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
            # logger.info(f"DEBUG: is_gene_space: {is_gene_space}")
            # logger.info(f"DEBUG: self.gene_decoder: {self.gene_decoder}")
            if is_gene_space or self.gene_decoder is None:
                out_pred = self.relu(out_pred)

            output = out_pred.reshape(-1, self.output_dim)

            if confidence_pred is not None:
                return output, confidence_pred
            else:
                return output

    def setup(self, stage: str = None):
        """Setup method to initialize mappings after data module is ready"""
        super().setup(stage)
        pert_onehot_map = self.trainer.datamodule.pert_onehot_map
        self.pert_list = list(pert_onehot_map.keys())
        self.pert_to_idx = {
            name: idx for idx, name in enumerate(pert_onehot_map.keys())
        }
        self.num_unique_perts = len(self.pert_to_idx)
        self.reward_evaluator = RewardEvaluator(
            gene_list=self.gene_list,
            pert_list=self.pert_list,
            reward_weights=self.reward_weights,
            min_perts_for_pds=self.min_perts_for_pds,
            k_percentage=self.k_percentage,
        )

    def _original_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """原始的StateTransition训练逻辑"""
        # Get model predictions (in latent space)
        confidence_pred = None
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded=padded)
        else:
            pred = self.forward(batch, padded=padded)

        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        main_loss = self.loss_fn(pred, target).nanmean()
        self.log("train_loss", main_loss)
        
        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, 'sinkhorn_loss') and hasattr(self.loss_fn, 'energy_loss'):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).nanmean()
            energy_component = self.loss_fn.energy_loss(pred, target).nanmean()
            self.log("train/sinkhorn_loss", sinkhorn_component)
            self.log("train/energy_loss", energy_component)

        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = total_loss.detach().clone().unsqueeze(0) * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", loss_target.mean())

            # Add to total loss with weighting
            confidence_weight = 0.1  # You can make this configurable
            total_loss = total_loss + confidence_weight * confidence_loss

            # Add to total loss
            total_loss = total_loss + confidence_loss

        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb

            # compute l1 loss
            l1_loss = torch.abs(delta).mean()

            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)

            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss

        return total_loss

    def _update_grpo_params_on_step(self):
        """Update GRPO hyperparameters based on the current training step."""
        current_step = self.global_step

        # Default weights
        self.ppo_weight = 1.0
        self.kl_weight = self.alpha_kl
        self.ent_weight = self.alpha_ent

        if current_step < self.stage0_steps:
            # Stage 0: Supervised warm-up
            self.stage0_mode = True
            self.current_k_samples = 1
            self.current_alpha_sup = 5.0
            self.ppo_weight = 0.0
            self.kl_weight = 0.0
            self.ent_weight = 0.0
            self.max_logsig = self.initial_max_logsig # No annealing yet
        else:
            self.stage0_mode = False
            # Stage 1: Light GRPO exploration
            if current_step < self.stage2_start_step:
                self.current_k_samples = 4
                self.current_alpha_sup = 2.0
                # Anneal max_logsig
                progress = (current_step - self.stage0_steps) / self.stage1_steps
                self.max_logsig = self.initial_max_logsig - progress * (self.initial_max_logsig - self.final_max_logsig)
            else:
                # Stage 2: Full GRPO exploration
                if current_step < self.stage2_start_step + 3000:
                    self.current_k_samples = 8
                else:
                    self.current_k_samples = 16
                self.current_alpha_sup = 0.5
                self.max_logsig = self.final_max_logsig # Annealing finished
        
        # Log dynamic hparams for monitoring
        self.log("hparams/current_k_samples", float(self.current_k_samples), on_step=True, on_epoch=False, prog_bar=True)
        self.log("hparams/current_alpha_sup", self.current_alpha_sup, on_step=True, on_epoch=False)
        self.log("hparams/max_logsig", self.max_logsig, on_step=True, on_epoch=False)
        if not self.stage0_mode:
            self.log("hparams/alpha_kl", self.alpha_kl, on_step=True, on_epoch=False)

    def _grpo_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """GRPO训练步骤"""
        # 1) Update dynamic hyperparameters based on training schedule
        self._update_grpo_params_on_step()

        # 2) Forward pass to get current policy distribution
        confidence_pred = None
        if self.confidence_token is not None:
            mu, log_sigma, confidence_pred = self.forward(batch, padded=padded)
        else:
            mu, log_sigma = self.forward(batch, padded=padded)
        
        B, S, G = mu.shape
        mu_flat = mu.reshape(-1, G)
        log_sigma_flat = log_sigma.reshape(-1, G)
        y_real = batch["pert_cell_counts"]
        
        # 3) Supervised Loss (always calculated)
        loss_sup = F.l1_loss(mu_flat, y_real)
        
        # 4) Stage 0: Supervised warm-up. Only use supervised loss.
        if self.stage0_mode:
            self.log("train/total_loss", loss_sup, on_step=True, on_epoch=False)
            self.log("train/supervised_loss", loss_sup, on_step=True, on_epoch=False)
            return loss_sup

        # --- GRPO Stages (1 and 2) ---
        
        # 5) Determine old policy and sample from it (correct importance sampling)
        with torch.no_grad():
            if (self.mu_old_buf is None) or (self.mu_old_buf.shape[0] != mu_flat.shape[0]):
                mu_old_use = mu_flat.detach()
                logsig_old_use = log_sigma_flat.detach()
            else:
                mu_old_use = self.mu_old_buf
                logsig_old_use = self.logsig_old_buf
            
            y_samples, logp_old = self.gaussian_decoder.sample(mu_old_use, logsig_old_use, k=self.current_k_samples)

        # 6) Evaluate samples on the new policy
        log_sigma_flat = log_sigma_flat.clamp(min=-2.5, max=self.max_logsig) # Clamp sigma for stability
        dist_new = Normal(mu_flat, log_sigma_flat.exp())
        logp_new = dist_new.log_prob(y_samples).sum(dim=-1)

        # 7) Compute rewards
        with torch.no_grad():
            R, pds_r, des_r, local_perts_num = self.reward_evaluator.aggregate_rewards(
                y_samples=y_samples.detach(),
                y_real=y_real.detach(),
                basal_cells=batch["ctrl_cell_counts"].detach(),
                pert_names=batch["pert_name"],
            )

        # 8) Advantage calculation with per-perturbation whitening
        A = self._grpo_advantage(R) # Group-wise normalization inside advantage calculation

        # 9) Calculate PPO loss, KL divergence, and entropy
        ppo_loss = self._ppo_clip_loss(logp_new, logp_old, A)
        kl_loss = self._gaussian_kl(mu_flat, log_sigma_flat, mu_old_use, logsig_old_use)
        entropy = (log_sigma_flat + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=-1).mean()
        
        # 10) Adaptive KL penalty
        with torch.no_grad():
            approx_kl = (logp_old - logp_new).mean()
            if approx_kl > 1.5 * self.target_kl:
                self.alpha_kl = min(self.alpha_kl * 1.5, 0.5)
            elif approx_kl < 0.5 * self.target_kl:
                self.alpha_kl = max(self.alpha_kl / 1.5, 1e-3)
        self.log("train/approx_kl", approx_kl, on_step=True, on_epoch=False)
        
        # 11) Combine all losses for the final objective
        loss = (
            self.ppo_weight * ppo_loss
            + self.kl_weight * kl_loss
            - self.ent_weight * entropy
            + self.current_alpha_sup * loss_sup
        )
        
        # 12) Update old-policy buffers
        with torch.no_grad():
            self.mu_old_buf = mu_flat.detach().clone()
            self.logsig_old_buf = log_sigma_flat.detach().clone()
        
        # 13) Logging
        self.log("train/total_loss", loss, on_step=True, on_epoch=False)
        self.log("train/ppo_loss", ppo_loss, on_step=True, on_epoch=False)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=False)
        self.log("train/entropy", entropy, on_step=True, on_epoch=False)
        self.log("train/supervised_loss", loss_sup, on_step=True, on_epoch=False)
        self.log("train/pds_reward", pds_r.mean(), on_step=True, on_epoch=False)
        self.log("train/des_reward", des_r.mean(), on_step=True, on_epoch=False)
        self.log("train/total_reward", R.mean(), on_step=True, on_epoch=False)
        self.log("train/local_perts_num", local_perts_num, on_step=True, on_epoch=False)

        if confidence_pred is not None:
            loss_target = loss.detach().clone() * 10
            if confidence_pred.dim() == 2:
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("train/confidence_loss", confidence_loss)
            loss = loss + 0.1 * confidence_loss
        
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """训练步骤的统一入口"""
        if self.grpo_mode:
            return self._grpo_training_step(batch, batch_idx, padded)
        else:
            return self._original_training_step(batch, batch_idx, padded)

    
    def _grpo_validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """GRPO validation step logic"""
        
        with torch.no_grad():
            # 1) 前向传播获得mu, log_sigma
            confidence_pred = None
            if self.confidence_token is not None:
                mu, log_sigma, confidence_pred = self.forward(batch)
            else:
                mu, log_sigma = self.forward(batch)
            
            B, S, G = mu.shape
            mu_flat = mu.reshape(-1, G)
            log_sigma_flat = log_sigma.reshape(-1, G)

            y_real = batch["pert_cell_counts"]
            basal_cells = batch["ctrl_cell_counts"]
            pert_names = batch["pert_name"]
        
            # 3) 采样K个候选
            y_samples, logp_new = self.gaussian_decoder.sample(mu_flat, log_sigma_flat, k=self.current_k_samples)
            
            # 4) 计算奖励 (使用轻量版DES)
            R, pds_r, des_r, local_perts_num = self.reward_evaluator.aggregate_rewards(
                y_samples=y_samples.detach(),
                y_real=y_real.detach(),
                basal_cells=basal_cells.detach(),
                pert_names=pert_names,
            )
            
            # 5) 计算损失
            loss_sup = F.l1_loss(mu_flat, y_real) 
            entropy = (log_sigma_flat + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=-1).mean()
            
            # 6) 记录验证指标 (基于轻量奖励)
            self.log("val_loss", loss_sup) 
            self.log("val/entropy", entropy)
            self.log("val/pds_reward", pds_r.mean())
            self.log("val/des_reward_approx", des_r.mean()) # 标注为 approx
            self.log("val/total_reward_approx", R.mean())
            
            # --- 新增：定期计算并记录精确的 DES 奖励 ---
            # 只在第一个 validation batch 并且满足全局步数频率时执行
            #if batch_idx == 0 and (self.global_step % self.val_exact_des_freq == 0):
            # 准备索引
            pert_indices_global = torch.tensor([self.pert_to_idx.get(name, -1) for name in pert_names], device=y_samples.device, dtype=torch.long)
            valid_mask = pert_indices_global != -1
            pert_indices_global_valid = pert_indices_global[valid_mask]
            _, local_indices = pert_indices_global_valid.unique(sorted=True, return_inverse=True)
            M = int(local_indices.max() + 1)
            
            # 调用精确但耗时的 DES 计算函数
            des_r_exact = self.reward_evaluator.calculate_des_reward_pytorch(
                y_samples=y_samples[:, valid_mask, :].detach(),
                y_real=y_real[valid_mask, :].detach(),
                basal_cells=basal_cells[valid_mask, :].detach(),
                pert_indices=local_indices,
                M=M
            )
            self.log("val/des_reward_exact", des_r_exact.mean())

            # 7) 如果有confidence prediction，也记录
            if confidence_pred is not None:
                loss_target = loss_sup.clone() * 10  
                if confidence_pred.dim() == 2:
                    loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
                else:
                    loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
                confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
                self.log("val/confidence_loss", confidence_loss)

    def _original_validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """original validation step logic"""
        confidence_pred = None
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch), None
        else:
            pred, confidence_pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)
        
        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, 'sinkhorn_loss') and hasattr(self.loss_fn, 'energy_loss'):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
            energy_component = self.loss_fn.energy_loss(pred, target).mean()
            self.log("val/sinkhorn_loss", sinkhorn_component)
            self.log("val/energy_loss", energy_component)

        # 添加reward计算用于对比实验
        with torch.no_grad():
            # 将确定性预测reshape为适合reward计算的格式
            pred_flat = pred.reshape(-1, self.output_dim)  # [B*S, G]
            
            # 准备数据
            y_real = batch["pert_cell_counts"]
            basal_cells = batch["ctrl_cell_counts"]
            
            # 构建pert索引
            pert_names = batch["pert_name"]
            
            # 将确定性预测作为单个"样本"用于reward计算
            # 形状从 [N, G] 扩展到 [1, N, G]
            y_samples = pred_flat.unsqueeze(0)  # [1, N, G]
            
            # 计算奖励（使用和GRPO相同的参数）
            R, pds_r, des_r, local_perts_num = self.reward_evaluator.aggregate_rewards(
                y_samples=y_samples.detach(),      # [K, N, G]
                y_real=y_real.detach(),            # [N, G]
                basal_cells=basal_cells.detach(),  # [N, G]
                pert_names=pert_names,         # [N]
            )
            
            # 记录原始模式的奖励指标
            #self.log("val/mae_reward", mae_r.mean())
            self.log("val/pds_reward", pds_r.mean())
            self.log("val/des_reward", des_r.mean())
            self.log("val/total_reward", R.mean())
            self.log("val/local_perts_num", local_perts_num)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                # Get decoder predictions
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("val/confidence_loss", confidence_loss)
            self.log("val/actual_loss", loss_target.mean())

        return {"loss": loss, "predictions": pred}


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        if self.grpo_mode:
            return self._grpo_validation_step(batch, batch_idx)
        else:
            return self._original_validation_step(batch, batch_idx)



    def _grpo_test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """GRPO测试步骤逻辑"""
        with torch.no_grad():
            # 1) 前向传播获得mu, log_sigma
            confidence_pred = None
            if self.confidence_token is not None:
                mu, log_sigma, confidence_pred = self.forward(batch, padded=False)
            else:
                mu, log_sigma = self.forward(batch, padded=False)
            
            # 2) 标准化形状处理，测试时通常是单个batch，所以reshape为[1, -1, G]
            # if mu.dim() == 3:
            B, S, G = mu.shape
            mu_flat = mu.reshape(-1, G)                 # [B*S, G]
            log_sigma_flat = log_sigma.reshape(-1, G)   # [B*S, G]
            # else:
            #     mu_flat = mu                               # [N, G]
            #     log_sigma_flat = log_sigma                 # [N, G]
            
            y_real = batch["pert_cell_counts"]
            basal_cells = batch["ctrl_cell_counts"]
            
            # 构建pert索引
            pert_names = batch["pert_name"]
            
            # 3) 采样K个候选样本
            y_samples, logp_new = self.gaussian_decoder.sample(mu_flat, log_sigma_flat, k=self.k_samples)
            
            # 4) 计算三项奖励指标
            R, pds_r, des_r, local_perts_num = self.reward_evaluator.aggregate_rewards(
                y_samples=y_samples,      # [K, N, G]
                y_real=y_real,            # [N, G]
                basal_cells=basal_cells,  # [N, G]
                pert_names=pert_names,         # [N]
            )
            
            # 5) 计算基础损失
            loss_sup = F.l1_loss(mu_flat, y_real) 
            entropy = (log_sigma_flat + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=-1).mean()
            
            # 6) 记录测试指标
            self.log("test_loss", loss_sup) 
            self.log("test/entropy", entropy)
            #self.log("test/mae_reward", mae_r.mean())
            self.log("test/pds_reward", pds_r.mean())
            self.log("test/des_reward", des_r.mean())
            self.log("test/total_reward", R.mean())
            self.log("test/local_perts_num", local_perts_num)
            
            # 7) 如果有confidence prediction，也记录
            if confidence_pred is not None:
                loss_target = loss_sup.clone() * 10  
                if confidence_pred.dim() == 2:
                    loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
                else:
                    loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
                confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
                self.log("test/confidence_loss", confidence_loss)

    def _original_test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """原始模式的测试步骤逻辑"""
        confidence_pred = None
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch, padded=False), None
        else:
            pred, confidence_pred = self.forward(batch, padded=False)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

        # 添加reward计算用于对比实验
        with torch.no_grad():
            # 将确定性预测reshape为适合reward计算的格式
            pred_flat = pred.reshape(-1, self.output_dim)  # [N, G]
            
            # 准备数据
            y_real = batch["pert_cell_counts"]
            basal_cells = batch["ctrl_cell_counts"]
            
            # 构建pert索引
            pert_names = batch["pert_name"]
            
            # 将确定性预测作为单个"样本"用于reward计算
            # 形状从 [N, G] 扩展到 [1, N, G]
            y_samples = pred_flat.unsqueeze(0)  # [1, N, G]
            
            # 计算奖励（使用和GRPO相同的参数）
            R, pds_r, des_r, local_perts_num = self.reward_evaluator.aggregate_rewards(
                y_samples=y_samples,      # [1, N, G] - 确定性预测作为单个样本
                y_real=y_real,            # [N, G]
                basal_cells=basal_cells,  # [N, G]
                pert_names=pert_names,         # [N]
            )
            
            # 记录原始模式的奖励指标
            # self.log("test/mae_reward", mae_r.mean())
            self.log("test/pds_reward", pds_r.mean())
            self.log("test/des_reward", des_r.mean())
            self.log("test/total_reward", R.mean())
            self.log("test/local_perts_num", local_perts_num)

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10.0

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("test/confidence_loss", confidence_loss)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """测试步骤的统一入口"""
        if self.grpo_mode:
            return self._grpo_test_step(batch, batch_idx)
        else:
            return self._original_test_step(batch, batch_idx)


    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. 
         returning 'preds', 'X', 'pert_name', etc.
        """
        if self.grpo_mode:
            return self._grpo_predict_step(batch, batch_idx, padded, **kwargs)
        else:
            return self._original_predict_step(batch, batch_idx, padded, **kwargs)

    def _original_predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """原始模式的predict step"""
        if self.confidence_token is None:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
            confidence_pred = None
        else:
            latent_output, confidence_pred = self.forward(batch, padded=padded)

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        # Add confidence prediction to output if available
        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred

        if self.gene_decoder is not None:
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

    def _grpo_predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """GRPO模式的predict step - 兼容_infer.py脚本"""
        with torch.no_grad():
            # 1) 前向传播获得mu, log_sigma
            confidence_pred = None
            if self.confidence_token is not None:
                mu, log_sigma, confidence_pred = self.forward(batch, padded=padded)
            else:
                mu, log_sigma = self.forward(batch, padded=padded)
            
            # 2) 标准化形状处理 - 兼容padded=False的情况
            #if mu.dim() == 3:
            B, S, G = mu.shape
            mu_flat = mu.reshape(-1, G)  # [B*S, G]
            # else:
            #     mu_flat = mu  # [N, G]

            # 3) 使用mu作为预测结果
            output_dict = {
                "preds": mu_flat,
                "pert_cell_emb": batch.get("pert_cell_emb", None),
                "pert_cell_counts": batch.get("pert_cell_counts", None),
                "pert_name": batch.get("pert_name", None),
                "celltype_name": batch.get("cell_type", None),
                "batch": batch.get("batch", None),
                "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
                "pert_cell_barcode": batch.get("pert_cell_barcode", None),
                "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
                "pert_cell_counts_preds": mu_flat, 
            }

            if confidence_pred is not None:
                output_dict["confidence_pred"] = confidence_pred

            return output_dict

    
    def _grpo_advantage(self, R: torch.Tensor) -> torch.Tensor:
        """
        R: [K,B] -> A: [K,B] 组内标准化优势，停止梯度回传
        """
        mu = R.mean(0, keepdim=True)
        sd = R.std(0, keepdim=True) + 1e-8
        A = (R - mu) / sd
        return A.detach()

    def _ppo_clip_loss(
        self, logp_new: torch.Tensor, logp_old: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """
        所有量均为 [K,B]
        """
        # 计算原始 log_ratio
        log_ratio_raw = logp_new - logp_old

        with torch.no_grad():
            # 统计ratio > 1+eps or ratio < 1-eps 的比例
            ratio_raw = torch.exp(log_ratio_raw)
            clipped_mask = (ratio_raw > 1 + self.ppo_eps) | (ratio_raw < 1 - self.ppo_eps)
            clip_fraction = torch.mean(clipped_mask.float())

            # 计算中位数和其他统计量
            median_val = torch.median(log_ratio_raw).item()
            mean_val = torch.mean(log_ratio_raw).item()
            max_val = torch.max(log_ratio_raw).item()
            min_val = torch.min(log_ratio_raw).item()
            
            # 记录到日志中（每步都记录）
            self.log("debug/log_ratio_clip_fraction", clip_fraction, on_step=True, on_epoch=False)
            self.log("debug/log_ratio_median", median_val, on_step=True, on_epoch=False)
            self.log("debug/log_ratio_mean", mean_val, on_step=True, on_epoch=False)
            self.log("debug/log_ratio_max", max_val, on_step=True, on_epoch=False)
            self.log("debug/log_ratio_min", min_val, on_step=True, on_epoch=False)
        
        # No longer clamp log_ratio here. Stability is handled by KL penalty and sigma clamping.
        ratio = torch.exp(log_ratio_raw)

        unclipped = ratio * A
        clipped = torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * A
        return -torch.min(unclipped, clipped).mean()

    def _gaussian_kl(
        self,
        mu_new: torch.Tensor,
        logsig_new: torch.Tensor,
        mu_old: torch.Tensor,
        logsig_old: torch.Tensor,
    ) -> torch.Tensor:
        """
        对角高斯 KL(new || old)：逐样本逐维求和，然后对batch取均值
        """
        var_new = (logsig_new.exp()) ** 2
        var_old = (logsig_old.exp()) ** 2
        # 增加一个小的epsilon防止var_old为0
        t1 = (var_new + (mu_old - mu_new) ** 2) / (var_old + 1e-8)
        t2 = 2.0 * (logsig_old - logsig_new)
        kl = 0.5 * (t1 + t2 - 1.0)
        return kl.sum(dim=-1).mean()

