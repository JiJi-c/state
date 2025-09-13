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
from .new_reward import RewardEvaluator
from .validation_saver import ValidationSaver

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
        self.validation_saver = ValidationSaver()
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
       
        self.min_perts_for_pds = grpo_config.get("min_perts_for_pds", kwargs.get("min_perts_for_pds", 3))
        self.k_percentage = grpo_config.get("des_k_percentage", kwargs.get("des_k_percentage", 0.05))

        self.gene_list = pd.read_csv(self.gene_list_path, header=None)[0].tolist()

        self.des_baseline = 0.106
        self.pds_baseline = 0.516
        self.mae_baseline = 0.027
    
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
        self.pert2idx = {
            name: idx for idx, name in enumerate(pert_onehot_map.keys())
        }
        self.num_unique_perts = len(self.pert2idx)
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

    def _grpo_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """GRPO training step"""
        confidence_pred = None
        if self.confidence_token is not None:
            mu, log_sigma, confidence_pred = self.forward(batch, padded=padded)
        else:
            mu, log_sigma = self.forward(batch, padded=padded)
        
        B, L, G = mu.shape

        y_samples, logp_new = self.gaussian_decoder.sample(mu, log_sigma, k=self.k_samples)   # [K,B,L,G]  [K,B,L]
        logp_new = logp_new.mean(dim=1)
        y_real = batch["pert_cell_emb"].reshape(-1, L, G)
        basal_cells = batch["ctrl_cell_counts"].reshape(-1, L, G)
        pert_names = batch["pert_name"]

        with torch.no_grad():
            R, pds_r, mae_r, local_perts_num, mae = self.reward_evaluator.aggregate_rewards(
                y_samples=y_samples.detach(),      # [K, B, L, G]
                y_real=y_real.detach(),            # [B, L, G]
                basal_cells=basal_cells.detach(),  # [B, L, G]
                pert_names=pert_names,         # [B*L]
            )   

        A = self._grpo_advantage(R).mean(dim=1)  # [K, B, L]
        
        if self.mu_old_buf is None :
            logp_old = logp_new.detach()
            mu_old_use, logsig_old_use = mu.detach(), log_sigma.detach()
        else:
            with torch.no_grad():
                dist_old = Normal(self.mu_old_buf, self.logsig_old_buf.exp())
                logp_old = dist_old.log_prob(y_samples).sum(dim=-1).mean(dim=1)  # [K, B, L]
            mu_old_use, logsig_old_use = self.mu_old_buf, self.logsig_old_buf
        
        ppo_loss = self._ppo_clip_loss(logp_new, logp_old, A)   
        kl_loss = self._gaussian_kl(mu, log_sigma, mu_old_use, logsig_old_use)
        entropy = (log_sigma + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=-1).mean()
        loss_sup = F.l1_loss(mu, y_real)

        loss = (
            ppo_loss
            + self.alpha_kl * kl_loss
            - self.alpha_ent * entropy
            + self.alpha_sup * loss_sup
        )

        with torch.no_grad():
            self.mu_old_buf = mu.detach().clone()
            self.logsig_old_buf = log_sigma.detach().clone()
        
        # 9) Logging 
        mae_prediction = mae.mean()
        mae_scaled = (self.mae_baseline - mae_prediction) / self.mae_baseline
        mae_baseline_delta = (self.mae_baseline - mae_prediction) / self.mae_baseline

        pds_prediction = pds_r.mean()
        pds_scaled = (pds_prediction - self.pds_baseline) / (1 - self.pds_baseline)
        pds_baseline_delta = (pds_prediction - self.pds_baseline) / (1 - self.pds_baseline)

        S = 100 * (mae_scaled + pds_scaled) / 2
        self.log("train/mae/mae_predition", mae_prediction, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/mae/mae_scaled", mae_scaled, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/mae/mae_baseline_delta", mae_baseline_delta, on_step=True, on_epoch=False, prog_bar=True)

        self.log("train/pds/pds_predition", pds_prediction, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/pds/pds_scaled", pds_scaled, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/pds/pds_baseline_delta", pds_baseline_delta, on_step=True, on_epoch=False, prog_bar=True)

        self.log("train/Score", S, on_step=True, on_epoch=False, prog_bar=True)

        self.log("train/loss/total_reward", R.mean(), on_step=True, on_epoch=False)
        self.log("train/loss/total_loss", loss, on_step=True, on_epoch=False)
        self.log("train/loss/ppo_loss", ppo_loss, on_step=True, on_epoch=False)
        self.log("train/loss/kl_loss", kl_loss, on_step=True, on_epoch=False)
        self.log("train/loss/entropy", entropy, on_step=True, on_epoch=False)
        self.log("train/loss/supervised_loss", loss_sup, on_step=True, on_epoch=False)
      
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
        """training step"""
        if self.grpo_mode:
            return self._grpo_training_step(batch, batch_idx, padded)
        else:
            return self._original_training_step(batch, batch_idx, padded)

    @torch.no_grad()
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

        B, L, G =  pred.shape
        self.validation_saver.add_batch(
            predictions=pred.unsqueeze(0),
            targets=target.reshape(-1, L, G),
            basals=batch["ctrl_cell_counts"].reshape(-1, L, G),
            pert_names=batch["pert_name"]
        )
        # R, pds_r, mae_r, local_perts_num, mae ,des_prediction= self.reward_evaluator.aggregate_rewards(
        #     y_samples=pred.unsqueeze(0).detach(),      # [1, B, L, G]
        #     y_real=batch["pert_cell_emb"].reshape(-1, L, G).detach(),       # [B, L, G]
        #     basal_cells=batch["ctrl_cell_counts"].reshape(-1, L, G).detach(),  # [B, L, G]
        #     pert_names=batch["pert_name"],    # [B*L]
        #     mode="val"
        # )   
            
        # des_prediction, des_scores = self.reward_evaluator.calculate_des_reward_hpdex(
        #     y_pred=pred.unsqueeze(0).detach(),      # [1, B, L, G]
        #     basal_cells=batch["ctrl_cell_counts"].reshape(-1, L, G).detach(), # [B, L, G]
        #     pert_names=batch["pert_name"],     # [B*L]
        # )

        # mae_prediction = mae.mean()
        # mae_scaled = (self.mae_baseline - mae_prediction) / self.mae_baseline
        # mae_baseline_delta = (self.mae_baseline - mae_prediction) / self.mae_baseline

        # pds_prediction = pds_r.mean()
        # pds_scaled = (pds_prediction - self.pds_baseline) / (1 - self.pds_baseline)
        # pds_baseline_delta = (pds_prediction - self.pds_baseline) / (1 - self.pds_baseline)
        
        # des_prediction = des_prediction
        # des_scaled = (des_prediction - self.des_baseline) / (1 - self.des_baseline)
        # des_baseline_delta = (des_prediction - self.des_baseline) / (1 - self.des_baseline)

        # S = 100 * (mae_scaled + pds_scaled + des_scaled) / 3
        # self.log("val/mae/mae_predition", mae_prediction)
        # self.log("val/mae/mae_scaled", mae_scaled)
        # self.log("val/mae/mae_baseline_delta", mae_baseline_delta)

        # self.log("val/pds/pds_predition", pds_prediction)
        # self.log("val/pds/pds_scaled", pds_scaled)
        # self.log("val/pds/pds_baseline_delta", pds_baseline_delta)

        # self.log("val/des/des_predition", des_prediction)
        # self.log("val/des/des_scaled", des_scaled)
        # self.log("val/des/des_baseline_delta", des_baseline_delta)

        # self.log("val/Score", S, on_step=True, on_epoch=False, prog_bar=True)

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


    @torch.no_grad()
    def _grpo_validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """GRPO validation step"""
        confidence_pred = None
        if self.confidence_token is not None:
            mu, log_sigma, confidence_pred = self.forward(batch)
        else:
            mu, log_sigma = self.forward(batch)
        
        B, L, G = mu.shape

        self.validation_saver.add_batch(
            predictions=mu.unsqueeze(0),
            targets=batch["pert_cell_emb"].reshape(-1, L, G),
            basals=batch["ctrl_cell_counts"].reshape(-1, L, G),
            pert_names=batch["pert_name"]
        )
        
        # R, pds_r, mae_r, local_perts_num, mae ,des_prediction = self.reward_evaluator.aggregate_rewards(
        #     y_samples=mu.unsqueeze(0).detach(),      # [1, B, L, G]
        #     y_real=batch["pert_cell_emb"].reshape(-1, L, G).detach(),       # [B, L, G]
        #     basal_cells=batch["ctrl_cell_counts"].reshape(-1, L, G).detach(),  # [B, L, G]
        #     pert_names=batch["pert_name"],    # [B*L]
        #     mode="val"
        # )   
            
        # des_prediction, des_scores = self.reward_evaluator.calculate_des_reward_hpdex(
        #     y_pred=pred.unsqueeze(0).detach(),      # [1, B, L, G]
        #     basal_cells=batch["ctrl_cell_counts"].reshape(-1, L, G).detach(), # [B, L, G]
        #     pert_names=batch["pert_name"],     # [B*L]
        # )

        # mae_prediction = mae.mean()
        # mae_scaled = (self.mae_baseline - mae_prediction) / self.mae_baseline
        # mae_baseline_delta = (self.mae_baseline - mae_prediction) / self.mae_baseline

        # pds_prediction = pds_r.mean()
        # pds_scaled = (pds_prediction - self.pds_baseline) / (1 - self.pds_baseline)
        # pds_baseline_delta = (pds_prediction - self.pds_baseline) / (1 - self.pds_baseline)
        
        # des_prediction = des_prediction
        # des_scaled = (des_prediction - self.des_baseline) / (1 - self.des_baseline)
        # des_baseline_delta = (des_prediction - self.des_baseline) / (1 - self.des_baseline)

        # S = 100 * (mae_scaled + pds_scaled + des_scaled) / 3
        # self.log("val/mae/mae_predition", mae_prediction)
        # self.log("val/mae/mae_scaled", mae_scaled)
        # self.log("val/mae/mae_baseline_delta", mae_baseline_delta)

        # self.log("val/pds/pds_predition", pds_prediction)
        # self.log("val/pds/pds_scaled", pds_scaled)
        # self.log("val/pds/pds_baseline_delta", pds_baseline_delta)

        # self.log("val/des/des_predition", des_prediction)
        # self.log("val/des/des_scaled", des_scaled)
        # self.log("val/des/des_baseline_delta", des_baseline_delta)

        # self.log("val/Score", S, on_step=True, on_epoch=False, prog_bar=True)
        loss_sup = F.l1_loss(mu, batch["pert_cell_emb"].reshape(-1, L, G).detach()) 
        entropy = (log_sigma + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=-1).mean()
        
        self.log("val/loss/supervised_loss", loss_sup) 
        self.log("val/loss/entropy", entropy)

        if confidence_pred is not None:
            loss_target = loss_sup.clone() * 10  
            if confidence_pred.dim() == 2:
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("val/confidence_loss", confidence_loss)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        if self.grpo_mode:
            return self._grpo_validation_step(batch, batch_idx)
        else:
            return self._original_validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        """Validation epoch结束时对所有数据跑reward"""
        predictions, targets, basals, pert_names = self.validation_saver.get_batches()
        self.validation_saver.clear()

        R, pds_r, mae_r, local_perts_num, mae ,des_prediction = self.reward_evaluator.aggregate_rewards(
            y_samples=predictions,
            y_real=targets,
            basal_cells=basals,
            pert_names=pert_names,
            mode="val"
        )
        
        mae_prediction = mae.mean()
        mae_scaled = (self.mae_baseline - mae_prediction) / self.mae_baseline
        mae_baseline_delta = (self.mae_baseline - mae_prediction) / self.mae_baseline

        pds_prediction = pds_r.mean()
        pds_scaled = (pds_prediction - self.pds_baseline) / (1 - self.pds_baseline)
        pds_baseline_delta = (pds_prediction - self.pds_baseline) / (1 - self.pds_baseline)
        
        des_prediction = des_prediction
        des_scaled = (des_prediction - self.des_baseline) / (1 - self.des_baseline)
        des_baseline_delta = (des_prediction - self.des_baseline) / (1 - self.des_baseline)

        S = 100 * (mae_scaled if mae_scaled > 0 else 0 + pds_scaled if pds_scaled > 0 else 0 + des_scaled if des_scaled > 0 else 0) / 3
        self.log("val/mae/mae_predition", mae_prediction)
        self.log("val/mae/mae_scaled", mae_scaled)
        self.log("val/mae/mae_baseline_delta", mae_baseline_delta)

        self.log("val/pds/pds_predition", pds_prediction)
        self.log("val/pds/pds_scaled", pds_scaled)
        self.log("val/pds/pds_baseline_delta", pds_baseline_delta)

        self.log("val/des/des_predition", des_prediction)
        self.log("val/des/des_scaled", des_scaled)
        self.log("val/des/des_baseline_delta", des_baseline_delta)

        self.log("val/Score", S)


    def _grpo_test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """GRPO test step"""
        with torch.no_grad():
            # 1) 前向传播获得mu, log_sigma
            confidence_pred = None
            if self.confidence_token is not None:
                mu, log_sigma, confidence_pred = self.forward(batch, padded=False)
            else:
                mu, log_sigma = self.forward(batch, padded=False)
            
            # 2) 标准化形状处理，测试时通常是单个batch，所以reshape为[1, -1, G]
            B, S, G = mu.shape
            mu_flat = mu.reshape(-1, G)                 # [B*S, G]
            log_sigma_flat = log_sigma.reshape(-1, G)   # [B*S, G]
            
            y_real = batch["pert_cell_emb"]
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
        """original test step"""
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

        with torch.no_grad():
            pred_flat = pred.reshape(-1, self.output_dim)  # [N, G]
            
            y_real = batch["pert_cell_emb"]
            basal_cells = batch["ctrl_cell_counts"]
            
            pert_names = batch["pert_name"]
            
            y_samples = pred_flat.unsqueeze(0)  # [1, N, G]
            
            R, pds_r, des_r, local_perts_num = self.reward_evaluator.aggregate_rewards(
                y_samples=y_samples,      # [1, N, G]
                y_real=y_real,            # [N, G]
                basal_cells=basal_cells,  # [N, G]
                pert_names=pert_names,         # [N]
            )
            
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
        """test step"""
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
        """original predict step"""
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
        """GRPO predict step"""
        with torch.no_grad():
            confidence_pred = None
            if self.confidence_token is not None:
                mu, log_sigma, confidence_pred = self.forward(batch, padded=padded)
            else:
                mu, log_sigma = self.forward(batch, padded=padded)
            
            B, S, G = mu.shape
            mu_flat = mu.reshape(-1, G)  # [B*S, G]

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
            }

            if confidence_pred is not None:
                output_dict["confidence_pred"] = confidence_pred

            return output_dict

    
    def _grpo_advantage(self, R: torch.Tensor) -> torch.Tensor:
        """
        R: [K,B,L] -> A: [K,B,L] 组内标准化优势，停止梯度回传
        """
        mu = R.mean(0, keepdim=True)
        sd = R.std(0, keepdim=True) + 1e-8
        A = (R - mu) / sd
        return A.detach()

    def _ppo_clip_loss(
        self, logp_new: torch.Tensor, logp_old: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """
        所有量均为 [K,N]
        """
        # 计算原始 log_ratio
        log_ratio_raw = logp_new - logp_old # [K,B,L]
        clamp_ratio = 5
        with torch.no_grad():
            # 统计被截断的比例
            num_clipped = (log_ratio_raw > clamp_ratio).sum().item()
            total_elements = log_ratio_raw.numel()
            clip_ratio = num_clipped / total_elements
            
            # 计算中位数和其他统计量
            median_val = torch.median(log_ratio_raw).item()
            mean_val = torch.mean(log_ratio_raw).item()
            max_val = torch.max(log_ratio_raw).item()
            min_val = torch.min(log_ratio_raw).item()
            
            # 记录到日志中（每步都记录）
            self.log("debug/log_ratio_clip_ratio", clip_ratio, on_step=True, on_epoch=False)
            self.log("debug/log_ratio_median", median_val, on_step=True, on_epoch=False)
            self.log("debug/log_ratio_mean", mean_val, on_step=True, on_epoch=False)
            self.log("debug/log_ratio_max", max_val, on_step=True, on_epoch=False)
            self.log("debug/log_ratio_min", min_val, on_step=True, on_epoch=False)
        
        # 应用截断
        log_ratio = log_ratio_raw.clamp(max=clamp_ratio)
        ratio = torch.exp(log_ratio)

        unclipped = ratio * A
        clipped = torch.clamp(ratio,0 if 1.0 - self.ppo_eps<0 else 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * A
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

