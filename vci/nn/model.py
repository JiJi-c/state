import warnings
warnings.filterwarnings("ignore")

import math
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from torch import nn, Tensor
from torch.nn import (TransformerEncoder,
                      TransformerEncoderLayer,
                      BCEWithLogitsLoss)
from vci.utils import compute_gene_overlap_cross_pert, compute_pearson_delta, compute_perturbation_ranking_score


import sys
sys.path.append('../')
import torch
import lightning as L

from tqdm.auto import tqdm
from typing import Any
from torch.optim.lr_scheduler import (ChainedScheduler,
                                      LinearLR,
                                      CosineAnnealingLR,
                                      ReduceLROnPlateau)
from vci.data import create_dataloader
from vci.utils import compute_gene_overlap_cross_pert, get_embedding_cfg
from vci.eval.emb import cluster_embedding
from .loss import WassersteinLoss, KLDivergenceLoss, MMDLoss


# if flash-attn package is installed and available
from vci.nn.flash_transformer import FlashTransformerEncoderLayer
from vci.nn.flash_transformer import FlashTransformerEncoder


class SkipBlock(nn.Module):
    def __init__(self, in_features):
        """
        Given input X of size in_features
        - out = layernorm(x + MLP(MLP(X))

        """
        super().__init__()
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features*2, bias=True)
        self.dense = nn.Linear(in_features*2, in_features, bias=True)
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp \
            (torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LitUCEModel(L.LightningModule):
    def __init__(self, token_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, output_dim:int, dropout: float = 0.0,
                 warmup_steps: int = 0,
                 compiled: bool = False,
                 max_lr=4e-4,
                 emb_cnt=145469, emb_size=5120, cfg=None,
                 collater=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.compiled = compiled
        self.model_type = 'Transformer'
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.max_lr = max_lr
        self.collater = collater
        # Encodes Tokens
        self.encoder = nn.Sequential(nn.Linear(token_dim, d_model, bias=True),
                                     nn.LayerNorm(d_model), # Moved before activation
                                     nn.SiLU(), # Changed to SiLU
                                    )

        # Check the configuration flag whether to use Flash Attention
        use_flash = getattr(self.cfg.model, "use_flash_attention", False)
        if use_flash and FlashTransformerEncoderLayer is not None:
            print('!!! Using Flash Attention !!!')
            # Create a list of FlashTransformerEncoderLayer instances
            layers = [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout)
                      for _ in range(nlayers)]
            self.transformer_encoder = FlashTransformerEncoder(layers)
        else:
            # Fallback to the standard PyTorch TransformerEncoderLayer
            encoder_layer = TransformerEncoderLayer(d_model,
                                                       nhead,
                                                       d_hid,
                                                       dropout=dropout,
                                                       batch_first=True,
                                                       activation="gelu")
            self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

        self.d_model = d_model
        self.dropout = dropout

        self.decoder = nn.Sequential(SkipBlock(d_model),
                                     nn.Linear(d_model, output_dim, bias=True),
                                    )

        if compiled:
            self.decoder = torch.compile(self.decoder)

        count_info = 1 if self.cfg.model.rda else 0
        self.binary_decoder = nn.Sequential(
            SkipBlock(output_dim + d_model + count_info),
            SkipBlock(output_dim + d_model + count_info),
            nn.Linear(output_dim + d_model + count_info, 1, bias=True)
        )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)

        # Encodes Tokens for Decoder
        self.gene_embedding_layer = self.encoder # reuse this layer

        if compiled:
            self.gene_embedding_layer = torch.compile(self.gene_embedding_layer)

        self.pe_embedding = None # TODO: make this cleaner for the type checker, right now it gets set externally after model init
        self.step_ctr = 0

        self.true_top_genes = None
        self.protein_embeds = None

        self._last_val_de_check = 0
        self._last_val_perturbation_check = 0

    def _compute_embedding_for_batch(self, batch):
        batch_sentences = batch[0].to(self.device)
        X = batch[1].to(self.device)
        Y = batch[2]
        batch_weights = batch[4]
        mask = batch[5]
        mask = mask.to(torch.bool)

        # convert the cell sentence and task sentence into embeddings
        batch_sentences = self.pe_embedding(batch_sentences)
        X = self.pe_embedding(X)

        # Normalize token outputs now # TODO YANAY EXPERIMENT WITH REMOVING THIS
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)

        # Add a learnable CLS token to the beginning of the sentence
        batch_sentences[:, 0, :] = self.cls_token.expand(batch_sentences.size(0), -1)

        # mask out the genes embeddings that appear in the task sentence
        if self.cfg.model.rda:
            total_counts = batch[6].to(self.device)
            _, embedding = self.forward(batch_sentences, mask=mask, total_counts=total_counts)
        else:
            _, embedding = self.forward(batch_sentences, mask=mask)

        X = self.gene_embedding_layer(X)
        return X, Y, batch_weights, embedding

    def get_gene_embedding(self, genes):
        if self.protein_embeds is None:
            self.protein_embeds = torch.load(get_embedding_cfg(self.cfg).all_embeddings)

        protein_embeds = [self.protein_embeds[x] \
                          if x in self.protein_embeds else torch.zeros(get_embedding_cfg(self.cfg).size) for x in genes]
        protein_embeds = torch.stack(protein_embeds).to(self.device)
        return self.gene_embedding_layer(protein_embeds)

    @staticmethod
    def resize_batch(cell_embeds, task_embeds, task_counts=None):
        A = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)
        B = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)
        if task_counts is not None:
            reshaped_counts = task_counts.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            mlp_input = torch.cat((A, B, reshaped_counts), dim=-1)
        else:
            # Original behavior if total_counts is None
            mlp_input = torch.cat([A, B], dim=-1)  # (batch_size, num_genes, 2*embed_dim)
        return mlp_input

    def _predict_exp_for_adata(self, adata, dataset_name, pert_col):
        dataloader = create_dataloader(self.cfg,
                                       adata=adata,
                                       adata_name=dataset_name,
                                       shuffle=False,
                                       sentence_collator=self.collater,)
        gene_embeds = self.get_gene_embedding(adata.var.index)
        logprobs_batchs = []
        for batch in tqdm(dataloader,
                          position=0,
                          leave=True,
                          ncols=100,
                          desc=f"Embeddings for {dataset_name}",):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _, _, _, emb = self._compute_embedding_for_batch(batch)
            if self.cfg.model.rda:
                task_counts = batch[6].to(self.device)
            else:
                task_counts = None

            merged_embs = LitUCEModel.resize_batch(emb, gene_embeds, task_counts)
            logprobs_batch = self.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().numpy()
            logprobs_batchs.append(logprobs_batch.squeeze())

        logprobs_batchs = np.vstack(logprobs_batchs)
        probs_df = pd.DataFrame(logprobs_batchs)
        probs_df[pert_col] = adata.obs[pert_col].values

        # Read config properties
        k = self.cfg.validations.diff_exp.top_k_rank
        pert_col = self.cfg.validations.diff_exp.obs_pert_col
        non_targating_label = self.cfg.validations.diff_exp.obs_filter_label

        probs_df = probs_df.groupby(pert_col).mean()
        ctrl = probs_df.loc[non_targating_label].values
        pert_effects = np.abs(probs_df - ctrl)
        top_k_indices = np.argsort(pert_effects.values, axis=1)[:, -k:][:, ::-1]
        top_k_genes = np.array(adata.var.index)[top_k_indices]

        de_genes = pd.DataFrame(top_k_genes)
        de_genes.index = pert_effects.index.values

        return de_genes

    def forward(self, src: Tensor, mask: Tensor):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, ntoken]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        mask = mask.to(self.device)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        gene_output = self.decoder(output) # batch x seq_len x 128
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[:, 0, :] # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1) # Normalize.
        return gene_output, embedding

    def shared_step(self, batch, batch_idx):
        X, Y, batch_weights, embs = self._compute_embedding_for_batch(batch)
        total_counts = batch[6] if self.cfg.model.rda else None

        embs = embs.unsqueeze(1).repeat(1, X.shape[1], 1)
        if total_counts is not None:
            reshaped_counts = total_counts.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, X.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            combine = torch.cat((X, embs, reshaped_counts), dim=2)
        else:
            # Original behavior if total_counts is None
            combine = torch.cat((X, embs), dim=2)

        # concatenate the counts
        decs = self.binary_decoder(combine)

        if self.cfg.loss.name == 'cross_entropy':
            criterion = BCEWithLogitsLoss()
            target = Y
        elif self.cfg.loss.name == 'mse':
            criterion = nn.MSELoss()
            target = Y
        elif self.cfg.loss.name == 'wasserstein':
            criterion = WassersteinLoss()
            target = Y
        elif self.cfg.loss.name == 'kl_divergence':
            criterion = KLDivergenceLoss(apply_normalization=self.cfg.loss.normalization)
            target = batch_weights
        elif self.cfg.loss.name == 'mmd':
            criterion = MMDLoss(kernel="energy")
            target = Y
        else:
            raise ValueError(f"Loss {self.cfg.loss.name} not supported")

        loss = criterion(decs.squeeze(), target)
        sch = self.lr_schedulers()

        for scheduler in sch._schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
        sch._last_lr = [group['lr'] for group in sch._schedulers[-1].optimizer.param_groups]
        return loss

    @torch.compile(disable=True)
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("trainer/train_loss", loss)
        return loss

    @torch.compile(disable=True)
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("validation/val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return

        self.eval()
        current_step = self.global_step
        try:
            self.trainer.strategy.barrier()
            current_step = self.global_step
            if self.cfg.validations.diff_exp.enable:
                interval = self.cfg.validations.diff_exp.eval_interval_multiple * self.cfg.experiment.val_check_interval
                current_step = current_step - (current_step % 10)
                if current_step >= interval and current_step % interval == 0:
                    self._compute_val_de()

            self.trainer.strategy.barrier()
            if self.cfg.validations.perturbation.enable:
                interval = self.cfg.validations.perturbation.eval_interval_multiple * self.cfg.experiment.val_check_interval
                current_step = current_step - (current_step % 10)
                if current_step >= interval and current_step % interval == 0:
                    self._compute_val_perturbation(current_step)
        finally:
            self.train()

    def _compute_val_perturbation(self, current_step):
        self.trainer.strategy.barrier()

        adata = sc.read_h5ad(self.cfg.validations.perturbation.dataset)
        adata.X = np.log1p(adata.X)
        dataloader = create_dataloader(self.cfg,
                                       adata=adata,
                                       adata_name=self.cfg.validations.perturbation.dataset_name,
                                       shuffle=False,
                                       sentence_collator=self.collater)
        all_embs = []
        for batch in tqdm(dataloader,
                          position=0,
                          leave=True,
                          ncols=100,
                          desc=f"Embeddings for {self.cfg.validations.perturbation.dataset_name}",):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            _, _, _, emb = self._compute_embedding_for_batch(batch)
            all_embs.append(emb.cpu().detach().numpy())

        all_embs = np.concatenate(all_embs, axis=0)
        adata.obsm['X_emb'] = all_embs
        cluster_embedding(adata, current_step, emb_key='X_emb', use_pca=True, job_name=self.cfg.experiment.name)

        col_id = self.cfg.validations.perturbation.pert_col
        ctrl_label = self.cfg.validations.perturbation.ctrl_label

        # Track metrics across all cell types
        all_correlations = []
        all_ranking_scores = []

        for holdout_cell_type in adata.obs['cell_type'].unique():
            train_adata = adata[adata.obs['cell_type'] != holdout_cell_type]
            test_adata = adata[adata.obs['cell_type'] == holdout_cell_type]

            mean_pert_dfs = [] # store perturbation mean deltas
            # for each cell type, train a cell type mean perturbation model
            for cell_type in train_adata.obs['cell_type'].unique():
                adata_cell = train_adata[train_adata.obs['cell_type'] == cell_type]
                ctrl_adata = adata_cell[adata_cell.obs[col_id] == ctrl_label]
                pert_adata = adata_cell[adata_cell.obs[col_id] != ctrl_label]

                mean_ctrl = ctrl_adata.obsm['X_emb'].mean(axis=0)  # shape: (embedding_dim,)
                pert_offsets = pert_adata.obsm['X_emb'] - mean_ctrl

                pert_df = pd.DataFrame(
                    pert_offsets,
                    index=pert_adata.obs_names,
                    columns=[f"emb_{i}" for i in range(pert_offsets.shape[1])]
                )

                # Add the perturbation label column for grouping
                pert_df[col_id] = pert_adata.obs[col_id].values

                # Group by the perturbation label and compute the mean offset for this cell type
                mean_pert_dfs.append(pert_df.groupby(col_id).mean())

            # Average over all mean perturbations
            mean_pert_df = pd.concat(mean_pert_dfs).groupby(level=0).mean()
            pert_mean_offsets = {row: vals.values for row, vals in mean_pert_df.iterrows()}
            pert_mean_offsets.update({ctrl_label: np.zeros(mean_ctrl.shape[0])})

            # Create predicted and real AnnData objects for the test set
            pred_x = np.zeros_like(test_adata.obsm['X_emb']).copy()
            real_adata = sc.AnnData(
                X=test_adata.obsm['X_emb'],
                obs=test_adata.obs.copy(),
            )

            # Sample control cells and compute predictions
            ctrl_cells = test_adata[test_adata.obs[col_id] == ctrl_label].obs.index

            pert_exclude = set()
            for i, idx in enumerate(test_adata.obs.index):
                pert = test_adata.obs.loc[idx, col_id]
                if pert not in pert_mean_offsets:
                    # we only want to compute on shared perturbations so add this
                    # to the blacklist
                    pert_exclude.add(pert)
                    continue
                elif pert == ctrl_label:
                    # For control cells, use their own embedding
                    sampled_ctrl_idx = idx
                else:
                    # For perturbed cells, sample a random control cell
                    sampled_ctrl_idx = np.random.choice(ctrl_cells)

                # Get basal expression (control cell embedding)
                basal = test_adata[sampled_ctrl_idx].obsm['X_emb']

                # Add perturbation effect
                pert_effect = pert_mean_offsets[pert]
                pred = basal + pert_effect

                # Store prediction
                pred_x[i] = pred

            pred_adata = sc.AnnData(
                X=pred_x,
                obs=test_adata.obs.copy(),
            )

            # retain only the cells in pred and real that are not in the blacklist
            pred_adata = pred_adata[pred_adata.obs[col_id].isin(pert_mean_offsets.keys())]
            real_adata = real_adata[real_adata.obs[col_id].isin(pert_mean_offsets.keys())]
            ctrl_adata = pred_adata[pred_adata.obs[col_id] == ctrl_label]

            # Compute metrics for this cell type. In our case, ctrl_pred = ctrl_true
            # because we use the zero vector as perturbation for ctrl cells
            correlation = compute_pearson_delta(pred_adata.X, real_adata.X, ctrl_adata.X, ctrl_adata.X)
            ranking_score = compute_perturbation_ranking_score(pred_adata, real_adata)

            all_correlations.append(correlation)
            all_ranking_scores.append(ranking_score)

        # Log average metrics across all cell types
        self.log("validation/perturbation_correlation_mean", np.mean(all_correlations))
        self.log("validation/perturbation_ranking_mean", np.mean(all_ranking_scores))

    def _compute_val_de(self):
        if self.true_top_genes is None:
            de_val_adata = sc.read_h5ad(self.cfg.validations.diff_exp.dataset)
            sc.pp.log1p(de_val_adata)
            sc.tl.rank_genes_groups(de_val_adata,
                                    groupby=self.cfg.validations.diff_exp.obs_pert_col,
                                    reference=self.cfg.validations.diff_exp.obs_filter_label,
                                    rankby_abs=True,
                                    n_genes=self.cfg.validations.diff_exp.top_k_rank,
                                    method=self.cfg.validations.diff_exp.method,
                                    use_raw=False)
            self.true_top_genes = pd.DataFrame(de_val_adata.uns['rank_genes_groups']['names'])
            self.true_top_genes = self.true_top_genes.T
            del de_val_adata

        tmp_adata = sc.read_h5ad(self.cfg.validations.diff_exp.dataset)
        pred_exp = self._predict_exp_for_adata(tmp_adata,
                                               self.cfg.validations.diff_exp.dataset_name,
                                               self.cfg.validations.diff_exp.obs_pert_col)
        torch.cuda.synchronize()
        de_metrics = compute_gene_overlap_cross_pert(pred_exp, self.true_top_genes)
        self.log("validation/de", np.array(list(de_metrics.values())).mean())

    def configure_optimizers(self):
        # Marcel Code
        max_lr = self.max_lr
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=max_lr,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches * 2 # not sure why need to do this

        lr_schedulers = [LinearLR(optimizer,
                                  start_factor=self.cfg.optimizer.start,
                                  end_factor=self.cfg.optimizer.end,
                                  total_iters=int(0.03 * total_steps))]
        lr_schedulers.append(CosineAnnealingLR(optimizer,
                                               eta_min=max_lr * 0.3,
                                               T_max=total_steps))
        scheduler = ChainedScheduler(lr_schedulers)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'step',
                'frequency': 1
            }
        }
