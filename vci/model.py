import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import scanpy as sc
from torch import nn, Tensor
from torch.nn import (TransformerEncoder,
                      TransformerEncoderLayer,
                      BCEWithLogitsLoss)

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
from vci.utils import compute_gene_overlap_cross_pert
# from vci.loss import wasserstein_loss
from vci.loss import WassersteinLoss


def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )


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
                 emb_cnt=145469, emb_size=5120, cfg=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.compiled = compiled
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.max_lr = max_lr
        # Encodes Tokens
        self.encoder = nn.Sequential(nn.Linear(token_dim, d_model, bias=True),
                                     nn.LayerNorm(d_model), # Moved before activation
                                     nn.SiLU(), # Changed to SiLU
                                    )

        encoder_layers = TransformerEncoderLayer(d_model,
                                                 nhead,
                                                 d_hid,
                                                 dropout=dropout,
                                                 batch_first=True,
                                                 activation="gelu") # switch to gelu activation
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

        self.d_model = d_model
        self.dropout = dropout

        self.decoder = nn.Sequential(SkipBlock(d_model),
                                     nn.Linear(d_model, output_dim, bias=True),
                                    )

        if compiled:
            self.decoder = torch.compile(self.decoder)

        self.binary_decoder = nn.Sequential(
            SkipBlock(output_dim + d_model),
            SkipBlock(output_dim + d_model),
            nn.Linear(output_dim + d_model, 1, bias=True)
        )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)

        # Encodes Tokens for Decoder
        self.gene_embedding_layer = self.encoder # reuse this layer

        if compiled:
            self.gene_embedding_layer = torch.compile(self.gene_embedding_layer)

        self.pe_embedding = None
        self.step_ctr = 0

        self.true_top_genes = None
        self.protein_embeds = None

    def _compute_embedding_for_batch(self, batch):
        batch_sentences = batch[0].to(self.device)
        mask = batch[1].to(self.device)
        X = batch[2].to(self.device)
        Y = batch[3]

        batch_sentences = self.pe_embedding(batch_sentences.long())
        X = self.pe_embedding(X.long())

        # Normalize token outputs now # TODO YANAY EXPERIMENT WITH REMOVING THIS
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
        _, embedding = self.forward(batch_sentences, mask=mask)

        X = self.gene_embedding_layer(X)
        return X, Y, embedding

    def get_gene_embedding(self, genes):
        if self.protein_embeds is None:
            self.protein_embeds = torch.load(self.cfg.embeddings.esm2.embedding_file)
        protein_embeds = [self.protein_embeds[x] \
                          if x in self.protein_embeds else torch.zeros(5120) for x in genes]
        protein_embeds = torch.stack(protein_embeds).to(self.device)
        return self.gene_embedding_layer(protein_embeds)

    @staticmethod
    def resize_batch(cell_embeds, task_embeds):
        A = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)
        B = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)
        mlp_input = torch.cat([A, B], dim=-1)  # (batch_size, num_genes, 2*embed_dim)
        return mlp_input

    def _predict_exp_for_adata(self, adata, dataset_name, pert_col):
        dataloader = create_dataloader(self.cfg,
                                       adata=adata,
                                       adata_name=dataset_name)
        gene_embeds = self.get_gene_embedding(adata.var.index)
        logprobs_batchs = []
        for batch in tqdm(dataloader,
                          position=0,
                          leave=True,
                          ncols=100,
                          desc=f"Embeddings for {dataset_name}",):
            torch.cuda.empty_cache()
            _, _, emb = self._compute_embedding_for_batch(batch)

            merged_embs = LitUCEModel.resize_batch(emb, gene_embeds)
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
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        gene_output = self.decoder(output) # batch x seq_len x 128
        # embedding = torch.mul(gene_output, mask.t().unsqueeze(2)).sum(0) # average over non zero genes
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[:, 0, :] # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1) # Normalize.
        return gene_output, embedding

    def shared_step(self, batch, batch_idx):
        # criterion = BCEWithLogitsLoss()
        criterion = WassersteinLoss(self.d_model)
        X, Y, embs = self._compute_embedding_for_batch(batch)
        embs = embs.unsqueeze(1).repeat(1, X.shape[1], 1)
        combine = torch.cat((X, embs), dim=2)
        decs = self.binary_decoder(combine)
        loss = criterion(input=decs.squeeze(), target=Y)
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

        current_step = self.global_step
        if self.cfg.validations.diff_exp.enable:
            interval = self.cfg.validations.diff_exp.eval_interval_multiple * self.cfg.experiment.val_check_interval

            # WAR for the global step being off by 1 when the training is restarted
            current_step = current_step - (current_step % 10)
            if current_step < interval or current_step % interval != 0:
                # Not to run after every eval epoch and before starting the training
                return

            if self.true_top_genes is None:
                de_val_adata = sc.read_h5ad(self.cfg.validations.diff_exp.dataset)
                sc.tl.rank_genes_groups(de_val_adata,
                                        groupby=self.cfg.validations.diff_exp.obs_pert_col,
                                        reference=self.cfg.validations.diff_exp.obs_filter_label,
                                        rankby_abs=True,
                                        n_genes=self.cfg.validations.diff_exp.top_k_rank,
                                        method=self.cfg.validations.diff_exp.method)
                self.true_top_genes = pd.DataFrame(de_val_adata.uns['rank_genes_groups']['names'])
                self.true_top_genes = self.true_top_genes.T
                del de_val_adata

            tmp_adata = sc.read_h5ad(self.cfg.validations.diff_exp.dataset)
            pred_exp = self._predict_exp_for_adata(tmp_adata,
                                                   self.cfg.validations.diff_exp.dataset_name,
                                                   self.cfg.validations.diff_exp.obs_pert_col)

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
