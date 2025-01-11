import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, BCEWithLogitsLoss

import sys
sys.path.append('../')
from typing import Any
import torch
import lightning as L
from torch.optim.lr_scheduler import (ChainedScheduler,
                                      LinearLR,
                                      CosineAnnealingLR,
                                      ReduceLROnPlateau)


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
        criterion = BCEWithLogitsLoss()
        batch_sentences = batch[0]
        mask = batch[1]
        cell_outputs_X_pe = batch[2]
        cell_outputs_Y = batch[3]
        # dataset_nums = batch[5]

        batch_sentences = self.pe_embedding(
            batch_sentences.long())
        cell_outputs_X_pe = self.pe_embedding(
            cell_outputs_X_pe.long())
        # dataset_num_emb = self.dataset_num_embedding(dataset_nums) # batch x emb shap

        batch_sentences = nn.functional.normalize(batch_sentences, dim=2) # Normalize token outputs now # TODO YANAY EXPERIMENT WITH REMOVING THIS
        _, embedding = self.forward(batch_sentences, mask=mask)

        X = cell_outputs_X_pe
        Y = cell_outputs_Y
        X = self.gene_embedding_layer(X)
        embs = embedding.unsqueeze(1).repeat(1, X.shape[1], 1)
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
