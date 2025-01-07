import os
import sys
import h5py as h5
import hydra
import logging
import numpy as np

from pathlib import Path
from omegaconf import DictConfig

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

sys.path.append('../')
from vci.data import H5adDatasetSentences, VCIDatasetSentenceCollator
from vci.model import LitUCEModel
from vci.train.trainer import get_ESM2_embeddings


log = logging.getLogger(__name__)


checkpoint = '/scratch/ctc/vci/checkpoint/pre_1/exp_pre_1_layers_4_dmodel_512_samples_2048_max_lr_0.0004_op_dim_256-epoch=1-step=91280.ckpt'
input_file = '/large_storage/ctc/userspace/rajesh.ilango/perf_bench/dataset/rpe1.h5ad'
emb_file = '/large_storage/ctc/userspace/rajesh.ilango/perf_bench/dataset/vci_pred_rpe1.h5ad'


@hydra.main(config_path="../conf", config_name="defaults")
def main(cfg: DictConfig):

    # Load and initialize model
    model = LitUCEModel.load_from_checkpoint(checkpoint)
    all_pe = get_ESM2_embeddings(cfg)
    all_pe.requires_grad= False
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model.pe_embedding.to(model.device)
    model.binary_decoder.requires_grad= False
    model.eval()

    dataset = Path(input_file).stem
    collator = VCIDatasetSentenceCollator(cfg=cfg)

    collator.dataset_to_protein_embeddings = \
        torch.load('/large_storage/ctc/userspace/rajesh.ilango/perf_bench/dataset/gene_embidx_mapping.torch')

    embeddings = []
    # Setup Data
    with h5.File(input_file) as h5f:
        attrs = dict(h5f['X'].attrs)
        if attrs['encoding-type'] == 'csr_matrix':
            num_cells = attrs['shape'][0]
            num_genes = attrs['shape'][1]
        elif attrs['encoding-type'] == 'array':
            num_cells = h5f['X'].shape[0]
            num_genes = h5f['X'].shape[1]
        else:
            raise ValueError('Input file contains count mtx in non-csr matrix')

        for idx in range(num_cells):
            if attrs['encoding-type'] == 'csr_matrix':
                indptrs = h5f["/X/indptr"]
                start_ptr = indptrs[idx]
                end_ptr = indptrs[idx + 1]
                sub_data = torch.tensor(
                    h5f["/X/data"][start_ptr:end_ptr],
                    dtype=torch.int32)
                sub_indices = torch.tensor(
                    h5f["/X/indices"][start_ptr:end_ptr],
                    dtype=torch.int32)

                counts = torch.sparse_csr_tensor(
                    [0,],
                    sub_indices,
                    sub_data,
                    (1, num_genes),
                )
                counts = counts.to_dense()
            else:
                counts = torch.tensor(h5f["X"][idx]).unsqueeze(0)

            batch_sentences, mask, X, Y = collator.sample_cell_sentences(counts, dataset)
            batch_sentences = batch_sentences.to(model.device)
            mask = mask.to(model.device)
            X = X.to(model.device)
            Y = Y.to(model.device)

            batch_sentences = model.pe_embedding(
                batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
            gene_output, embedding = model(batch_sentences, mask=mask)
            embedding = embedding.squeeze()
            embeddings.append(embedding.detach().cpu().numpy())


    with h5.File(input_file) as input_h5f:
        with h5.File(emb_file, "a") as emb_h5f:
            for name, obj in input_h5f.items():
                input_h5f.copy(obj, emb_h5f)
            emb_h5f.create_dataset('/obsm/X_emb',
                            chunks=True,
                            data=embeddings,)


if __name__ == "__main__":
    main()
