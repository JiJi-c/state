import os
import sys
import h5py
import hydra
import logging
import numpy as np

from omegaconf import DictConfig

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

sys.path.append('.')
from vci.data import H5adDatasetSentences, VCIDatasetSentenceCollator
from vci.model import LitUCEModel
from vci.train.trainer import get_ESM2_embeddings


log = logging.getLogger(__name__)


checkpoint = '/scratch/ctc/vci/checkpoint/pre_1/exp_pre_1_layers_4_dmodel_512_samples_2048_max_lr_0.0004_op_dim_256-epoch=1-step=91280.ckpt'
emb_file = '/large_storage/goodarzilab/userspace/mohsen/VCI/ZeroShotCellType_Replogle_jurkat_edist_train/predictions/scvi_pred_map_random.h5ad'

def update_data(embeddings, losses, f):
    embeddings = np.vstack(embeddings)
    losses = np.vstack(losses)
    log.info(f'Loss {losses.mean()}')
    f['embedding'].resize((f['embedding'].shape[0] + embeddings.shape[0]), axis=0)
    f['embedding'][-embeddings.shape[0]:] = embeddings

    f['loss'].resize((f['loss'].shape[0] + losses.shape[0]), axis=0)
    f['loss'][-losses.shape[0]:] = losses


@hydra.main(config_path="../conf", config_name="defaults")
def main(cfg: DictConfig):
    log.info(f'Embeddings will be saved at {emb_file}')
    log.info(f'Loading checkpoint from {checkpoint}')
    model = LitUCEModel.load_from_checkpoint(checkpoint)
    all_pe = get_ESM2_embeddings(cfg)
    all_pe.requires_grad= False
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model.pe_embedding.to(model.device)
    model.binary_decoder.requires_grad= False
    model.eval()

    # Setup Data
    dataset = H5adDatasetSentences(cfg)
    multi_dataset_sentence_collator = VCIDatasetSentenceCollator(cfg)

    # Make the dataloader outside of the
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=multi_dataset_sentence_collator,
                            num_workers=1,
                            persistent_workers=True)

    embeddings = []
    losses = []
    f = None
    with torch.no_grad():
        i = 1
        criterion = BCEWithLogitsLoss()
        for i, batch in enumerate(dataloader):
            torch.cuda.empty_cache()

            batch_sentences = batch[0].to(model.device)
            mask = batch[1].to(model.device)
            X = batch[2].to(model.device)
            Y = batch[3].to(model.device).squeeze()
            # dataset_nums = batch[5].to(model.device)

            batch_sentences = model.pe_embedding(
                batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
            gene_output, embedding = model(batch_sentences, mask=mask)

            embeddings.append(embedding.detach().cpu().numpy())

            # dataset_num_emb = model.dataset_num_embedding(dataset_nums)
            X = model.pe_embedding(X.long())
            X = model.gene_embedding_layer(X)

            embedding = embedding.unsqueeze(1).repeat(1, X.shape[1], 1)
            # dataset_num_emb = dataset_num_emb.unsqueeze(1).repeat(1, X.shape[1], 1)

            combine = torch.cat((X, embedding), dim=2)
            decs = model.binary_decoder(combine).squeeze()

            loss = criterion(input=decs, target=Y)
            losses.append(loss.detach().cpu().numpy())
            i = i + 1

            if i % 10 == 0:
                if not os.path.exists(emb_file):
                    embeddings = np.vstack(embeddings)
                    losses = np.vstack(losses)
                    with h5py.File(emb_file, "a") as h5f:
                        h5f.create_dataset('embedding',
                                        chunks=True,
                                        data=embeddings,
                                        maxshape=(None, 256))
                        h5f.create_dataset('loss',
                                        chunks=True,
                                        data=losses,
                                        maxshape=(None, 1))
                else:
                    with h5py.File(emb_file, "a") as h5f:
                        update_data(embeddings, losses, h5f)

                embeddings = []
                losses = []

        # if embeddings and losses:
        #     with h5py.File(emb_file, "a") as h5f:
        #         update_data(embeddings, losses, h5f)


if __name__ == "__main__":
    main()
