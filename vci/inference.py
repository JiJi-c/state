import os
import logging
import torch

from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from vci.model import LitUCEModel
from vci.train.trainer import get_ESM2_embeddings
from vci.data import H5adDatasetSentences, VCIDatasetSentenceCollator


log = logging.getLogger(__name__)


class Inference():

    def __init__(self, cfg):
        self._vci_conf = cfg
        self.model = None
        self.collator = None

    def load_model(self, checkpoint):
        # Load and initialize model for eval

        self.model = LitUCEModel.load_from_checkpoint(checkpoint)
        all_pe = get_ESM2_embeddings(self._vci_conf)
        all_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.model.device)
        self.model.binary_decoder.requires_grad = False
        self.model.eval()

    def create_dataloader(self,
                          datasets,
                          shape_dict,
                          batch_size=32,
                          workers=1,
                          data_dir=None):
        if data_dir:
            self._vci_conf.dataset.data_dir = data_dir

        dataset = H5adDatasetSentences(self._vci_conf,
                                       datasets=datasets,
                                       shape_dict=shape_dict)
        sentence_collator = VCIDatasetSentenceCollator(self._vci_conf)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=sentence_collator,
                                num_workers=workers,
                                persistent_workers=True)
        return dataloader

    def encode(self, dataloader):
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                torch.cuda.empty_cache()
                batch_sentences = batch[0].to(self.model.device)
                mask = batch[1].to(self.model.device)

                batch_sentences = self.model.pe_embedding(batch_sentences.long())
                batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
                gene_output, embedding = self.model(batch_sentences, mask=mask)
                embeddings = embedding.detach().cpu().numpy()

                yield embeddings

    def decode(self, adata_path: str, emb_key: str):
        # X = self.model.pe_embedding(X.long())
        # X = self.model.gene_embedding_layer(X)
        # embedding = embedding.unsqueeze(1).repeat(1, X.shape[1], 1)
        # combine = torch.cat((X, embedding), dim=2)
        # decs = self.model.binary_decoder(combine).squeeze()
        pass
