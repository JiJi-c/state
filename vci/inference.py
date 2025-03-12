import os
import logging
import torch
import anndata

from tqdm import tqdm
from torch import nn

from vci.model import LitUCEModel
from vci.train.trainer import get_ESM2_embeddings
from vci.data import create_dataloader
from vci.utils import get_embedding_cfg

log = logging.getLogger(__name__)


class Inference():

    def __init__(self, cfg):
        self._vci_conf = cfg
        self.model = None
        self.collator = None
        self.protein_embeds = None

    def load_model(self, checkpoint):
        if self.model:
            raise ValueError('Model already initialized')

        # Load and initialize model for eval
        self.model = LitUCEModel.load_from_checkpoint(checkpoint)
        all_pe = get_ESM2_embeddings(self._vci_conf)
        all_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.model.device)
        self.model.binary_decoder.requires_grad = False
        self.model.eval()

        self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf))

    def init_from_model(self, model, protein_embeds=None):
        '''
        Intended for use during training
        '''
        self.model = model
        if protein_embeds:
            self.protein_embeds = protein_embeds
        else:
            self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf))

    def get_gene_embedding(self, genes):
        protein_embeds = [self.protein_embeds[x] \
                          if x in self.protein_embeds else torch.zeros(5120) for x in genes]
        protein_embeds = torch.stack(protein_embeds).to(self.model.device)
        return self.model.gene_embedding_layer(protein_embeds)

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

    def encode_adata(self,
                     input_adata_path: str,
                     output_adata_path: str,
                     emb_key: str = 'X_emb',
                     batch_size: int = 32,):
        shape_dict = self.__load_dataset_meta(input_adata_path)
        datasets = list(shape_dict.keys())

        dataloader = create_dataloader(datasets=datasets,
                                       shape_dict=shape_dict,
                                       batch_size=batch_size,
                                       data_dir=os.path.dirname(input_adata_path))

        for embeddings in self.encode(dataloader):
            self._save_data(input_adata_path, output_adata_path, emb_key, embeddings)

        return output_adata_path

    def decode(self, adata_path: str, emb_key: str, batch_size=32):
        adata = anndata.read_h5ad(adata_path)
        genes = adata.var.index
        cell_embs = adata.obsm[emb_key]
        cell_embs = torch.Tensor(cell_embs).to(self.model.device)

        gene_embeds = self.get_gene_embedding(genes)
        for i in tqdm(range(0, cell_embs.size(0), batch_size),
                      total=int(cell_embs.size(0) // batch_size)):
            cell_embeds_batch = cell_embs[i:i + batch_size]
            merged_embs = LitUCEModel.resize_batch(cell_embeds_batch, gene_embeds)
            logprobs_batch = self.model.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().numpy()
            yield logprobs_batch.squeeze()
