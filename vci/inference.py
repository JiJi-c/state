import os
import logging
import torch
import anndata
import h5py as h5

from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from torch import nn

from vci.nn.model import LitUCEModel
from vci.train.trainer import get_ESM2_embeddings
from vci.data import H5adDatasetSentences, VCIDatasetSentenceCollator, create_dataloader


log = logging.getLogger(__name__)


class Inference():

    def __init__(self, cfg):
        self.model = None
        self.collator = None
        self.protein_embeds = None

        if isinstance(cfg, str):
            self._vci_conf = OmegaConf.load(cfg)
        else:
            self._vci_conf = cfg

    def _save_data(self, input_adata_path, output_adata_path, obsm_key, data):
        '''
        Save data in the output file. This function addresses following cases:
        - output_adata_path does not exist:
          In this case, the function copies the rest of the input file to the
          output file then adds the data to the output file.
        - output_adata_path exists but the dataset does not exist:
          In this case, the function adds the dataset to the output file.
        - output_adata_path exists and the dataset exists:
          In this case, the function resizes the dataset and appends the data to
          the dataset.
        '''
        if not os.path.exists(output_adata_path):
            os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
            # Copy rest of the input file to output file
            with h5.File(input_adata_path) as input_h5f:
                with h5.File(output_adata_path, "a") as output_h5f:
                    # Replicate the input data to the output file
                    for _, obj in input_h5f.items():
                        input_h5f.copy(obj, output_h5f)
                    output_h5f.create_dataset(f'/obsm/{obsm_key}',
                                              chunks=True,
                                              data=data,
                                              maxshape=(None, data.shape[1]))
        else:
            with h5.File(output_adata_path, "a") as output_h5f:
                # If the dataset is added to an existing file that does not have the dataset
                if f'/obsm/{obsm_key}' not in output_h5f:
                    output_h5f.create_dataset(f'/obsm/{obsm_key}',
                                              chunks=True,
                                              data=data,
                                              maxshape=(None, data.shape[1]))
                else:
                    output_h5f[f'/obsm/{obsm_key}'].resize(
                        (output_h5f[f'/obsm/{obsm_key}'].shape[0] + data.shape[0]),
                        axis=0)
                    output_h5f[f'/obsm/{obsm_key}'][-data.shape[0]:] = data

    def load_model(self, checkpoint):
        if self.model:
            raise ValueError('Model already initialized')

        # Load and initialize model for eval
        self.model = LitUCEModel.load_from_checkpoint(checkpoint, strict=False)
        all_pe = get_ESM2_embeddings(self._vci_conf)
        all_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.model.device)
        self.model.binary_decoder.requires_grad = False
        self.model.eval()

        self.protein_embeds = torch.load(self._vci_conf.embeddings.esm2.embedding_file)

    def init_from_model(self, model, protein_embeds=None):
        '''
        Intended for use during training
        '''
        self.model = model
        if protein_embeds:
            self.protein_embeds = protein_embeds
        else:
            self.protein_embeds = torch.load(self._vci_conf.embeddings.esm2.embedding_file)

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

        adata = anndata.read_h5ad(input_adata_path)
        dataset_name = os.path.basename(input_adata_path).split('.')[0]
        dataloader = create_dataloader(self._vci_conf,
                                       adata=adata,
                                       adata_name=dataset_name)

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
