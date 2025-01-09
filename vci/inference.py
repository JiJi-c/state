import logging
import torch
import anndata

from tqdm import tqdm
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
        self.protein_embeds = None

    def load_model(self, checkpoint):
        # Load and initialize model for eval

        self.model = LitUCEModel.load_from_checkpoint(checkpoint)
        all_pe = get_ESM2_embeddings(self._vci_conf)
        all_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.model.device)
        self.model.binary_decoder.requires_grad = False
        self.model.eval()

        self.protein_embeds = torch.load(self._vci_conf.embeddings.esm2.embedding_file)

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

    def get_gene_embedding(self, genes):
        protein_embeds = [self.protein_embeds[x] \
                          if x in self.protein_embeds else torch.zeros(5120) for x in genes]
        protein_embeds = torch.stack(protein_embeds).to(self.model.device)
        return self.model.gene_embedding_layer(protein_embeds)

    def resize_batch(self, cell_embeds, task_embeds):
        A = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)  # (batch_size, num_genes, embed_dim)
        B = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)  # (batch_size, num_genes, embed_dim)
        # Concatenating along the last dimension (embedding dimension)
        mlp_input = torch.cat([A, B], dim=-1)  # (batch_size, num_genes, 2*embed_dim)
        return mlp_input

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

    def decode(self, adata_path: str, emb_key: str, batch_size=32):
        adata = anndata.read_h5ad(adata_path)
        genes = adata.var.index
        cell_embs = adata.obsm[emb_key]
        cell_embs = torch.Tensor(cell_embs).to(self.model.device)

        gene_embeds = self.get_gene_embedding(genes)
        for i in tqdm(range(0, cell_embs.size(0), batch_size),
                      total=int(cell_embs.size(0) // batch_size)):
            cell_embeds_batch = cell_embs[i:i + batch_size]
            merged_embs = self.resize_batch(cell_embeds_batch, gene_embeds)
            logprobs_batch = self.model.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().numpy()
            yield logprobs_batch.squeeze()
