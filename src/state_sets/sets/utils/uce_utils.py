import logging
import warnings

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .uce_model import LitUCEModel
from .uce_model_old import TransformerModel

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class UCEGenePredictor:
    """
    Class to predict binary gene expression from UCE embeddings

    Usage:
    gene_predictor = UCEGenePredictor(device='cuda:0')
    cell_embeds = sc.read_h5ad('path_to_uce_embeds')
    gene_logprobs = gene_predictor.compute_gene_prob_group(genes, cell_embeds)

    """

    def __init__(
        self,
        model_loc=None,
        gene_idx_file="/large_storage/ctc/ML/data/cell/misc/new_tabula_sapiens_ep_8_sn_251656_nlayers_4_sample_size_1024_pe_idx.torch",
        token_file="/large_storage/ctc/ML/data/cell/misc/all_tokens.torch",
        protein_embeds_file="/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        token_dim=5120,
        emsize=1280,
        d_hid=5120,
        nlayers=33,
        nhead=20,
        dropout=0.05,
        output_dim=1280,
        device="cuda:0",
    ):
        self.device = device
        self.emsize = emsize
        self.gene_idx = torch.load(gene_idx_file)
        self.gene_idx = list(self.gene_idx.values())[0]
        self.token_file = token_file
        self.protein_embeds = torch.load(protein_embeds_file)

        # TODO: Remove the need for EMBED_MODEL_PATHS. Ideally the model should be
        # always decided by the caller. It is ok to have a default but not selection
        # in the lower level functions.
        self.model_loc = model_loc
        log.info(f"Loading model at {self.model_loc}")
        try:
            self.model = LitUCEModel.load_from_checkpoint(self.model_loc, strict=False)
            self.model = self.model.to(self.device)
        except:
            log.info(f"{self.model_loc} is not a lighting module. Trying TransformerModel...")
            # TODO: Cleanup required. This is for backward compaitibility with
            # older model files. E.g. /large_storage/ctc/ML/data/cell/misc/model_used_in_paper_33l_8ep_1024t_1280.torch
            self.model = TransformerModel(
                token_dim=token_dim,
                d_model=emsize,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=nlayers,
                dropout=dropout,
                output_dim=output_dim,
            )
            self.model = self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.model_loc), strict=False)

        self.all_pe = self.get_ESM2_embeddings()
        self.all_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(self.all_pe)
        self.model = self.model.eval()

    def get_ESM2_embeddings(self):
        all_pe = torch.load(self.token_file)
        if all_pe.shape[0] == 143574:
            torch.manual_seed(23)
            CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, self.token_dim))
            all_pe = torch.vstack((all_pe, CHROM_TENSORS))
            all_pe.requires_grad = False

        return all_pe

    def get_reduced_embeds(self, genes):
        # TODO: what should be done if self.protein_embeds does not contain the gene? E.g. 'non-targeting'
        return self.model.gene_embedding_layer(
            torch.stack([self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(5120) for x in genes]).to(
                self.device
            )
        )

    def get_MLP_input(self, cell_embed, task_embeds):
        A = task_embeds
        B = torch.Tensor(cell_embed).unsqueeze(1).repeat(1, task_embeds.shape[0]).T
        mlp_input = torch.cat([A, B], 1)
        return mlp_input

    def compute_gene_prob(self, genenames, cell_embed):
        task_embeds = self.get_reduced_embeds(genenames)
        mlp_input = self.get_MLP_input(cell_embed, task_embeds)
        mlp_input = mlp_input.to(self.device)
        return self.model.binary_decoder(mlp_input).detach().cpu()

    def compute_gene_prob_group(self, cell_embeds, genenames):
        all_logprobs = []
        for cell_embed in tqdm(cell_embeds, total=len(cell_embeds)):
            cell_embed = torch.Tensor(cell_embed).to(self.device)
            all_logprobs.append(self.compute_gene_prob(genenames, cell_embed))
        return all_logprobs

    def get_MLP_input_batched(self, cell_embeds, task_embeds):
        """
        Create the MLP input for multiple cell embeddings.
        """
        A = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)  # (batch_size, num_genes, embed_dim)
        B = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)  # (batch_size, num_genes, embed_dim)

        # Concatenating along the last dimension (embedding dimension)
        mlp_input = torch.cat([A, B], dim=-1)  # (batch_size, num_genes, 2*embed_dim)
        return mlp_input

    def compute_gene_prob_group_batched(self, cell_embeds, genenames, batch_size=32):
        """
        Compute gene probabilities for a group of cells (batch processing).
        """
        # Convert cell_embeds to a tensor if not already
        cell_embeds = torch.Tensor(cell_embeds).to(self.device)

        # Get task (gene) embeddings for the batch
        task_embeds = self.get_reduced_embeds(genenames)
        # task_embeds = task_embeds.unsqueeze(0).repeat(batch_size, 1, 1)  # (b, num_genes, embed_dim)

        # Create batches of cell embeddings
        for i in tqdm(
            range(0, cell_embeds.size(0), batch_size),
            total=int(np.ceil(cell_embeds.size(0) / batch_size)),
        ):
            cell_embeds_batch = cell_embeds[i : i + batch_size]
            mlp_input = self.get_MLP_input_batched(cell_embeds_batch, task_embeds)
            mlp_input = mlp_input.to(self.device)
            logprobs_batch = self.model.binary_decoder(mlp_input)
            del mlp_input
            if i == 0:
                logprobs = logprobs_batch.detach().cpu()
            else:
                logprobs = torch.cat([logprobs, logprobs_batch.detach().cpu()])

        return logprobs.squeeze()
