import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from utils import UCEGenePredictor

# set up logger
import logging

logger = logging.getLogger(__name__)


class DecoderInterface(ABC):
    """
    An interface for decoding latent model outputs into DE genes.

    Specific implementations may pass in their own kwargs.
    """

    @abstractmethod
    def compute_de_genes(self, adata_latent, **kwargs):
        """
        Given an AnnData whose .X is in latent space, return a data frame of DE genes.
        """
        pass


class UCELogProbDecoder(DecoderInterface):
    """
    This class decodes UCE embeddings into log probabilities of expression in gene space.
    """

    def __init__(
        self,
        model_loc="/large_storage/ctc/ML/data/cell/misc/model_used_in_paper_33l_8ep_1024t_1280.torch",
    ):
        self.model_loc = model_loc

    def compute_de_genes(self, adata_latent, pert_col, control_pert, genes, k=50):
        """
        Compute DE in gene space using UCE predictions, by decoding probability in each gene and
        manipulating the log probs as a statistical test.
        """
        logger.info(f"Computing DE genes using UCE log probs decoder.")
        gene_predictor = UCEGenePredictor(device="cuda:0", model_loc=self.model_loc)
        gene_logprobs = gene_predictor.compute_gene_prob_group_batched(adata_latent.X, genes, batch_size=96)
        probs_df = pd.DataFrame(gene_logprobs)
        probs_df["pert"] = adata_latent.obs[pert_col].values
        mean_df = probs_df.groupby("pert").mean()

        ctrl = mean_df.loc[control_pert].values
        pert_effects = np.abs(mean_df - ctrl)

        top_k_indices = np.argsort(pert_effects.values, axis=1)[:, -k:][:, ::-1]
        top_k_genes = np.array(genes)[top_k_indices]
        de_genes = pd.DataFrame(top_k_genes)
        de_genes.index = pert_effects.index.values
        res = de_genes

        return res
