import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from utils import UCEGenePredictor

# set up logger
import logging

logger = logging.getLogger(__name__)

import sys
import os

from vci_pretrain.vci.inference import Inference

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

class VCICountsDecoder(DecoderInterface):
    def __init__(
        self,
        model_loc="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_mmd_counts_2048_2/step=766000.ckpt",
        config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_mmd_counts_2048_2/tahoe_config.yaml",
        read_depth=10000,
    ):
        self.model_loc = model_loc
        self.config = config
        self.inference = Inference(OmegaConf.load(self.config))
        self.inference.load_model(self.model_loc)
        self.read_depth = read_depth
    
    def compute_de_genes(self, adata_latent, pert_col, control_pert, genes):
        logger.info("Computing DE genes using VCI counts decoder.")
        # initialize an empty matrix to store the decoded counts
        decoded_counts = np.zeros((adata_latent.shape[0], len(genes)))
        # read from key .X since predictions are stored in .X
        batch_size = 64
        start_idx = 0
        for pred_counts in self.inference.decode_from_adata(adata_latent, genes, 'X', read_depth=self.read_depth, batch_size=batch_size):
            # pred_counts is count data over the gene space (hvg or transcriptome)
            # update the entries of decoded_counts with the predictions
            current_batch_size = pred_counts.shape[0]
            decoded_counts[start_idx:start_idx+current_batch_size, :] = pred_counts
            start_idx += current_batch_size
        

        # undo the log transformation
        decoded_counts = np.expm1(decoded_counts)
        # normalize the total to 10000
        decoded_counts = decoded_counts / decoded_counts.sum(axis=1)[:, None] * 10000

        probs_df = pd.DataFrame(decoded_counts)
        probs_df["pert"] = adata_latent.obs[pert_col].values
        mean_df = probs_df.groupby("pert").mean()

        ctrl = mean_df.loc[control_pert].values
        pert_effects = np.abs(mean_df - ctrl)

        sorted_indices = np.argsort(pert_effects.values, axis=1)[:, ::-1]
        sorted_genes = np.array(genes)[sorted_indices]
        de_genes = pd.DataFrame(sorted_genes)
        de_genes.index = pert_effects.index.values
        return de_genes


class UCELogProbDecoder(DecoderInterface):
    """
    This class decodes UCE embeddings into log probabilities of expression in gene space.
    """

    def __init__(
        self,
        model_loc="/large_storage/ctc/ML/data/cell/misc/model_used_in_paper_33l_8ep_1024t_1280.torch",
    ):
        self.model_loc = model_loc

    def compute_de_genes(self, adata_latent, pert_col, control_pert, genes):
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

        sorted_indices = np.argsort(pert_effects.values, axis=1)[:, ::-1]
        sorted_genes = np.array(genes)[sorted_indices]
        de_genes = pd.DataFrame(sorted_genes)
        de_genes.index = pert_effects.index.values
        return de_genes
