import os
import logging
import time
import torch
import shutil

from Bio.Seq import Seq

import torch
import yaml

from functools import lru_cache
from os import getenv
from pathlib import Path

from vci.data.gene_emb import parse_genome_for_gene_seq_map
from .base import BaseEmbedding
import torch
from evo2 import Evo2


class Evo2Embedding(BaseEmbedding):

    def __init__(self,
                 species,
                 name_suffix='',
                 mapping_output_loc=None,
                 seq_generator_fn=None,
                 layer_name = 'blocks.26.pre_norm'):
        super().__init__(species,
                         mapping_output_loc=mapping_output_loc,
                         seq_generator_fn=seq_generator_fn,
                         seq_type='dna',
                         name='Evo2',
                         name_suffix=name_suffix)
        self.evo2_model = Evo2('evo2_7b')
        # self.layer_name = 'blocks.28.mlp.l3'
        self.layer_name = layer_name


    def generate_gene_emb_mapping(self, output_dir):
        ctr = 0
        for species, gene, sequences in self.seq_generator_fn():
            # Ideally dataloader should be already skipping the processed genes
            # This is here as a precaution to avoid overwriting the embeddings
            if self.gene_emb_mapping.get(gene) is not None:
                logging.info(f"Skipping {species} {gene}  {len(sequences[0])}...")
                continue

            ctr += 1
            logging.info(f"Processing {species} {gene}  {len(sequences[0])}...")
            sequence = sequences[0]
            # if len(sequences) > 70000:
            #     sequences = sequences[:70000]
            fwd_input_ids = torch.tensor(
                self.evo2_model.tokenizer.tokenize(sequence),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')


            dna_sequence = Seq(sequence)
            reverse_complement_seq = str(dna_sequence.reverse_complement())
            rev_input_ids = torch.tensor(
                self.evo2_model.tokenizer.tokenize(reverse_complement_seq),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')

            _, fwd_emb = self.evo2_model(fwd_input_ids, return_embeddings=True, layer_names=[self.layer_name])
            _, rev_emb = self.evo2_model(rev_input_ids, return_embeddings=True, layer_names=[self.layer_name])

            fwd_emb = fwd_emb[self.layer_name].squeeze()
            rev_emb = rev_emb[self.layer_name].squeeze()

            self.gene_emb_mapping[gene] = torch.mean(torch.stack([fwd_emb, rev_emb]), dim=0)
            logging.debug(f'Embedding of {gene} is {fwd_emb}')

            self.save_gene_emb_mapping(ctr, self.output_file, output_dir)

            if ctr % 100 == 0:
                torch.cuda.empty_cache()

        logging.info(f'Saving final mapping...')
        torch.save(self.gene_emb_mapping, self.output_file)
