import os
import ast
import logging
import torch
import shutil
import time
import pandas as pd

from Bio.Seq import Seq


class BaseEmbedding(object):
    def __init__(self,
                 species,
                 mapping_output_loc=None,
                 seq_generator_fn=None,
                 seq_type=None,
                 name=None,
                 name_suffix='',
                 max_seq_len=8280):
        self.mapping_output_loc = mapping_output_loc
        self.species = species

        self.seq_type = seq_type
        self.name = name
        self.max_seq_len = max_seq_len

        self.gene_emb_mapping = {}

        output_file = os.path.join(mapping_output_loc,
                                f'{self.name}_ensemble{name_suffix}',
                                f'{self.name}_emb_{self.species.lower()}.torch')
        if os.path.exists(output_file):
            logging.info(f'Loading {output_file}...')
            self.gene_emb_mapping = torch.load(output_file)
        else:
            logging.info(f'Creating mapping file at {output_file}...')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.output_file = output_file

        if seq_generator_fn is None:
            self.seq_generator_fn = self._generate_gene_emb_mapping
        else:
            self.seq_generator_fn = seq_generator_fn

    def _generate_gene_emb_mapping(self):
        gene_list_dir = os.path.join(self.mapping_output_loc, 'gene_lists')
        gene_seq_file = os.path.join(gene_list_dir, f'{self.species}-gene_seq.csv')

        df = pd.read_csv(gene_seq_file)
        gene_seq_map = df.set_index('ensemble_id')['sequence'].to_dict()

        for gene, sequences in gene_seq_map.items():
            if '.' in gene:
                gene = gene.split('.')[0]
            if not sequences:
                continue

            if gene in self.gene_emb_mapping:
                logging.info(f"Skipping {gene}...")
                continue

            if sequences.startswith('['):
                sequences = ast.literal_eval(sequences)[0]

            if self.seq_type == 'protein':
                sequences = str(Seq(sequences).translate())

            logging.info(f"Processing {self.species} {gene} {len(sequences)}...")
            sequences = sequences[:self.max_seq_len]
            yield self.species, gene, sequences

    def save_gene_emb_mapping(self, ctr, output_file, output_dir):
        if ctr % (100) == 0:
            logging.info(f'Saving after {ctr} batches...')
            torch.save(self.gene_emb_mapping, output_file)

        if ctr % (1000) == 0:
            logging.info(f'creating checkpoint {ctr}...')
            chk_dir = os.path.join(output_dir, 'chk')
            if chk_dir:
                os.makedirs(chk_dir, exist_ok=True)
            checkpoint_file = os.path.join(chk_dir,
                                           f'{self.name}_emb_{self.species}.{time.time()}.torch')
            shutil.copyfile(output_file, checkpoint_file)