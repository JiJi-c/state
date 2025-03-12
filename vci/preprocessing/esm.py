import os
import logging
import torch
import shutil
import time

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from vci.data.gene_emb import parse_genome_for_gene_seq_map


class ESMEmbedding(object):

    def __init__(self,
                 ref_genome,
                 geneome_loc = '/large_storage/ctc/projects/vci/ref_genome',
                 seq_generator_fn=None,
                 species=None):
        self.geneome_loc = geneome_loc
        if ref_genome is not None:
            self.ref_genome = ref_genome
            self.species = ref_genome.split('.')[0].lower()
        else:
            self.species = species

        self.seq_type = 'protein'
        self.gene_emb_mapping = {}
        self.name = 'ESM2'

        if seq_generator_fn is None:
            self.seq_generator_fn = self._generate_gene_emb_mapping
        else:
            self.seq_generator_fn = seq_generator_fn

    def _generate_gene_emb_mapping(self, max_seq_len=8280):
        ref_genome_file = Path(os.path.join(self.geneome_loc, self.ref_genome))
        gene_seq_mapping, _ = parse_genome_for_gene_seq_map(self.species, ref_genome_file, return_type=self.seq_type)

        for gene, (chroms, sequences) in gene_seq_mapping.items():
            if '.' in gene:
                gene = gene.split('.')[0]

            if gene in self.gene_emb_mapping:
                logging.info(f"Skipping {gene}...")
                continue

            seq_len = sum([len(s) for s in sequences])
            while seq_len > max_seq_len:
                logging.info(f"Too large sequence {gene} Len: {seq_len}")
                if len(sequences) > 1:
                    sequences = sequences[:len(sequences) - 1]
                    seq_len = sum([len(s) for s in sequences])
                if len(sequences) == 1:
                    sequences[0] = sequences[0][:max_seq_len]
                    seq_len = sum([len(s) for s in sequences])
            logging.info(f"Processing {self.species} {gene} {seq_len}...")
            yield self.species, gene, sequences

    def generate_gene_emb_mapping(self, output_dir):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "facebook/esm2_t33_650M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{self.name}_emb_{self.species}.torch')
        if os.path.exists(output_file):
            self.gene_emb_mapping = torch.load(output_file)

        ctr = 0
        for species, gene, sequences in self.seq_generator_fn():
            ctr += 1
            # Tokenize the sequence
            inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)

            self.gene_emb_mapping[gene] = outputs.last_hidden_state.mean(1).mean(0).cpu()

            if ctr % (100) == 0:
                logging.info(f'Saving after {ctr} batches...')
                torch.save(self.gene_emb_mapping, output_file)

            if ctr % (1000) == 0:
                logging.info(f'creating checkpoint {ctr}...')
                chk_dir = os.path.join(output_dir, 'chk')
                if chk_dir:
                    os.makedirs(os.path.join(output_dir, 'chk'), exist_ok=True)
                checkpoint_file = os.path.join(chk_dir, f'{self.name}_emb_{self.species}.{time.time()}.torch')
                shutil.copyfile(output_file, checkpoint_file)

            del outputs
            torch.cuda.empty_cache()

        torch.save(self.gene_emb_mapping, output_file)
