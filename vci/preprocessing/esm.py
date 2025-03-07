import os
import logging
import torch
import shutil

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from vci.data.gene_emb import parse_genome_for_gene_seq_map


class ESMEmbedding(object):

    def __init__(self,
                 ref_genome,
                 geneome_loc = '/large_storage/ctc/projects/vci/ref_genome'):
        self.geneome_loc = geneome_loc
        self.ref_genome = ref_genome
        self.species = ref_genome.split('.')[0].lower()
        self.seq_type = 'protein'
        self.gene_emb_mapping = {}
        self.name = 'ESM2'

    def _generate_gene_emb_mapping(self,
                                   ref_genome,
                                   max_seq_len=16559):
        ref_genome_file = Path(os.path.join(self.geneome_loc, ref_genome))
        species = ref_genome.split('.')[0].lower()
        gene_seq_mapping = parse_genome_for_gene_seq_map(species, ref_genome_file, return_type=self.seq_type)

        for gene, (chroms, sequences) in gene_seq_mapping.items():
            if gene in self.gene_emb_mapping:
                logging.info(f"Skipping {gene}...")
                continue

            seq_len = sum([len(s) for s in sequences])
            while seq_len > max_seq_len:
                logging.info(f"Too large sequence {gene} {seq_len} Len: {len(sequences)}")
                sequences = sequences[:len(sequences) - 1]
                seq_len = sum([len(s) for s in sequences])
                if len(sequences) == 1:
                    sequences[0] = sequences[0][:16559]
                    seq_len = sum([len(s) for s in sequences])
            yield species, gene, sequences

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
        for species, gene, sequences in self._generate_gene_emb_mapping(self.ref_genome):
            ctr += 1
            logging.info(f"Processing {species} {gene}...")

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
                checkpoint_file = output_file.replace('.torch', f'.{ctr}.torch')
                shutil.copyfile(output_file, checkpoint_file)

            del outputs
            torch.cuda.empty_cache()

        torch.save(self.gene_emb_mapping, output_file)
