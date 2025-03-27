import os
import ast
import logging
import torch
import shutil
import time
import pandas as pd
import tenacity

from Bio.Seq import Seq

class BaseEmbedding(object):
    def __init__(self,
                 species,
                 mapping_output_loc=None,
                 seq_generator_fn=None,
                 seq_type=None,
                 name=None,
                 max_seq_len=8280):
        self.mapping_output_loc = mapping_output_loc
        self.species = species

        self.seq_type = seq_type
        self.name = name
        self.max_seq_len = max_seq_len

        self.gene_emb_mapping = {}
        output_file = os.path.join(mapping_output_loc,
                                   f'{self.name}_ensemble',
                                   f'{self.name}_emb_{self.species.lower()}.torch')
        if os.path.exists(output_file):
            self.gene_emb_mapping = torch.load(output_file)

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


class ESM3Embedding(BaseEmbedding):

    def __init__(self,
                 species,
                 mapping_output_loc=None,
                 seq_generator_fn=None):
        super().__init__(species,
                         mapping_output_loc=mapping_output_loc,
                         seq_generator_fn=seq_generator_fn,
                         seq_type='protein',
                         name='ESM3',
                         max_seq_len=2046)

    def generate_gene_emb_mapping(self, output_dir):
        from esm.sdk.forge import ESM3ForgeInferenceClient
        from esm.sdk.api import ESMProtein, LogitsConfig

        client =  ESM3ForgeInferenceClient(model="esmc-6b-2024-12",
                                           url="https://forge.evolutionaryscale.ai",
                                           token=os.environ.get("ESM_API_TOKEN"))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{self.name}_emb_{self.species}.torch')
        if os.path.exists(output_file):
            logging.info(f"Loading existing embeddings for {output_file}...")
            self.gene_emb_mapping = torch.load(output_file)

        ctr = 0
        for species, gene, sequences in self.seq_generator_fn():
            # Ideally dataloader should be already skipping the processed genes
            # This is here as a precaution to avoid overwriting the embeddings
            if self.gene_emb_mapping.get(gene) is not None:
                logging.info(f"Skipping {species} {gene}  {len(sequences[0])}...")
                continue

            ctr += 1
            if isinstance(sequences, list):
                sequences = sequences[0]

            # Tokenize the sequence
            while True:
                try:
                    protein = ESMProtein(sequence=sequences)
                    protein_tensor = client.encode(protein)
                    logits_output = client.logits(protein_tensor,
                                                LogitsConfig(sequence=True, return_embeddings=True))
                    self.gene_emb_mapping[gene] = logits_output.embeddings.mean(1).cpu()[0]

                    self.save_gene_emb_mapping(ctr, output_file, output_dir)
                    torch.cuda.empty_cache()
                    break
                except tenacity.RetryError as e:
                    logging.error(f"Error processing {gene}: {e}")
                    time.sleep(60)
                except Exception as e:
                    logging.exception(f"Exception {gene}: {e}")
                    break

        torch.save(self.gene_emb_mapping, output_file)
        logging.info("Done Processing")


class ESM2Embedding(BaseEmbedding):

    def __init__(self,
                 species,
                 mapping_output_loc=None,
                 seq_generator_fn=None):
        super().__init__(species,
                         mapping_output_loc=mapping_output_loc,
                         seq_generator_fn=seq_generator_fn,
                         seq_type='protein',
                         name='ESM2')

    def generate_gene_emb_mapping(self, output_dir):
        from transformers import AutoTokenizer, AutoModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "facebook/esm2_t48_15B_UR50D"
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
            # Ideally dataloader should be already skipping the processed genes
            # This is here as a precaution to avoid overwriting the embeddings
            if self.gene_emb_mapping.get(gene) is not None:
                logging.info(f"Skipping {species} {gene}  {len(sequences[0])}...")
                continue

            ctr += 1
            # Tokenize the sequence
            inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)

            self.gene_emb_mapping[gene] = outputs.last_hidden_state.mean(1).cpu()[0]

            self.save_gene_emb_mapping(ctr, output_file, output_dir)
            del outputs
            torch.cuda.empty_cache()

        torch.save(self.gene_emb_mapping, output_file)
        logging.info("Done Processing")
