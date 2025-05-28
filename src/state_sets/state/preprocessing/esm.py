import logging
import os
import time

import tenacity
import torch

from .base import BaseEmbedding


class ESM3Embedding(BaseEmbedding):
    def __init__(self, species, name_suffix="", mapping_output_loc=None, seq_generator_fn=None):
        super().__init__(
            species,
            mapping_output_loc=mapping_output_loc,
            seq_generator_fn=seq_generator_fn,
            seq_type="protein",
            name="ESM3",
            name_suffix=name_suffix,
            max_seq_len=2046,
        )

    def generate_gene_emb_mapping(self, output_dir):
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.sdk.forge import ESM3ForgeInferenceClient

        client = ESM3ForgeInferenceClient(
            model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=os.environ.get("ESM_API_TOKEN")
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{self.name}_emb_{self.species}.torch")
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
                    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
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
    def __init__(
        self,
        species,
        name_suffix="",
        esm2_model="facebook/esm2_t36_3B_UR50D",
        mapping_output_loc=None,
        seq_generator_fn=None,
    ):
        super().__init__(
            species,
            mapping_output_loc=mapping_output_loc,
            seq_generator_fn=seq_generator_fn,
            seq_type="protein",
            name="ESM2",
            name_suffix=name_suffix,
        )
        self.esm_model_name = esm2_model

    def generate_gene_emb_mapping(self, output_dir):
        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.esm_model_name)
        model = AutoModel.from_pretrained(self.esm_model_name)
        model = model.to(device)
        model.eval()

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

            self.save_gene_emb_mapping(ctr, self.output_file, output_dir)
            del outputs
            torch.cuda.empty_cache()

        torch.save(self.gene_emb_mapping, self.output_file)
        logging.info("Done Processing")

    def fetch_emb(self, sequence, esm2_model="facebook/esm2_t36_3B_UR50D"):
        from transformers import AutoModel, AutoTokenizer

        if not esm2_model:
            esm_model_name = self.esm_model_name

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        model = AutoModel.from_pretrained(esm_model_name)
        model = model.to(device)
        model.eval()

        # Tokenize the sequence
        inputs = tokenizer(sequence, return_tensors="pt", padding=True).to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.last_hidden_state.mean(1).cpu()[0]
