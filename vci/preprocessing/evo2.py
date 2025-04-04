import os
import gzip
import torch
import yaml
import logging
import numpy as np
import pandas as pd

from argtyped import Arguments
from typing import Optional
from tqdm import tqdm
from Bio.Seq import Seq
from Bio import SeqIO
from functools import lru_cache

from .base import BaseEmbedding


from ..models import load_model
from eval.embeddings import extract_embeddings_batch


SEED = 21
DEFAULT_OUTPUT_PATH = 'out/embeddings/'


class ExtractEmbeddingsArgs(Arguments, underscore=True):
    input_fasta_path: str
    model_name: Optional[str] = 'evo2_7b_1m_gen'
    layer: Optional[str] = 'blocks.26.pre_norm'
    output_path: Optional[str] = DEFAULT_OUTPUT_PATH
    batch_size: Optional[int] = 1
    device: Optional[str] = 'cuda:0'


class ExtractEmbeddings(object):
    def __init__(self, args: ExtractEmbeddingsArgs):
        self.args = args
        np.random.seed(SEED)

        # Read sequences into dataframe
        loci_ids = []
        loci_seqs = []

        # Check file format of input fasta
        if self.args.input_fasta_path.endswith('.gz'):
            with gzip.open(self.args.input_fasta_path, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    loci_ids.append(record.id)
                    loci_seqs.append(str(record.seq))
        elif self.args.input_fasta_path.endswith('.fasta'):
            with open(self.args.input_fasta_path, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    loci_ids.append(record.id)
                    loci_seqs.append(str(record.seq))
        else:
            raise ValueError(f'Input file must be a .fasta or .fasta.gz file, got {self.args.input_fasta_path}')

        # Build dataframe
        self.loci_df = pd.DataFrame({
            'id': loci_ids,
            'seq': loci_seqs
        })

        # Add length column to dataframe
        self.loci_df['length'] = self.loci_df['seq'].apply(len)

        # Make output directory if it doesn't exist
        os.makedirs(self.args.output_path, exist_ok=True)

        # Make subdirectory for forward and reverse complement embeddings
        os.makedirs(os.path.join(self.args.output_path, 'forward'), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_path, 'reverse'), exist_ok=True)

        # Write dataframe to tsv except for seq column
        self.loci_df.drop(columns=['seq']).to_csv(os.path.join(self.args.output_path, 'loci.tsv'), sep='\t', index=True)

        # Load model
        self.model = load_model(self.args.model_name, self.args.device)


    def run(self) -> None:
        logging.info('Extracting embeddings...')
        self._extract_embeddings()

    def _extract_embeddings(self):
        """
        Extract embeddings for each locus.
        """
        for i in tqdm(range(len(self.loci_df))):
            seq = self.loci_df.seq[i]
            embedding_orig = extract_embeddings_batch(
                [seq],
                self.model.model,
                self.model.tokenizer,
                self.args.layer,
                batch_size=self.args.batch_size,
                device=self.args.device
            )[0]

            embedding_rc = extract_embeddings_batch(
                [str(Seq(seq).reverse_complement())],
                self.model.model,
                self.model.tokenizer,
                self.args.layer,
                batch_size=self.args.batch_size,
                device=self.args.device
            )[0]

            np.save(os.path.join(self.args.output_path, 'forward', f'{i}.npy'), embedding_orig)
            np.save(os.path.join(self.args.output_path, 'reverse', f'{i}.npy'), embedding_rc)


# @lru_cache
# def is_fp8_supported():
#     from transformer_engine.pytorch.fp8 import check_fp8_support
#     logging.info(f"{check_fp8_support()=}")
#     return check_fp8_support()[0]


# @lru_cache
# def should_use_cached_generation():
#     mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
#     # So far cached generation is only practical on A100/H100 and above.
#     if mem_gb > 60:
#         logging.info(f"Will use cached generation, {mem_gb=}")
#         return True
#     gpus = torch.cuda.device_count()
#     if gpus >= 2:
#         logging.info(f"Will use cached generation, {gpus=}")
#         return True
#     logging.info(f"Will not use cached generation, {mem_gb=}")
#     return False


# @lru_cache
# def detect_force_prompt_threshold():
#     env = getenv("NIM_EVO2_FORCE_PROMPT_THRESHOLD")
#     if env is not None:
#         logging.info(f"Will use force_prompt_threshold from env variable: {env=}")
#         return int(env)

#     mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
#     gpus = torch.cuda.device_count()
#     if gpus >= 2 and mem_gb > 120: # e.g. h200-x2
#         ret = 8192
#     elif mem_gb > 120: # e.g. h200-x1
#         ret = 4096
#     elif gpus >= 2 and mem_gb > 60: # e.g. h100-x2
#         ret = 512
#     else: # e.g. l40-x2
#         ret = 128
#     logging.info(f"Will use force_prompt_threshold={ret}, {gpus=} {mem_gb=}")
#     return ret


# @lru_cache(maxsize=1)
# def get_model(*,
#     config_path,
#     checkpoint_path,
# ):
#     get_model.cache_clear()
#     import gc
#     gc.collect()

#     from vortex.model.model import StripedHyena
#     from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
#     from vortex.model.utils import dotdict#, load_checkpoint

#     torch.set_printoptions(precision=2, threshold=5)

#     torch.manual_seed(1)
#     torch.cuda.manual_seed(1)

#     config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))

#     if config.use_fp8_input_projections and not is_fp8_supported():
#         logging.info("fp8 forced off as the support is not present")
#         config.use_fp8_input_projections = False

#     if config.tokenizer_type == "CharLevelTokenizer":
#         tokenizer = CharLevelTokenizer(config.vocab_size)
#     else:
#         tokenizer = HFAutoTokenizer(config.vocab_file)

#     m = StripedHyena(config)

#     state_dict = torch.load(checkpoint_path, weights_only=False)#, map_location=device)
#     m.custom_load_state_dict(state_dict, strict=False)

#     print(f"Number of parameters: {sum(p.numel() for p in m.parameters())}")
#     m = m.to("cuda:0")
#     return m, tokenizer, "cuda:0"


# def run_forward(
#     input_string,
#     *,
#     layers=["blocks.26.pre_norm", "embedding_layer", "unembed", "blocks.0.mlp.l1"],
#     config_path="/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_1m/shc-evo2-7b-8k-2T-1m.yml",
#     checkpoint_path='/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_1m/iter_12500.pt'
# ):
#     m, tokenizer, device = get_model(
#         config_path=config_path,
#         checkpoint_path=checkpoint_path,
#     )
#     store = {}
#     hooks = []
#     try:
#         for l in layers:
#             hooks.append(
#                 m.get_submodule(l).register_forward_hook(
#                     LayerHook(layer_name=l, store=store).hook_fn
#                 )
#             )

#         with torch.no_grad():
#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 x = tokenizer.tokenize(input_string)
#                 x = torch.LongTensor(x).unsqueeze(0).to(device)
#                 m(x)
#     finally:
#         for h in hooks:
#             h.remove()

#     return store


# class LayerHook:
#     def __init__(self, *, layer_name, store):
#         self.layer_name = layer_name
#         self.store = store

#     def hook_fn(self, module, input, output):
#         self.store[self.layer_name + ".output"] = output.cpu()


# class Evo2Embedding(BaseEmbedding):

#     def __init__(self,
#                  species,
#                  name_suffix='',
#                  mapping_output_loc=None,
#                  seq_generator_fn=None):
#         super().__init__(species,
#                          mapping_output_loc=mapping_output_loc,
#                          seq_generator_fn=seq_generator_fn,
#                          seq_type='dna',
#                          name='Evo2',
#                          name_suffix=name_suffix)

#     def generate_gene_emb_mapping(self, output_dir):
#         ctr = 0
#         for species, gene, sequences in self.seq_generator_fn():
#             # Ideally dataloader should be already skipping the processed genes
#             # This is here as a precaution to avoid overwriting the embeddings
#             if self.gene_emb_mapping.get(gene) is not None:
#                 logging.info(f"Skipping {species} {gene}  {len(sequences[0])}...")
#                 continue

#             ctr += 1
#             logging.info(f"Processing {species} {gene}  {len(sequences[0])}...")
#             sequences = sequences[0]

#             # if len(sequences) > 70000:
#             #     sequences = sequences[:70000]

#             octs = run_forward(sequences)
#             dna_sequence = Seq(sequences)
#             reverse_complement_seq = str(dna_sequence.reverse_complement())
#             rev_octs = run_forward(reverse_complement_seq)

#             emb = octs['blocks.26.pre_norm.output'].squeeze().mean(0)
#             rev_emb = rev_octs['blocks.26.pre_norm.output'].squeeze().mean(0)

#             self.gene_emb_mapping[gene] = torch.mean(torch.stack([emb, rev_emb]), dim=0)
#             logging.debug(f'Embedding of {gene} is {emb}')

#             self.save_gene_emb_mapping(ctr, self.output_file, output_dir)
#             torch.cuda.empty_cache()

#         logging.info(f'Saving final mapping...')
#         torch.save(self.gene_emb_mapping, self.output_file)

