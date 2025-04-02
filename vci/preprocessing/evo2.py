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


@lru_cache
def is_fp8_supported():
    from transformer_engine.pytorch.fp8 import check_fp8_support
    logging.info(f"{check_fp8_support()=}")
    return check_fp8_support()[0]


@lru_cache
def should_use_cached_generation():
    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
    # So far cached generation is only practical on A100/H100 and above.
    if mem_gb > 60:
        logging.info(f"Will use cached generation, {mem_gb=}")
        return True
    gpus = torch.cuda.device_count()
    if gpus >= 2:
        logging.info(f"Will use cached generation, {gpus=}")
        return True
    logging.info(f"Will not use cached generation, {mem_gb=}")
    return False


@lru_cache
def detect_force_prompt_threshold():
    env = getenv("NIM_EVO2_FORCE_PROMPT_THRESHOLD")
    if env is not None:
        logging.info(f"Will use force_prompt_threshold from env variable: {env=}")
        return int(env)

    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
    gpus = torch.cuda.device_count()
    if gpus >= 2 and mem_gb > 120: # e.g. h200-x2
        ret = 8192
    elif mem_gb > 120: # e.g. h200-x1
        ret = 4096
    elif gpus >= 2 and mem_gb > 60: # e.g. h100-x2
        ret = 512
    else: # e.g. l40-x2
        ret = 128
    logging.info(f"Will use force_prompt_threshold={ret}, {gpus=} {mem_gb=}")
    return ret


@lru_cache(maxsize=1)
def get_model(*,
    config_path,
    checkpoint_path,
):
    get_model.cache_clear()
    import gc
    gc.collect()

    from vortex.model.model import StripedHyena
    from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
    from vortex.model.utils import dotdict#, load_checkpoint

    torch.set_printoptions(precision=2, threshold=5)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))

    if config.use_fp8_input_projections and not is_fp8_supported():
        logging.info("fp8 forced off as the support is not present")
        config.use_fp8_input_projections = False

    if config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(config.vocab_file)

    m = StripedHyena(config)

    state_dict = torch.load(checkpoint_path, weights_only=False)#, map_location=device)
    m.custom_load_state_dict(state_dict, strict=False)

    print(f"Number of parameters: {sum(p.numel() for p in m.parameters())}")
    m = m.to("cuda:0")
    return m, tokenizer, "cuda:0"


def run_forward(
    input_string,
    *,
    layers=["blocks.26.pre_norm", "embedding_layer", "unembed", "blocks.0.mlp.l1"],
    config_path="/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_1m/shc-evo2-7b-8k-2T-1m.yml",
    checkpoint_path='/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_1m/iter_12500.pt'
):
    m, tokenizer, device = get_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    store = {}
    hooks = []
    try:
        for l in layers:
            hooks.append(
                m.get_submodule(l).register_forward_hook(
                    LayerHook(layer_name=l, store=store).hook_fn
                )
            )

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x = tokenizer.tokenize(input_string)
                x = torch.LongTensor(x).unsqueeze(0).to(device)
                m(x)
    finally:
        for h in hooks:
            h.remove()

    return store


class LayerHook:
    def __init__(self, *, layer_name, store):
        self.layer_name = layer_name
        self.store = store

    def hook_fn(self, module, input, output):
        self.store[self.layer_name + ".output"] = output.cpu()


class Evo2Embedding(object):

    def __init__(self,
                 species,
                 geneome_loc = '/large_storage/ctc/projects/vci/ref_genome',
                 seq_generator_fn=None):
        self.geneome_loc = geneome_loc
        self.species = species

        self.seq_type = 'dna'
        self.name = 'Evo2'

        self.gene_emb_mapping = {}
        if seq_generator_fn is None:
            self.seq_generator_fn = self._generate_gene_emb_mapping
        else:
            self.seq_generator_fn = seq_generator_fn

    def _generate_gene_emb_mapping(self):
        ref_genome_file = Path(os.path.join(self.geneome_loc, self.ref_genome))
        gene_seq_mapping, _ = parse_genome_for_gene_seq_map(self.species,
                                                         ref_genome_file,
                                                         return_type=self.seq_type)
        for gene, (chroms, sequences) in gene_seq_mapping.items():
            if '.' in gene:
                gene = gene.split('.')[0]
            if gene in self.gene_emb_mapping:
                logging.info(f"Skipping {gene}...")
                continue
            yield self.species, gene, sequences

    def generate_gene_emb_mapping(self,
                                  output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{self.name}_emb_{self.species.lower()}.torch')
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
            logging.info(f"Processing {species} {gene}  {len(sequences[0])}...")
            sequences = sequences[0]

            # if len(sequences) > 70000:
            #     sequences = sequences[:70000]

            octs = run_forward(sequences)
            dna_sequence = Seq(sequences)
            reverse_complement_seq = str(dna_sequence.reverse_complement())
            rev_octs = run_forward(reverse_complement_seq)

            emb = octs['blocks.26.pre_norm.output'].squeeze().mean(0)
            rev_emb = rev_octs['blocks.26.pre_norm.output'].squeeze().mean(0)

            self.gene_emb_mapping[gene] = torch.mean(torch.stack([emb, rev_emb]), dim=0)
            logging.debug(f'Embedding of {gene} is {emb}')

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
            torch.cuda.empty_cache()

        logging.info(f'Saving final mapping...')
        torch.save(self.gene_emb_mapping, output_file)
