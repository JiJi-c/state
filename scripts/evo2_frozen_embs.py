'''
This script only create emb for Evo2 and expects https://github.com/Zymrael/vortex to be installed
'''

import os
import logging
import argparse
import pandas as pd

from base64 import decodebytes
from io import BytesIO
from numpy import load


from vci.data.gene_emb import create_genename_sequence_map


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

import torch
import yaml

from dataclasses import dataclass
from functools import lru_cache
from os import getenv

import logging
log = logging.getLogger(__name__)


@lru_cache
def is_fp8_supported():
    from transformer_engine.pytorch.fp8 import check_fp8_support
    log.info(f"{check_fp8_support()=}")
    return check_fp8_support()[0]

@lru_cache
def should_use_cached_generation():
    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
    # So far cached generation is only practical on A100/H100 and above.
    if mem_gb > 60:
        log.info(f"Will use cached generation, {mem_gb=}")
        return True
    gpus = torch.cuda.device_count()
    if gpus >= 2:
        log.info(f"Will use cached generation, {gpus=}")
        return True
    log.info(f"Will not use cached generation, {mem_gb=}")
    return False

@lru_cache
def detect_force_prompt_threshold():
    env = getenv("NIM_EVO2_FORCE_PROMPT_THRESHOLD")
    if env is not None:
        log.info(f"Will use force_prompt_threshold from env variable: {env=}")
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
    log.info(f"Will use force_prompt_threshold={ret}, {gpus=} {mem_gb=}")
    return ret

@lru_cache(maxsize=1)
def get_model(*,
    config_path,
    dry_run,
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
        log.info("fp8 forced off as the support is not present")
        config.use_fp8_input_projections = False

    if config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(config.vocab_file)

    m = StripedHyena(config)

    state_dict = torch.load('/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_1m/iter_12500.pt')#, map_location=device)
    m.custom_load_state_dict(state_dict, strict=False)

    print(f"Number of parameters: {sum(p.numel() for p in m.parameters())}")
    m = m.to("cuda:0")
    return m, tokenizer, "cuda:0"

class LayerHook:
    def __init__(self, *, layer_name, store):
        self.layer_name = layer_name
        self.store = store

    def hook_fn(self, module, input, output):
        self.store[self.layer_name + ".output"] = output.cpu()

def run_forward(
    input_string,
    *,
    layers=["embedding_layer", "unembed", "blocks.0.mlp.l1"],
    config_path="/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_1m/shc-evo2-7b-8k-2T-1m.yml",
    dry_run=True,
    checkpoint_path=None,
):
    m, tokenizer, device = get_model(
        config_path=config_path,
        dry_run=dry_run,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create dataset list CSV file"
    )
    parser.add_argument(
        "--ref_genome",
        type=str,
        default ='/large_storage/ctc/public/dataset/Homo_sapiens.GRCh38.cdna.all.fa.gz',
        help="Reference genome fasta file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default='/large_storage/ctc/ML/data/cell/embs',
        help="Path to save results",
    )

    args = parser.parse_args()
    # Create intermedate genename_sequence mapping file
    mapping_filename = os.path.join(args.output, "genename_sequence_mapping.tsv")
    if not os.path.exists(mapping_filename):
        create_genename_sequence_map(args.ref_genome, mapping_filename)

    gene_seq_map = pd.read_csv(mapping_filename, delimiter='\t')
    mapping_file = os.path.join(args.output,
                                'Homo_sapiens.GRCh38.gene_symbol_to_embedding_Evo2_7B_mean.pt')

    mappings = {}
    if os.path.exists(mapping_file):
        mappings = torch.load(mapping_file)

    for idx, row in gene_seq_map.iterrows():
        gene = row[0]
        if gene in mappings:
            logging.info(f'Emb for {gene} already exists')
            continue

        seq = row[1]
        octs = run_forward(seq)
        emb = octs['embedding_layer.output'].squeeze().mean(0)
        mappings[gene] = emb
        logging.debug(f'Embedding of {gene} is {emb}')

        if idx % 100 == 0:
            logging.info(f'Saving after {idx}')
            torch.save(mappings, mapping_file)

    torch.save(mappings, mapping_file)
