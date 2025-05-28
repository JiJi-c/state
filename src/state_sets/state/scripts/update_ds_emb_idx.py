#!/bin/env python3

import os
import logging
import argparse
import pandas as pd
import torch
import anndata

from pathlib import Path
from vci.data.preprocess import Preprocessor
from vci.data.gene_emb import convert_symbols_to_ensembl
'''
Creates a CSV file with following columns.
    #,path,species,num_cells,num_genes,names
'''


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Update gene embeddings index for a given h5ad file."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name to be added to the dataset gene embedding mapping file",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="H5ad file for which the gene embedding indexes has to be added to the mapping file",
    )
    parser.add_argument(
        "--emb_idx_file",
        type=str,
        default='/large_storage/ctc/projects/vci/scbasecamp/ESM2_3B/dataset_emb_idx.torch',
        help="Path to save the output summary file",
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        default='/large_storage/ctc/projects/vci/scbasecamp/ESM2_3B/all_species.torch',
        help="Path to save the output summary file",
    )

    args = parser.parse_args()
    preprocess = Preprocessor(None, None, None, None,
                              args.embedding_file,
                              args.emb_idx_file)

    filetype = Path(args.dataset_file).suffix
    if filetype == '.h5ad' or filetype == '.h5':
        adata = anndata.read_h5ad(args.dataset_file)
        adata = convert_symbols_to_ensembl(adata)
        adata.write(args.dataset_file)
        preprocess.update_dataset_emb_idx(args.dataset_file, args.dataset_name)
    elif filetype == '.csv':
        df = pd.read_csv(args.dataset_file)
        dataset_emb_idx = {}
        if os.path.exists(args.emb_idx_file):
            dataset_emb_idx = torch.load(args.emb_idx_file)

        for i, rec in df.iterrows():
            if rec['names'] in dataset_emb_idx:
                log.info(f"Skipping {rec['names']}: {rec['path']} datasets")
                continue

            log.info(f"Processed {rec['names']}: {rec['path']} datasets")
            emb_idxs = preprocess._update_dataset_emb_idx(rec['path'], 'gene_symbols')

            dataset_emb_idx[rec['names']] = emb_idxs
            if i % 100 == 0:
                log.info(f'Saving {args.emb_idx_file}...')
                torch.save(dataset_emb_idx, args.emb_idx_file)
