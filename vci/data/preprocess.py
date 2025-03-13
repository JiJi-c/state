import os
import shutil
import logging

import torch
import h5py as h5
import pandas as pd
import scipy.sparse as sp

from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


class Preprocessor:

    def __init__(self, species, source, dest, summary_file,
                 emb_file, emb_idx_file, emb_offset=0) -> None:
        self.species = species
        self.source = source
        self.dest = dest
        self.summary_file = summary_file
        self.emb_file = emb_file
        self.emb_idx_file = emb_idx_file

        self.emb_offset = emb_offset

        # Load gene embedding and convert gene to lower case
        self.gene_embs = torch.load(self.emb_file)
        self.gene_embs = {gene: emb for gene, emb in self.gene_embs.items()}
        self.gene_filter = list(self.gene_embs.keys())

        if self.summary_file:
            if not os.path.exists(summary_file):
                with open(summary_file, 'w') as f:
                    f.write('species,path,names,num_cells,num_genes\n')

    def _convert_to_csr(self, h5f, attrs):

        num_cells = attrs['shape'][0]
        num_genes = attrs['shape'][1]

        data = h5f['X/data'][:]
        indices = h5f['X/indices'][:]
        indptr = h5f['X/indptr'][:]

        csc_matrix = sp.csc_matrix((data, indices, indptr),
                                   shape=(num_cells, num_genes))
        csr_matrix = csc_matrix.tocsr()

        del h5f['X']
        x_group = h5f.create_group('X')
        x_group.attrs['encoding-type'] = 'csr_matrix'
        x_group.attrs['encoding-version'] = '0.1.0'
        x_group.attrs['shape'] = [num_cells, num_genes]

        x_group.create_dataset('data', data=csr_matrix.data, compression="gzip")
        x_group.create_dataset('indices', data=csr_matrix.indices, compression="gzip")
        x_group.create_dataset('indptr', data=csr_matrix.indptr, compression="gzip")

        return num_cells, num_genes

    def _process(self, h5f):
        log.info(f'Processing {h5f.filename}...')

        attrs = dict(h5f['X'].attrs)
        if attrs['encoding-type'] == 'csr_matrix':
            num_cells = attrs['shape'][0]
            num_genes = attrs['shape'][1]
        elif attrs['encoding-type'] == 'array':
            num_cells = h5f['X'].shape[0]
            num_genes = h5f['X'].shape[1]
        elif attrs['encoding-type'] == 'csc_matrix':
            num_cells, num_genes = self._convert_to_csr(h5f, attrs)
        else:
            raise ValueError('Input file contains count mtx in non-csr matrix')

        return num_cells, num_genes

    def update_dataset_emb_idx(self, dataset_file, dataset, feature_field='gene_symbols'):
        dataset_emb_idx = {}
        if os.path.exists(self.emb_idx_file):
            dataset_emb_idx = torch.load(self.emb_idx_file)

        emb_idxs = self._update_dataset_emb_idx(dataset_file, feature_field)

        if dataset in dataset_emb_idx:
            raise ValueError(f'{dataset} already exists in the emb_idx_file')

        dataset_emb_idx[dataset] = emb_idxs
        torch.save(dataset_emb_idx, self.emb_idx_file)

    def _update_dataset_emb_idx(self, dataset_file, feature_field):
        with h5.File(dataset_file, mode='r') as h5f:
            cat_data = pd.Categorical.from_codes(h5f[f'var/{feature_field}/codes'][:],
                                                 categories=h5f[f'var/{feature_field}/categories'][:])
            try:
                idxs = []
                for k in cat_data:
                    k = k.decode('utf-8').lower()
                    if k in self.gene_filter:
                        idx = self.gene_filter.index(k) + self.emb_offset
                    else:
                        idx = -1
                    idxs.append(idx)

                emb_idxs = torch.tensor(idxs).long()
            except ValueError as ex:
                log.error(f'Gene not found in the embedding file: {dataset_file} {ex}')
                return
        return emb_idxs

    def process(self):
        h5ad_files = [f.name for f in Path(self.source).iterdir() if f.is_file()]
        h5ad_files = sorted(h5ad_files)

        os.makedirs(os.path.join(self.dest, self.species), exist_ok=True)

        for h5ad_file in h5ad_files:
            h5ad_file = Path(os.path.join(self.source, h5ad_file))
            if h5ad_file.suffix != '.h5ad':
                continue

            # Copy the file to the destination with temporary name
            dataset = h5ad_file.stem
            dest_h5ad_file = os.path.join(self.dest, self.species, f'{dataset}.h5ad')

            if not os.path.exists(dest_h5ad_file):
                log.info(f'{dest_h5ad_file} already exists')
                shutil.copyfile(str(h5ad_file), dest_h5ad_file)

            with h5.File(dest_h5ad_file, mode='r+') as h5f:
                try:
                    logging.info(f'Processing file {h5ad_file}...')
                    num_cells, num_genes = self._process(h5f)

                    with open(self.summary_file, 'a') as f:
                        f.write(f'{self.species},{dest_h5ad_file},{dataset},{num_cells},{num_genes}\n')

                    # self.update_dataset_emb_idx(adata_path, dataset)
                    # del adata
                except Exception as ex:
                    log.exception(ex)
                    continue