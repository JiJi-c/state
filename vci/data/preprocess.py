import os
import logging

import torch
import scanpy as sc

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
        self.gene_embs = {gene.lower(): emb for gene, emb in self.gene_embs.items()}
        self.gene_filter = list(self.gene_embs.keys())

        if self.summary_file:
            if not os.path.exists(summary_file):
                with open(summary_file, 'w') as f:
                    f.write('species,path,names,num_cells,num_genes\n')

    def _update_summary(self, adata, species, path, dataset):
        num_cells = adata.X.shape[0]
        num_genes = adata.X.shape[1]

        with open(self.summary_file, 'a') as f:
            f.write(f'{species},{path},{dataset},{num_cells},{num_genes}\n')

    def _process(self, h5ad_file, ):
        log.info(f'Processing {h5ad_file}...')
        species = 'human'

        adata = sc.read(h5ad_file)
        original_shape = adata.shape

        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.filter_cells(adata, min_genes=25)
        basic_flt_shape = adata.shape

        try:
            adata = adata[:, adata.var.feature_name.str.lower().isin(self.gene_filter)]
        except:
            adata.var['feature_name'] = adata.var.index.values
            adata = adata[:, adata.var.feature_name.str.lower().isin(self.gene_filter)]

        emb_flt_shape = adata.shape

        log.info(f'{h5ad_file} Original size: {original_shape}.'
                 f' After basic filter: {basic_flt_shape}.'
                 f' After gene with embedding filter: {emb_flt_shape}.')

        adata.var.feature_name = adata.var.feature_name.str.lower()

        return adata, species

    def update_dataset_emb_idx(self, adata, dataset):
        if 'feature_name' not in adata.var:
            adata.var['feature_name'] = adata.var.index.values
            adata.var.feature_name = adata.var.feature_name.str.lower()
        emb_idxs = torch.tensor([self.gene_filter.index(k) + self.emb_offset \
                                 for k in adata.var.feature_name]).long()

        dataset_emb_idx = {}
        if os.path.exists(self.emb_idx_file):
            dataset_emb_idx = torch.load(self.emb_idx_file)

        if dataset in dataset_emb_idx:
            raise ValueError(f'{dataset} already exists in the emb_idx_file')

        dataset_emb_idx[dataset] = emb_idxs
        torch.save(dataset_emb_idx, self.emb_idx_file)

    def process(self):
        h5ad_files = [f.name for f in Path(self.source).iterdir() if f.is_file()]
        h5ad_files = sorted(h5ad_files)

        for h5ad_file in h5ad_files:
            path = Path(h5ad_file)
            dataset = path.stem
            path = path.name
            adata_path = os.path.join(self.dest, f'{dataset}.h5ad')
            adata_tmp_path = os.path.join(self.dest, f'{dataset}_tmp.h5ad')

            if os.path.exists(adata_path):
                log.info(f'{h5ad_file} already processed...')
                continue

            if os.path.getsize(os.path.join(self.source, h5ad_file)) > (10 * (1024 ** 3)):
                log.warning(f'{h5ad_file} is too big. Skipping for now.')
                continue

            if not os.path.exists(adata_tmp_path):
                try:
                    adata, species = \
                        self._process(os.path.join(self.source, h5ad_file))

                    del adata.uns
                    adata.write(adata_tmp_path)

                    self.update_dataset_emb_idx(adata, dataset)
                    self._update_summary(adata, species, path, dataset)
                    del adata
                except Exception as ex:
                    log.exception(ex)
                    continue

            os.rename(adata_tmp_path, adata_path)