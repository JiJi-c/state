import os
import logging
import argparse

import pandas as pd
import scanpy as sc
from pathlib import Path
from hydra import compose, initialize

from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


def main(cfg):
    datasets = cfg.dataset.train
    data_root = cfg.dataset.data_dir

    df = pd.read_csv(datasets)
    #TODO: Parallelize this
    for row in df.iterrows():
        dataset_path = os.path.join(data_root, row[1][1])
        logging.info(f'Processing file {dataset_path}...')

        adata = None
        try:
            adata = sc.read_h5ad(dataset_path)

            # if 'ranked_genes' in adata.uns and 'leiden' in adata.obs:
            #     logging.info('Dataset already clustered and DE computed. Skipping...')
            #     continue

            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=min(5000, adata.shape[1]))

            adata = adata[:, adata.var.highly_variable]
            sc.pp.scale(adata)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata)
            leiden = adata.obs['leiden']
            sc.tl.rank_genes_groups(adata,
                                    'leiden',
                                    method='wilcoxon',
                                    n_genes=adata.shape[1],
                                    use_raw=False)

            df_gene = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
            df_score = pd.DataFrame(adata.uns['rank_genes_groups']['scores'])

            df_gene_idx = pd.DataFrame()
            for cluster in df_gene.columns:
                df_gene_idx[cluster] = adata.var.index.get_indexer(df_gene[cluster])

            ranked_genes = {
                'gene_names': df_gene,
                'gene_scores': df_score,
                'gene_indices': df_gene_idx
            }

        except Exception as e:
            logging.exception(f'Error reading file {dataset_path}: {e}')
            continue
        finally:
            if adata is not None:
                adata.file.close()
                del adata

        adata_write = sc.read(dataset_path)
        adata_write.uns['ranked_genes'] = ranked_genes
        adata_write.obs['leiden'] = leiden
        adata_write.write_h5ad(dataset_path)
        adata_write.file.close()
        del adata_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster individial dataset and save DE"
    )
    parser.add_argument(
        '-c', "--config",
        type=str,
        help="Training configuration file.",
    )
    args = parser.parse_args()
    config_file = Path(args.config)

    config_path = os.path.relpath(Path(config_file).parent, Path(__file__).parent)
    with initialize(version_base=None, config_path=config_path):
        logging.info(f'Loading config {config_file}...')
        cfg = compose(config_name=config_file.name)
        main(cfg)
