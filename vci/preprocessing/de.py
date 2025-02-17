import os
import logging
import fire

import pandas as pd
import scanpy as sc
from pathlib import Path
from hydra import compose, initialize


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

N_TOP_GENES = 5000
# ds_files = [f'/large_storage/ctc/userspace/alishba.imran/tahoe/plate{i}.h5' for i in range(1, 15)]
ds_files = ['/large_storage/ctc/userspace/alishba.imran/tahoe/plate1.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate2.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate3.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate4.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate5.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate6.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate7.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate8.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate9.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate10.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate11.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate13.h5',
    '/large_storage/ctc/userspace/alishba.imran/tahoe/plate14.h5'
]


def add_dataset_de(config_file, start_idx, payload_len=100):
    config_file = Path(config_file)
    config_path = os.path.relpath(Path(config_file).parent, Path(__file__).parent)
    with initialize(version_base=None, config_path=config_path):
        logging.info(f'Loading config {config_file}...')
        cfg = compose(config_name=config_file.name)

        datasets = cfg.dataset.val
        data_root = cfg.dataset.data_dir

        df = pd.read_csv(datasets)
        df = df.iloc[start_idx:start_idx + payload_len]
        failed_files = []
        # ds_files = ds_files[start_idx:start_idx+payload_len]
        for dataset_path in ds_files:
            # dataset_path = os.path.join(data_root, dataset_path)
            logging.info(f'Processing file {dataset_path}...')

            adata = None
            try:
                adata = sc.read_h5ad(dataset_path)

                if 'ranked_genes' in adata.uns and 'leiden' in adata.obs:
                    logging.info('Dataset already clustered and DE computed. Skipping...')
                    continue

                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=min(N_TOP_GENES, adata.shape[1]))

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
                failed_files.append(dataset_path)
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

        if len(failed_files) > 0:
            logging.error(f'Failed to process {len(failed_files)} files: {failed_files}')


def validate_de(config_file):
    config_file = Path(config_file)
    config_path = os.path.relpath(Path(config_file).parent, Path(__file__).parent)
    with initialize(version_base=None, config_path=config_path):
        logging.info(f'Loading config {config_file}...')
        cfg = compose(config_name=config_file.name)

        ds_paths = [cfg.dataset.val, cfg.dataset.train]
        file_loc = cfg.dataset.data_dir

        issue_files = []
        for ds_path in ds_paths:
            logging.info(f'Validating dataset {ds_path}...')
            df = pd.read_csv(ds_path)
            datasets = df['path']

            for dataset in datasets:
                try:
                    h5ad_path = os.path.join(file_loc, dataset)
                    # TODO: Use h5py instead to make it faster
                    adata = sc.read_h5ad(h5ad_path)
                    if 'ranked_genes' in adata.uns and 'leiden' in adata.obs:
                        logging.info('Dataset already clustered and DE computed. Skipping...')
                    else:
                        issue_files.append(dataset)
                except Exception as e:
                    logging.exception(f'Error reading file {dataset}: {e}')
                    issue_files.append(dataset)

        if len(issue_files) > 0:
            logging.error(f'Found {len(issue_files)} files with missing DE. Files: {issue_files}')
        else:
            logging.info('All files are validated.')


if __name__ == '__main__':
    fire.Fire()
