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
ds_files = ['3bbb6cf9-72b9-41be-b568-656de6eb18b5.h5ad', '3f32121d-126b-4e8d-9f69-d86502d2a1b1.h5ad', 'be35c935-ee4f-475c-9d3c-97630d59a735.h5ad', 'ca421096-6240-4cee-8c12-d20899b3e005.h5ad', '00099d5e-154f-4a7a-aa8d-fa30c8c0c43c.h5ad', '019c7af2-c827-4454-9970-44d5e39ce068.h5ad', '1252c5fb-945f-42d6-b1a8-8a3bd864384b.h5ad', '218acb0f-9f2f-4f76-b90b-15a4b7c7f629.h5ad', '4b9e0a15-c006-45d9-860f-b8a43ccf7d9d.h5ad', '524d4513-8afd-4120-9b7b-fe31ae10c29b.h5ad', '6a270451-b4d9-43e0-aa89-e33aac1ac74b.h5ad', '83b5e943-a1d5-4164-b3f2-f7a37f01b524.h5ad', '9434b020-de42-43eb-bcc4-542b2be69015.h5ad', '975e13b6-bec1-4eed-b46a-9be1f1357373.h5ad', 'ac2fea99-ce08-4fca-8d03-a19f37bf21a3.h5ad', 'ae4f8ddd-cac9-4172-9681-2175da462f2e.h5ad', 'c20f1b97-0d47-4192-af0a-1e012621f8d9.h5ad', 'cb34cb2d-aee2-4272-86ed-f8e1af870e52.h5ad', 'd5c67a4e-a8d9-456d-a273-fa01adb1b308.h5ad', 'd6dfdef1-406d-4efb-808c-3c5eddbfe0cb.h5ad', 'e0ed3c55-aff6-4bb7-b6ff-98a2d90b890c.h5ad', 'e0f37114-5e98-406e-bbeb-594603360606.h5ad', 'e1f595f6-ba2c-495e-9bee-7056f116b1e4.h5ad', 'e22482ee-19e8-40bc-9f6e-541dc3c82c20.h5ad', 'e25e50f8-ed2a-4bb0-b6b6-1a4f4ea5af37.h5ad', 'e2808a6e-e2ea-41b9-b38c-4a08f1677f02.h5ad', 'e2a3c32d-71e2-4f38-b19c-dfcb8729cf46.h5ad', 'e2b469d4-b5c3-4a35-9d19-ee71ce61cae0.h5ad', 'e40c6272-af77-4a10-9385-62a398884f27.h5ad', 'f512b8b6-369d-4a85-a695-116e0806857f.h5ad']


def add_dataset_de(config_file, start_idx, payload_len=100):
    config_file = Path(config_file)
    config_path = os.path.relpath(Path(config_file).parent, Path(__file__).parent)
    with initialize(version_base=None, config_path=config_path):
        logging.info(f'Loading config {config_file}...')
        cfg = compose(config_name=config_file.name)

        datasets = cfg.dataset.val
        data_root = cfg.dataset.data_dir

        # df = pd.read_csv(datasets)
        # df = df.iloc[start_idx:start_idx + payload_len]
        failed_files = []
        for dataset_path in ds_files:
            dataset_path = os.path.join(data_root, dataset_path)
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
