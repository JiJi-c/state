import logging
import os
from pathlib import Path

import fire
import pandas as pd
import scanpy as sc
from hydra import compose, initialize
from vci.utils import get_dataset_cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
)


def _compute_de(adata, dataset_path, top_genes=5000, group_by="leiden", reference=None):
    logging.info("Clustering...")
    if group_by == "leiden":
        # TODO: Check if dataset alrady has leiden cluster info
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(top_genes, adata.shape[1]))

        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata)
        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)

    logging.info("Ranking...")
    sc.tl.rank_genes_groups(
        adata,
        group_by,
        method="wilcoxon",
        reference=reference if reference != None else "rest",
        n_genes=adata.shape[1],
        use_raw=False,
    )

    df_gene = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
    df_score = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"])

    df_gene_idx = pd.DataFrame()
    for cluster in df_gene.columns:
        df_gene_idx[cluster] = adata.var.index.get_indexer(df_gene[cluster])

    ranked_genes = {"gene_names": df_gene, "gene_scores": df_score, "gene_indices": df_gene_idx}
    adata.uns["ranked_genes"] = ranked_genes

    if reference != None:
        logging.info("Filtering reference gene...")
        adata = adata[adata.obs[group_by] != reference]

    logging.info("Persisting DE data...")
    adata.write_h5ad(dataset_path)
    adata.file.close()


def process(
    dataset: str | list[str],
    start_idx: int = 0,
    payload_len: int = 100,
    top_genes: int = 5000,
    reprocess=False,
    perturbed=False,
    perturb_group="gene",
    reference="non-targeting",
):
    """
    Process the dataset by computing DE genes and clustering.

    Args:
    dataset: str
        Path to the dataset CSV file or a list of files to process.
    start_idx: int
        Starting index of the dataset to process.
    payload_len: int
        Number of datasets to process.
    top_genes: int
        Number of top genes to use for clustering.
    reprocess: bool
        Reprocess the dataset even if it has been processed before.

    Example:
        To process all dataset files in a CSV file:
            python de.py process --dataset /path/to/dataset.csv --start_idx 0 --payload_len 100

        To process a list of dataset files:
            python3 vci/preprocessing/de.py process --dataset="['./cellxgene/jurkat.h5ad', './cellxgene/rpe1.h5ad']"

        To process perturbed dataset:
            python3 vci/preprocessing/de.py process --dataset="['./cellxgene/jurkat.h5ad', './cellxgene/rpe1.h5ad'] --perturbed=True"

        To process perturbed dataset with custom obs column:
            python3 vci/preprocessing/de.py process --dataset="['./cellxgene/jurkat.h5ad', './cellxgene/rpe1.h5ad'] --perturbed=True" --perturb_group="drug"
    """
    if isinstance(dataset, str):
        if os.path.splitext(dataset)[1] != ".csv":
            raise ValueError("Dataset must be a CSV file. For processing individual adatas, use a list of files.")
        df = pd.read_csv(dataset)
        df = df.iloc[start_idx : start_idx + payload_len]
        ds_files = df["path"].to_list()
        ds_files = ds_files[start_idx : start_idx + payload_len]
    else:
        ds_files = dataset

    failed_files = []
    for dataset_path in ds_files:
        dataset_path = dataset_path.strip()
        logging.info(f"Processing file {dataset_path}...")
        adata = sc.read_h5ad(dataset_path)
        if not reprocess and "ranked_genes" in adata.uns and "leiden" in adata.obs:
            logging.info("Dataset already clustered and DE computed. Skipping...")
            continue
        try:
            _compute_de(adata, dataset_path, top_genes=top_genes)
        except Exception as e:
            failed_files.append(dataset_path)
            logging.exception(f"Error reading file {dataset_path}: {e}")
            continue
        finally:
            if adata is not None:
                adata.file.close()
                del adata

        if perturbed:
            logging.info("Processing perturbed dataset...")
            adata = sc.read_h5ad(dataset_path)
            try:
                _compute_de(
                    adata,
                    dataset_path.replace(".h5ad", "_perturbed.h5ad"),
                    top_genes=top_genes,
                    group_by=perturb_group,
                    reference=reference,
                )
            except Exception as e:
                failed_files.append(dataset_path)
                logging.exception(f"Error reading file {dataset_path}: {e}")
                continue
            finally:
                if adata is not None:
                    adata.file.close()
                    del adata

    if len(failed_files) > 0:
        failed_files = set(failed_files)
        logging.error(f"Failed to process {len(failed_files)} files: {failed_files}")


def validate(config_file):
    config_file = Path(config_file)
    config_path = os.path.relpath(Path(config_file).parent, Path(__file__).parent)
    with initialize(version_base=None, config_path=config_path):
        logging.info(f"Loading config {config_file}...")
        cfg = compose(config_name=config_file.name)

        ds_paths = [get_dataset_cfg(cfg).val, get_dataset_cfg(cfg).train]
        file_loc = get_dataset_cfg(cfg).data_dir

        issue_files = []
        for ds_path in ds_paths:
            logging.info(f"Validating dataset {ds_path}...")
            df = pd.read_csv(ds_path)
            datasets = df["path"]

            for dataset in datasets:
                try:
                    h5ad_path = os.path.join(file_loc, dataset)
                    # TODO: Use h5py instead to make it faster
                    adata = sc.read_h5ad(h5ad_path)
                    if "ranked_genes" in adata.uns and "leiden" in adata.obs:
                        logging.info("Dataset already clustered and DE computed. Skipping...")
                    else:
                        issue_files.append(dataset)
                except Exception as e:
                    logging.exception(f"Error reading file {dataset}: {e}")
                    issue_files.append(dataset)

        if len(issue_files) > 0:
            logging.error(f"Found {len(issue_files)} files with missing DE. Files: {issue_files}")
        else:
            logging.info("All files are validated.")


if __name__ == "__main__":
    fire.Fire()
