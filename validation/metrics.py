import scipy
import pandas as pd
from collections import defaultdict
from validation.metric_utils import (
    to_dense,
    compute_mse,
    compute_pearson_delta,
    compute_pearson_delta_separate_controls,
    compute_cosine_similarity,
    compute_gene_overlap_cross_pert,
    compute_DE_for_truth_and_pred,
    compute_perturbation_ranking_score,
    compute_pearson_delta_batched,
)
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import pearsonr
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils import time_it

# setup logger
import logging

logger = logging.getLogger(__name__)


## TODO: Redefine this to take as input the mapping information
def compute_metrics(
    adata_pred,  # predictions in uce space
    adata_real,  # true values in uce space
    adata_pred_gene=None,  # predictions in gene space
    adata_real_gene=None,  # true values in gene space
    embed_key=None,
    include_dist_metrics=False,
    control_pert="non-targeting",
    pert_col="pert_name",
    celltype_col="celltype_name",
    batch_col="gem_group",
    model_loc=None,
    DE_metric_flag=True,
    class_score_flag=True,
    output_space="gene",
    decoder=None,
    shared_perts=None,
    outdir=None, # output directory to store raw de results
):
    pred_celltype_pert_dict = adata_pred.obs.groupby(celltype_col)[pert_col].agg(set).to_dict()
    real_celltype_pert_dict = adata_real.obs.groupby(celltype_col)[pert_col].agg(set).to_dict()

    assert set(pred_celltype_pert_dict.keys()) == set(real_celltype_pert_dict.keys()), (
        "Pred and real adatas do not have identical celltypes"
    )
    for celltype in pred_celltype_pert_dict.keys():
        assert pred_celltype_pert_dict[celltype] == real_celltype_pert_dict[celltype], (
            f"Pred and real adatas have different set of perturbations for {celltype}"
        )

    # compute metrics
    if type(adata_real.obs.index) != pd.core.indexes.range.RangeIndex:
        adata_real.obs = adata_real.obs.reset_index()
    adata_pred.obs = adata_pred.obs.reset_index()

    metrics = {}
    for celltype in pred_celltype_pert_dict:
        with time_it(f"compute_metrics_cell_type_{celltype}"):
            metrics[celltype] = defaultdict(list)

            adata_pred_control = get_samples_by_pert_and_celltype(
                adata_pred,
                pert=control_pert,
                celltype=celltype,
                pert_col=pert_col,
                celltype_col=celltype_col,
            )

            adata_real_control = get_samples_by_pert_and_celltype(
                adata_real,
                pert=control_pert,
                celltype=celltype,
                pert_col=pert_col,
                celltype_col=celltype_col,
            )

            # only evaluate on perturbations that are shared between the train and test sets
            if shared_perts:
                all_perts = shared_perts & pred_celltype_pert_dict[celltype]
            else:
                all_perts = pred_celltype_pert_dict[celltype]

            pred_groups = adata_pred.obs[adata_pred.obs[celltype_col] == celltype].groupby(pert_col).indices
            real_groups = adata_real.obs[adata_real.obs[celltype_col] == celltype].groupby(pert_col).indices

            # use tqdm to track progress
            for pert in tqdm(all_perts, desc=f"Computing metrics for {celltype}", leave=False):
                try:
                    if pert == control_pert:
                        continue

                    with time_it(f"compute_metrics_pert_{pert}"):
                        pred_idx = pred_groups.get(pert, [])
                        true_idx = real_groups.get(pert, [])
                        if len(pred_idx) == 0 or len(true_idx) == 0:
                            continue

                        # Extract data slices in a vectorized way
                        adata_pred_pert = adata_pred[pred_idx]
                        adata_real_pert = adata_real[true_idx]

                        ## Use softmap to generate artificial control distributions
                        pert_idx = adata_pred_pert.obs.index.astype("int").tolist()

                        adata_pred_control.obs.index = pd.Categorical(adata_pred_control.obs.index)
                        adata_real_control.obs.index = pd.Categorical(adata_real_control.obs.index)

                        ## Get the predictions and true values
                        pert_preds = to_dense(adata_pred_pert.X)
                        pert_true = to_dense(adata_real_pert.X)
                        control_true = to_dense(adata_real_control.X)
                        control_preds = to_dense(adata_pred_control.X)

                        ## If matrix is sparse convert to dense
                        try:
                            pert_true = pert_true.toarray()
                            control_true = control_true.toarray()
                        except:
                            pass

                        ## Compute metrics across all batches for a specific perturbation
                        curr_metrics = _compute_metrics_dict(
                            pert_preds,
                            pert_true,
                            control_true,
                            control_preds,
                            suffix="cell_type",
                            include_dist_metrics=include_dist_metrics,
                        )

                        metrics[celltype]["pert"].append(pert)
                        for k, v in curr_metrics.items():
                            metrics[celltype][k].append(v)

                except:
                    print(f"Failed for {celltype} {pert}")
                    pass

            adata_real_ct = adata_real[adata_real.obs[celltype_col] == celltype]
            adata_pred_ct = adata_pred[adata_pred.obs[celltype_col] == celltype]

            # gene level metrics may not be available if the output_space was specified to be latent
            adata_real_gene_ct = None
            adata_pred_gene_ct = None
            if adata_real_gene is not None:
                logger.info(f"Using gene expression data for true {celltype}")
                adata_real_gene_ct = adata_real_gene[adata_real_gene.obs[celltype_col] == celltype]
            if adata_pred_gene is not None:
                logger.info(f"Using gene expression data for pred {celltype}")
                adata_pred_gene_ct = adata_pred_gene[adata_pred_gene.obs[celltype_col] == celltype]

            if DE_metric_flag:
                ## Compute differential expression at the full adata level for speed

                # 2) Actually compute DE for both truth & pred
                logger.info(f"Computing DE for 50 genes")
                DE_true_fc, DE_pred_fc, DE_true_pval, DE_pred_pval, DE_true_pval_fc, DE_pred_pval_fc = compute_DE_for_truth_and_pred(
                    adata_real_gene_ct or adata_real_ct,
                    adata_pred_gene_ct or adata_pred_ct,
                    control_pert=control_pert,
                    pert_col=pert_col,
                    n_top_genes=2000,  # default HVG
                    k_de_genes=50,
                    output_space=output_space,
                    model_decoder=decoder,
                    outdir=outdir,
                )

                # Compute overlap for fold change-based DE
                DE_metrics_fc = compute_gene_overlap_cross_pert(DE_true_fc, DE_pred_fc, control_pert=control_pert)
                metrics[celltype]['DE_fc'] = [DE_metrics_fc.get(p, 0.0) for p in metrics[celltype]["pert"]]
                metrics[celltype]['DE_fc_avg'] = np.mean(list(DE_metrics_fc.values()))
                
                # Compute overlap for p-value-based DE  
                DE_metrics_pval = compute_gene_overlap_cross_pert(DE_true_pval, DE_pred_pval, control_pert=control_pert)
                metrics[celltype]['DE_pval'] = [DE_metrics_pval.get(p, 0.0) for p in metrics[celltype]["pert"]]
                metrics[celltype]['DE_pval_avg'] = np.mean(list(DE_metrics_pval.values()))

                DE_metrics_pval_fc = compute_gene_overlap_cross_pert(DE_true_pval_fc, DE_pred_pval_fc, control_pert=control_pert)
                metrics[celltype]['DE_pval_fc'] = [DE_metrics_pval_fc.get(p, 0.0) for p in metrics[celltype]["pert"]]
                metrics[celltype]['DE_pval_fc_avg'] = np.mean(list(DE_metrics_pval_fc.values()))

                # Compute the actual top-k gene lists per perturbation
                de_pred_genes_col = []
                de_true_genes_col = []

                for p in metrics[celltype]["pert"]:
                    if p == control_pert:
                        de_pred_genes_col.append("")
                        de_true_genes_col.append("")
                        continue

                    # Retrieve predicted and true DE genes for p, if available
                    if p in DE_pred_pval.index:
                        pred_genes = list(DE_pred_pval.loc[p].values)
                    else:
                        pred_genes = []

                    if p in DE_true_pval.index:
                        true_genes = list(DE_true_pval.loc[p].values)
                    else:
                        true_genes = []

                    # Convert lists to comma-separated strings
                    de_pred_genes_col.append("|".join(pred_genes))
                    de_true_genes_col.append("|".join(true_genes))

                # Store them as new columns
                metrics[celltype]["DE_pred_genes"] = de_pred_genes_col
                metrics[celltype]["DE_true_genes"] = de_true_genes_col

            if class_score_flag:
                ## Compute classification score
                class_score = compute_perturbation_ranking_score(
                    adata_pred_ct,
                    adata_real_ct,
                    pert_col=pert_col,
                    ctrl_pert=control_pert,
                )
                metrics[celltype]["perturbation_id"] = class_score
                metrics[celltype]["perturbation_score"] = 1 - class_score

    try:
        for celltype, stats in metrics.items():
            metrics[celltype] = pd.DataFrame(stats).set_index("pert")
        return metrics
    except Exception as e:
        print(e)
        return metrics

def _compute_metrics_dict(pert_pred, pert_true, ctrl_true, ctrl_pred, suffix="", include_dist_metrics=False):
    metrics = {}
    metrics["mse_" + suffix] = compute_mse(pert_pred, pert_true, ctrl_true, ctrl_pred)
    metrics["pearson_delta_" + suffix] = compute_pearson_delta(pert_pred, pert_true, ctrl_true, ctrl_pred)
    metrics["pearson_delta_sep_ctrls_" + suffix] = compute_pearson_delta_separate_controls(
        pert_pred, pert_true, ctrl_true, ctrl_pred
    )
    metrics["cosine_" + suffix] = compute_cosine_similarity(pert_pred, pert_true, ctrl_true, ctrl_pred)
    return metrics

def get_samples_by_pert_and_celltype(adata, pert, celltype, pert_col, celltype_col):
    pert_idx = (adata.obs[pert_col] == pert).to_numpy()
    celltype_idx = (adata.obs[celltype_col] == celltype).to_numpy()
    out = adata[pert_idx & celltype_idx]
    return out


def get_batched_mean(X, batches):
    if scipy.sparse.issparse(X):
        df = pd.DataFrame(X.todense())
    else:
        df = pd.DataFrame(X)

    df["batch"] = batches
    return df.groupby("batch").mean(numeric_only=True)
