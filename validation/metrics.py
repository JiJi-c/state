import scipy
import pandas as pd
from collections import defaultdict
from validation.metric_utils import (
    to_dense,
    compute_mse,
    compute_pearson_delta,
    compute_pearson_delta_separate_controls,
    compute_wasserstein,
    compute_mmd,
    compute_cosine_similarity,
    compute_cosine_similarity_v2,
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
    adata_real_exp=None,  # gene expression values
    embed_key=None,
    include_dist_metrics=False,
    control_pert="non-targeting",
    pert_col="pert_name",
    celltype_col="celltype_name",
    batch_col="gem_group",
    model_loc=None,
    num_de_genes=50,
    DE_metric_flag=True,
    class_score_flag=True,
    checking_mapping_quality=False,
    transform=None,  # transformation to apply to the data to go from gene expression to non-UCE embedding space
    output_space="gene",
    decoder=None,
    shared_perts=None,
):
    pred_celltype_pert_dict = adata_pred.obs.groupby(celltype_col)[pert_col].agg(set).to_dict()
    real_celltype_pert_dict = adata_real.obs.groupby(celltype_col)[pert_col].agg(set).to_dict()

    # TODO: Test this
    # assert adata_pred.obs[pert_col].values.tolist() == adata_real.obs[pert_col].values.tolist(), "Pred and real adatas do not have identical perturbations"
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
    should_use_exp = (adata_real_exp) and (
        (output_space == "gene") or (output_space == "latent" and decoder is not None)
    )
    metrics = {}
    for celltype in tqdm(pred_celltype_pert_dict, desc="celltypes"):
        with time_it(f"compute_metrics_cell_type_{celltype}"):
            metrics[celltype] = defaultdict(list)

            if checking_mapping_quality:
                metrics[celltype]["true_mean_corr"] = []
                metrics[celltype]["pred_mean_corr"] = []

            adata_pred_control = get_samples_by_pert_and_celltype(
                adata_pred,
                pert=control_pert,
                celltype=celltype,
                pert_col=pert_col,
                celltype_col=celltype_col,
            )

            adata_real_control = get_samples_by_pert_and_celltype(
                adata_real if adata_pred.X.shape == adata_real.X.shape else adata_real_exp,
                pert=control_pert,
                celltype=celltype,
                pert_col=pert_col,
                celltype_col=celltype_col,
            )

            all_perts = shared_perts if shared_perts else pred_celltype_pert_dict[celltype]
            for pert in tqdm(all_perts, desc="perts"):
                if pert == control_pert:
                    continue

                with time_it(f"compute_metrics_pert_{pert}"):
                    adata_pred_pert = get_samples_by_pert_and_celltype(
                        adata_pred,
                        pert=pert,
                        celltype=celltype,
                        pert_col=pert_col,
                        celltype_col=celltype_col,
                    )

                    adata_real_pert = get_samples_by_pert_and_celltype(
                        adata_real if adata_pred.X.shape == adata_real.X.shape else adata_real_exp,
                        pert=pert,
                        celltype=celltype,
                        pert_col=pert_col,
                        celltype_col=celltype_col,
                    )

                    ## Use softmap to generate artificial control distributions
                    pert_idx = adata_pred_pert.obs.index.astype("int").tolist()

                    adata_pred_control.obs.index = pd.Categorical(adata_pred_control.obs.index)
                    adata_real_control.obs.index = pd.Categorical(adata_real_control.obs.index)

                    ## Get the predictions and true values
                    pert_preds = to_dense(adata_pred_pert.X)
                    pert_true = to_dense(adata_real_pert.X)
                    control_true = to_dense(adata_real_control.X)
                    control_preds = to_dense(adata_pred_control.X)
                    pred_batches = adata_real_pert.obs[batch_col].values
                    ctrl_batches = adata_real_control.obs[batch_col].values

                    ## If matrix is sparse convert to dense
                    try:
                        pert_true = pert_true.toarray()
                        control_true = control_true.toarray()
                    except:
                        pass

                    ## Compute metrics at the batch level
                    batched_metrics = _compute_metrics_dict_batched(
                        pert_preds,
                        pert_true,
                        control_true,
                        control_preds,
                        pred_batches,
                        ctrl_batches,
                        include_dist_metrics=include_dist_metrics,
                    )

                    ## Compute metrics at the level of mapped controls for each sample
                    # nn_metrics = _compute_metrics_nearest(pert_preds,
                    #                                     pert_true,
                    #                                     nn_ctrls)

                    ## Compute metrics across all batches for a specific perturbation
                    curr_metrics = _compute_metrics_dict(
                        pert_preds,
                        pert_true,
                        control_true,
                        control_preds,
                        suffix="cell_type",
                        include_dist_metrics=include_dist_metrics,
                    )

                    ## Softmap metrics
                    # softmap_metrics = _compute_metrics_dict(pert_preds,
                    #                                         pert_true,
                    #                                         control_true_softmap,
                    #                                         control_pred_softmap,
                    #                                         suffix="softmap",
                    #                         include_dist_metrics=include_dist_metrics)

                    ## Compute alignment across samples
                    if checking_mapping_quality:
                        mapped_controls = adata_pred_pert.layers["mapped_control"]

                        true_corr_matrix = np.corrcoef(pert_true - mapped_controls)
                        upper_tri = np.triu(true_corr_matrix, k=1)
                        corr_values = upper_tri[upper_tri != 0]
                        true_mean_corr = np.mean(corr_values)
                        metrics[celltype]["true_mean_corr"].append(true_mean_corr)

                        pred_corr_matrix = np.corrcoef(pert_preds - mapped_controls)
                        upper_tri = np.triu(pred_corr_matrix, k=1)
                        corr_values = upper_tri[upper_tri != 0]
                        pred_mean_corr = np.mean(corr_values)
                        metrics[celltype]["pred_mean_corr"].append(pred_mean_corr)

                        ## Also measure the magnitude of predicted and true
                        # perturbation effects
                        true_norm = np.linalg.norm(pert_true - mapped_controls, axis=1)
                        pred_norm = np.linalg.norm(pert_preds - mapped_controls, axis=1)
                        metrics[celltype]["true_effect_norm"] = np.mean(true_norm)
                        metrics[celltype]["pred_effect_norm"] = np.mean(pred_norm)

                    metrics[celltype]["pert"].append(pert)
                    for k, v in curr_metrics.items():
                        metrics[celltype][k].append(v)

                    for k, v in batched_metrics.items():
                        metrics[celltype][k].append(v)

            adata_real_ct = adata_real[adata_real.obs[celltype_col] == celltype]
            adata_pred_ct = adata_pred[adata_pred.obs[celltype_col] == celltype]

            if should_use_exp:
                logger.info(f"Using gene expression data for {celltype}")
                adata_real_exp_ct = adata_real_exp[adata_real_exp.obs[celltype_col] == celltype]
            else:
                adata_real_exp_ct = None

            if adata_real_exp_ct and output_space == "gene":
                adata_pred_ct.var.index = adata_real_exp_ct.var.index

            if DE_metric_flag:
                ## Compute differential expression at the full adata level for speed

                # 2) Actually compute DE for both truth & pred
                # for num_de in [10, 50, 100, 150, 200]:
                for num_de in [50]:
                    logger.info(f"Computing DE for {num_de} genes")
                    DE_true, DE_pred = compute_DE_for_truth_and_pred(
                        adata_real_exp_ct or adata_real_ct,
                        adata_pred_ct,
                        control_pert=control_pert,
                        pert_col=pert_col,
                        n_top_genes=2000,  # default HVG
                        k_de_genes=num_de,
                        output_space=output_space,
                        model_decoder=decoder,
                    )

                    DE_metrics = compute_gene_overlap_cross_pert(DE_true, DE_pred)
                    metrics[celltype][f"DE_{num_de}"] = [DE_metrics[k] for k in DE_metrics]
                    # metrics[celltype]['DE'] = [DE_metrics[k] for k in metrics[celltype]['pert']]

                # Compute the actual top-k gene lists per perturbation
                de_pred_genes_col = []
                de_true_genes_col = []

                for p in metrics[celltype]["pert"]:
                    if p == control_pert:
                        de_pred_genes_col.append("")
                        de_true_genes_col.append("")
                        continue

                    # Retrieve predicted and true DE genes for p, if available
                    if p in DE_pred.index:
                        pred_genes = list(DE_pred.loc[p].values)
                    else:
                        pred_genes = []

                    if p in DE_true.index:
                        true_genes = list(DE_true.loc[p].values)
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
                    adata_real_ct if adata_pred_ct.X.shape == adata_real_ct.X.shape else adata_real_exp_ct,
                    pert_col=pert_col,
                    ctrl_pert=control_pert,
                )
                print(f"Perturbation ranking for {celltype}: {class_score}")
                metrics[celltype]["perturbation_id"] = class_score

    for celltype, stats in metrics.items():
        metrics[celltype] = pd.DataFrame(stats).set_index("pert")

    return metrics


def _compute_metrics_dict(pert_pred, pert_true, ctrl_true, ctrl_pred, suffix="", include_dist_metrics=False):
    metrics = {}
    metrics["mse_" + suffix] = compute_mse(pert_pred, pert_true, ctrl_true, ctrl_pred)
    metrics["pearson_delta_" + suffix] = compute_pearson_delta(pert_pred, pert_true, ctrl_true, ctrl_pred)
    metrics["pearson_delta_sep_ctrls_" + suffix] = compute_pearson_delta_separate_controls(
        pert_pred, pert_true, ctrl_true, ctrl_pred
    )
    metrics["cosine_" + suffix] = compute_cosine_similarity(pert_pred, pert_true, ctrl_true, ctrl_pred)
    metrics["cosine_v2_" + suffix] = compute_cosine_similarity_v2(pert_pred, pert_true, ctrl_true, ctrl_pred)
    if include_dist_metrics:
        with time_it("compute_wasserstein"):
            metrics["wasserstein_" + suffix] = compute_wasserstein(pert_pred, pert_true, ctrl_true, ctrl_pred)
        with time_it("compute_mmd"):
            metrics["mmd_" + suffix] = compute_mmd(pert_pred, pert_true, ctrl_true, ctrl_pred)
    return metrics


def _compute_metrics_dict_batched(
    pert_pred,
    pert_true,
    ctrl_true,
    ctrl_pred,
    pred_batches,
    ctrl_batches,
    include_dist_metrics=False,
):
    batched_means = {}
    # pd.DataFrame(np.append(np.delete(pert_pred, 9193, 1), np.array([pred_batches]).T, axis=1), columns=([str(x) for x in range(0, 9193)] + ['batch']))

    if pert_pred.shape[1] == pert_true.shape[1] + 1:
        pert_pred = np.delete(pert_pred, pert_pred.shape[1] - 1, 1)

    batched_means["pert_pred"] = get_batched_mean(pert_pred, pred_batches)
    batched_means["pert_true"] = get_batched_mean(pert_true, pred_batches)
    batched_means["ctrl_true"] = get_batched_mean(ctrl_true, ctrl_batches)
    batched_means["ctrl_pred"] = get_batched_mean(ctrl_pred, ctrl_batches)
    weightings = {b: sum(np.array(pred_batches) == b) for b in pred_batches}

    metrics = {}
    metrics["pearson_delta_batched_controls"] = compute_pearson_delta_batched(batched_means, weightings)
    return metrics


##TODO implementation
# def _compute_metrics_nearest(pert_pred, pert_true, mapped_controls):


#     res = pearsonr((pert_pred - mapped_controls).mean(0), \
#                             (pert_true - mapped_controls).mean(0))[0]
#     metrics = {}
#     metrics["pearson_delta_nearest"] = res
#     return metrics

# def _compute_cross_pert_metrics_dict(adata_real, adata_pred):
#     metrics = {}
#     metrics['gene_overlap'] = compute_gene_overlap(adata_pred.X, adata_real.X)


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
