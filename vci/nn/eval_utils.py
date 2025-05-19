import scanpy as sc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vci.nn.model import LitUCEModel
from vci.data import create_dataloader
from vci.eval.emb import cluster_embedding
from vci.utils import compute_pearson_delta, compute_perturbation_ranking_score, compute_gene_overlap_cross_pert


def evaluate_perturbation(model, cfg, device=None, logger=print):
    """
    Standalone evaluation of perturbation effects.
    """



    adata = sc.read_h5ad(cfg['validations']['perturbation']['dataset'])
    adata.X = np.log1p(adata.X)
    # TEMPORARY: Subset to 1/10th of the data for faster evaluation
    adata = adata[:max(1, adata.n_obs // 50)].copy()
    dataloader = create_dataloader(cfg,
                                   adata=adata,
                                   adata_name=cfg['validations']['perturbation']['dataset_name'],
                                   shuffle=False,
                                   sentence_collator=getattr(model, 'collater', None))
    all_embs = []
    for batch in tqdm(dataloader, desc=f"Perturbation Embeddings: {cfg['validations']['perturbation']['dataset_name']}"):
        with torch.no_grad():
            _, _, _, emb, _ = model._compute_embedding_for_batch(batch)
            all_embs.append(emb.cpu().detach().numpy())
    all_embs = np.concatenate(all_embs, axis=0)
    adata.obsm['X_emb'] = all_embs
    cluster_embedding(adata, 0, emb_key='X_emb', use_pca=True, job_name=cfg['experiment']['name'])
    col_id = cfg['validations']['perturbation']['pert_col']
    ctrl_label = cfg['validations']['perturbation']['ctrl_label']
    all_correlations = []
    all_ranking_scores = []
    for holdout_cell_type in adata.obs['cell_type'].unique():
        train_adata = adata[adata.obs['cell_type'] != holdout_cell_type]
        test_adata = adata[adata.obs['cell_type'] == holdout_cell_type]
        mean_pert_dfs = []
        for cell_type in train_adata.obs['cell_type'].unique():
            adata_cell = train_adata[train_adata.obs['cell_type'] == cell_type]
            ctrl_adata = adata_cell[adata_cell.obs[col_id] == ctrl_label]
            pert_adata = adata_cell[adata_cell.obs[col_id] != ctrl_label]
            mean_ctrl = ctrl_adata.obsm['X_emb'].mean(axis=0)
            pert_offsets = pert_adata.obsm['X_emb'] - mean_ctrl
            pert_df = pd.DataFrame(
                pert_offsets,
                index=pert_adata.obs_names,
                columns=[f"emb_{i}" for i in range(pert_offsets.shape[1])]
            )
            pert_df[col_id] = pert_adata.obs[col_id].values
            mean_pert_dfs.append(pert_df.groupby(col_id).mean())
        mean_pert_df = pd.concat(mean_pert_dfs).groupby(level=0).mean()
        pert_mean_offsets = {row: vals.values for row, vals in mean_pert_df.iterrows()}
        pert_mean_offsets.update({ctrl_label: np.zeros(mean_ctrl.shape[0])})
        pred_x = np.zeros_like(test_adata.obsm['X_emb']).copy()
        real_adata = sc.AnnData(X=test_adata.obsm['X_emb'], obs=test_adata.obs.copy())
        ctrl_cells = test_adata[test_adata.obs[col_id] == ctrl_label].obs.index
        pert_exclude = set()
        for i, idx in enumerate(test_adata.obs.index):
            pert = test_adata.obs.loc[idx, col_id]
            if pert not in pert_mean_offsets:
                pert_exclude.add(pert)
                continue
            elif pert == ctrl_label:
                sampled_ctrl_idx = idx
            else:
                sampled_ctrl_idx = np.random.choice(ctrl_cells)
            basal = test_adata[sampled_ctrl_idx].obsm['X_emb']
            pert_effect = pert_mean_offsets[pert]
            pred = basal + pert_effect
            pred_x[i] = pred
        pred_adata = sc.AnnData(X=pred_x, obs=test_adata.obs.copy())
        pred_adata = pred_adata[pred_adata.obs[col_id].isin(pert_mean_offsets.keys())]
        real_adata = real_adata[real_adata.obs[col_id].isin(pert_mean_offsets.keys())]
        ctrl_adata = pred_adata[pred_adata.obs[col_id] == ctrl_label]
        correlation = compute_pearson_delta(pred_adata.X, real_adata.X, ctrl_adata.X, ctrl_adata.X)
        ranking_score = compute_perturbation_ranking_score(pred_adata, real_adata)
        all_correlations.append(correlation)
        all_ranking_scores.append(ranking_score)
    logger(f"Perturbation correlation mean: {np.mean(all_correlations):.4f}")
    logger(f"Perturbation ranking mean: {np.mean(all_ranking_scores):.4f}")
    return {
        'perturbation_correlation_mean': float(np.mean(all_correlations)),
        'perturbation_ranking_mean': float(np.mean(all_ranking_scores)),
    }

def evaluate_de(model, cfg, device=None, logger=print):
    """
    Standalone evaluation of differential expression (DE).
    """



    # Get ground truth DE genes
    de_val_adata = sc.read_h5ad(cfg['validations']['diff_exp']['dataset'])
    # TEMPORARY: Subset to 1/50th of the data for faster evaluation
    # de_val_adata = de_val_adata[:max(1, de_val_adata.n_obs // 50)].copy()
    sc.pp.log1p(de_val_adata)
    sc.tl.rank_genes_groups(
        de_val_adata,
        groupby=cfg['validations']['diff_exp']['obs_pert_col'],
        reference=cfg['validations']['diff_exp']['obs_filter_label'],
        rankby_abs=True,
        n_genes=cfg['validations']['diff_exp']['top_k_rank'],
        method=cfg['validations']['diff_exp']['method'],
        use_raw=False
    )
    true_top_genes = pd.DataFrame(de_val_adata.uns['rank_genes_groups']['names']).T
    del de_val_adata
    tmp_adata = sc.read_h5ad(cfg['validations']['diff_exp']['dataset'])
    pred_exp = model._predict_exp_for_adata(
        tmp_adata,
        cfg['validations']['diff_exp']['dataset_name'],
        cfg['validations']['diff_exp']['obs_pert_col']
    )
    torch.cuda.synchronize()
    de_metrics = compute_gene_overlap_cross_pert(
        pred_exp,
        true_top_genes,
        k=cfg['validations']['diff_exp']['top_k_rank']
    )
    mean_overlap = float(np.array(list(de_metrics.values())).mean())
    logger(f"DE gene overlap mean: {mean_overlap:.4f}")
    return {
        'de_overlap_mean': mean_overlap
    }
