import os
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
import uuid

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

def compute_pearson_delta(pred, true, ctrl, ctrl_true):
    """
    pred, true, ctrl, ctrl_true are numpy arrays of shape [n_cells, n_genes],
    or you can pass means if you prefer.

    We'll compute the correlation of (pred.mean - ctrl.mean) with (true.mean - ctrl_true.mean).
    """
    return pearsonr(pred.mean(axis=0) - ctrl.mean(axis=0),
                    true.mean(axis=0) - ctrl_true.mean(axis=0))[0]

def compute_perturbation_ranking_score(adata_pred, adata_real,
                                       pert_col='gene',
                                       ctrl_pert='non-targeting'):
    """
    1) compute mean perturbation effect for each perturbation in pred and real
    2) measure how well the real perturbation matches the predicted one by rank
    returns the mean normalized rank of the true perturbation
    """
    # Step 1: compute mean effects
    mean_real_effect = _compute_mean_perturbation_effect(adata_real, pert_col, ctrl_pert)
    mean_pred_effect = _compute_mean_perturbation_effect(adata_pred, pert_col, ctrl_pert)
    all_perts = mean_real_effect.index.values

    ranks = []
    for pert in all_perts:
        real_effect = mean_real_effect.loc[pert].values.reshape(1, -1)
        pred_effects = mean_pred_effect.values

        # Compute pairwise cosine similarities
        similarities = cosine_similarity(real_effect, pred_effects).flatten()
        # where is the true row? (the same index in `all_perts`)
        true_pert_index = np.where(all_perts == pert)[0][0]

        # Sort by descending similarity
        sorted_indices = np.argsort(similarities)[::-1]
        # rank of the correct one:
        rank_of_true_pert = np.where(sorted_indices == true_pert_index)[0][0]
        ranks.append(rank_of_true_pert)

    mean_rank = np.mean(ranks)/len(all_perts)
    return mean_rank

def _compute_mean_perturbation_effect(adata, pert_col='gene', ctrl_pert='non-targeting'):
    """
    Helper to compute the mean effect (abs difference from control) for each perturbation.
    Actually we do the absolute difference from control row.
    """
    # shape: adata.X is (#cells, #genes)
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    df = pd.DataFrame(X)
    df[pert_col] = adata.obs[pert_col].values
    mean_df = df.groupby(pert_col).mean(numeric_only=True)
    # difference from control
    return np.abs(mean_df - mean_df.loc[ctrl_pert])

def get_latest_checkpoint(cfg):
    run_name = "exp_{0}_layers_{1}_dmodel_{2}_samples_{3}_max_lr_{4}_op_dim_{5}".format(
        cfg.experiment.name,
        cfg.model.nlayers,
        cfg.model.emsize,
        cfg.dataset.pad_length,
        cfg.optimizer.max_lr,
        cfg.model.output_dim)

    if cfg.experiment.checkpoint.path is None:
        return run_name, None
    chk_dir = os.path.join(cfg.experiment.checkpoint.path,
                           cfg.experiment.name)
    chk = os.path.join(chk_dir, f'last.ckpt')
    if not os.path.exists(chk) or len(chk) == 0:
        chk = None

    return run_name, chk

def compute_gene_overlap_cross_pert(DE_pred, DE_true,
                                    control_pert='non-targeting', k=50):
    all_overlaps = {}
    for c_gene in DE_pred.index:
        if c_gene == control_pert:
            continue
        all_overlaps[c_gene] = len(set(DE_true.loc[c_gene].values).intersection(
                              set(DE_pred.loc[c_gene].values))) /k

    return all_overlaps


def parse_chk_info(chk):
    chk_filename = Path(chk)
    epoch = chk_filename.stem.split('_')[-1].split('-')[1].split('=')[1]
    steps = chk_filename.stem.split('_')[-1].split('-')[2].split('=')[1]

    return int(epoch), int(steps)

def get_shapes_dict(dataset_path):
    datasets_df = pd.read_csv(dataset_path)
    sorted_dataset_names = sorted(datasets_df["names"])
    datasets_df = datasets_df.drop_duplicates() ## TODO: there should be no duplicates

    shapes_dict = {}
    dataset_path_map = {}
    dataset_group_map = {} # Name of the obs column to be used for retrieing DE scrores

    for name in sorted_dataset_names:
        shapes_dict[name] = (int(datasets_df.set_index("names").loc[name]["num_cells"]), 8000)
        dataset_path_map[name] = datasets_df.set_index("names").loc[name]["path"]

        if "groupid_for_de" in datasets_df.columns:
            dataset_group_map[name] = datasets_df.set_index("names").loc[name]["groupid_for_de"]
        else:
            # This is for backward compatibility with old datasets CSV
            dataset_group_map[name] = 'leiden'

    for row in datasets_df.iterrows():
        ngenes = row[1].num_genes
        ncells = row[1].num_cells
        name = row[1].names
        if not np.isnan(ngenes):
            shapes_dict[name] = (int(ncells), int(ngenes))

    return datasets_df, sorted_dataset_names, shapes_dict, dataset_path_map, dataset_group_map
