import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
import wandb


def cluster_embedding(adata, current_step, emb_key='X_emb', use_pca=True, job_name=''):
    embedding = PCA(n_components=2).fit_transform(adata.obsm[emb_key])

    # Get the cell type information as a categorical series
    cell_types = adata.obs['cell_type'].astype('category')

    # Create a color palette based on the number of unique cell types
    palette = sns.color_palette("hsv", len(cell_types.cat.categories))
    color_dict = dict(zip(cell_types.cat.categories, palette))

    # Instead of using .map (which may fail with a MultiIndex), use a list comprehension
    colors = [color_dict[ct] for ct in cell_types]

    # Plot the embedding
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5, alpha=0.7)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Embedding ({'PCA'}) for {emb_key} ({job_name} - Iteration: {current_step})")

    # Create legend handles for each cell type
    handles = [
        mlines.Line2D([], [], color=color_dict[ct], marker='o', linestyle='None', markersize=6, label=ct)
        for ct in cell_types.cat.categories
    ]
    plt.legend(handles=handles, title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig = plt.gcf()
    wandb.log({f"Clusters using embedding Iteration: {current_step}": fig})