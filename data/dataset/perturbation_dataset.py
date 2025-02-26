"""
PerturbationDataset is used to load perturbation data from h5 files.
Originally, each file was assumed to contain a single cell type.
Now, we remove that assumption so that each file (a plate) may contain
multiple cell types. 
"""

from typing import Dict, List, Optional, Union, Literal
import functools
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Subset
from data.utils.data_utils import safe_decode_array, H5MetadataCache, GlobalH5MetadataCache
import h5py
import numpy as np
from pathlib import Path
import logging

# We import our mapping strategy base class for type hints
from data.mapping_strategies import BaseMappingStrategy

logger = logging.getLogger(__name__)

class PerturbationDataset(Dataset):
    """
    Dataset class for loading perturbation data from h5 files.

    Each instance handles a single dataset-cell_type combination. Therefore this class is responsible for
    serving a single dataset/cell_type pair. Future improvements will also allow for splitting on
    the perturbation level.

    Currently there are three strategies for mapping basal cells to perturbed cells:
    - "batch": Basal cells are sampled from the same batch as the perturbed cell
    - "random": Basal cells are sampled randomly from the same cell type as the perturbed cell
    - "nearest": Basal cells are sampled from the nearest neighbors of the perturbed cell

    A control cell is always mapped to a perturbed cell within the same dataset and with the same cell type.
    """

    def __init__(
        self,
        name: str,
        h5_path: Union[str, Path],
        mapping_strategy: BaseMappingStrategy,
        pert_onehot_map: Optional[Dict[str, int]] = None,
        batch_onehot_map: Optional[Dict[str, int]] = None,
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "gem_group",
        control_pert: str = "non-targeting",
        embed_key: Literal["X_uce", "X_pca"] = "X_uce",
        store_raw_expression: bool = False,
        random_state: int = 42,
        pert_tracker = None,
        should_yield_control_cells: bool = True,
        preload_data: bool = False,
        **kwargs,
    ):
        """
        Args:
            name: Name of the dataset
            h5_path: Path to the h5 file containing the dataset
            mapping_strategy: Strategy for mapping basal cells to perturbed cells, one of "batch", "random", "nearest"
            pert_onehot_map: Global mapping of perturbation names to one-hot encodings
            batch_onehot_map: Global mapping of batch names to one-hot encodings
            pert_col: Column in the h5 file containing perturbation information
            cell_type_key: Column in the h5 file containing cell type information
            batch_col: Column in the h5 file containing batch information
            control_pert: Name of the control perturbation
            embed_key: Key in the h5 file containing the expression data, one of "X" or "X_uce"
            random_state: Random seed for reproducibility
            pert_tracker: PerturbationTracker instance for tracking valid perturbations
            should_yield_control_cells: If True, control cells will be included in the dataset
        """
        super().__init__()
        self.name = name
        self.h5_path = Path(h5_path)
        self.mapping_strategy = mapping_strategy
        self.pert_onehot_map = pert_onehot_map
        self.batch_onehot_map = batch_onehot_map
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.store_raw_expression = store_raw_expression
        self.rng = np.random.RandomState(random_state)
        self.pert_tracker = pert_tracker
        self.should_yield_control_cells = should_yield_control_cells
        self.should_yield_controls = should_yield_control_cells

        self.metadata_cache = GlobalH5MetadataCache().get_cache(
                str(self.h5_path),
                self.pert_col,
                self.cell_type_key,
                self.control_pert,
                self.batch_col,
            )

        # Load file
        self.h5_file = h5py.File(self.h5_path, "r")

        # Use metadata cache for categories (perturbation, cell type, control cells)
        self.pert_categories = self.metadata_cache.pert_categories
        self.all_cell_types = self.metadata_cache.cell_type_categories
        self.control_mask = self.metadata_cache.control_mask

        # Determine the full set of indices in the file
        self.all_indices = np.arange(self.metadata_cache.n_cells)

                # Determine the full set of indices in the file
        self.all_indices = np.arange(self._get_num_cells())

        # Store number of genes from the expression matrix (will use in _get_num_genes)
        self.n_genes = self._get_num_genes()

        # Also track the number of cells (after filtering)
        self.n_cells = len(self.all_indices)

        # TODO-Abhi: we should move this logic to MultiDatasetPerturbationDataModule
        if self.pert_tracker is not None:
            cell_type = str(self.cell_type_categories[0])  # single type per file
            dataset_name = name
            self.valid_perts = self.pert_tracker.track_celltype_perturbations(self.h5_file, dataset_name, cell_type)

        # We'll store the indices for each split.
        self.split_perturbed_indices = {
            "train": set(),
            "train_eval": set(),
            "val": set(),
            "test": set(),
        }
        self.split_control_indices = {
            "train": set(),
            "train_eval": set(),
            "val": set(),
            "test": set(),
        }

        self.preloaded_data = {}
        if preload_data:
            logger.info(f"[{self.name}] Preloading all data into memory...")
            self._preload_all_data()

    def set_store_raw_expression(self, flag: bool):
        """
        Dynamically enable/disable whether this dataset yields raw gene expression.

        This is so that we don't load raw expression, which can be expensive, if we are only training
        latent models.
        """
        self.store_raw_expression = flag
        logger.info(f"[{self.name}] to yield raw gene expression: {flag}")

    def reset_mapping_strategy(self, new_strategy: BaseMappingStrategy, stage="train", **strategy_kwargs):
        """
        Re-run register_split_indices(...) for each known split in this dataset,
        using the new mapping strategy. This preserves the exact same indices
        (perturbed + control) but changes how basal controls will be sampled.
        """
        self.mapping_strategy = new_strategy(**strategy_kwargs)
        self.mapping_strategy.stage = stage
        all_splits = list(self.split_perturbed_indices.keys())
        for split_name in all_splits:
            # gather perturbed + control as arrays
            pert_array = np.array(sorted(list(self.split_perturbed_indices[split_name])))
            ctrl_array = np.array(sorted(list(self.split_control_indices[split_name])))
            # call the new strategy’s register
            if len(pert_array) > 0 and len(ctrl_array) > 0:
                self.mapping_strategy.register_split_indices(self, split_name, pert_array, ctrl_array)

    def __getitem__(self, idx: int):
        """
        Returns a dictionary with:
            - 'X': the (possibly transformed) expression of the perturbed cell
            - 'basal': the control cell’s expression as chosen by the mapping strategy
            - 'pert': the one-hot encoding (or other featurization) for the perturbation
            - 'pert_name': the perturbation name
            - 'cell_type': the cell type (from the full array)
            - 'gem_group': the batch (as an int or string)
        
        The index `idx` here is into the filtered set of cells.
        """
        # Map idx to the underlying file index
        underlying_idx = int(self.all_indices[idx])
        split = self._find_split_for_idx(underlying_idx)

        # Get expression from the h5 file.
        # For now, we assume the data is stored in "X" (could be counts) and/or in obsm (embed_key)
        # (It is up to the downstream code to decide whether to use raw gene expression or a precomputed embedding.)
        if self.embed_key:
            # For example, get the embedding from obsm/X_uce
            pert_expr = torch.tensor(self.h5_file[f"obsm/{self.embed_key}"][underlying_idx])
        else:
            # Otherwise, fetch from X directly
            pert_expr = self.fetch_gene_expression(underlying_idx)

        pert_expr, ctrl_expr = self.mapping_strategy.get_mapped_expressions(self, split, underlying_idx)
        
        # Get perturbation information using metadata cache
        pert_code = self.metadata_cache.pert_codes[underlying_idx]
        pert_name = self.metadata_cache.pert_categories[pert_code]
        if self.pert_onehot_map is not None:
            # map across all files to a consistent one hot encoding
            pert_onehot = self.pert_onehot_map[pert_name]
        else:
            pert_onehot = None

        # Get cell type using metadata cache
        cell_type_code = self.metadata_cache.cell_type_codes[underlying_idx]
        cell_type = self.metadata_cache.cell_type_categories[cell_type_code]

        # Get batch information
        batch = self.metadata_cache.batch_codes[underlying_idx]

        sample = {
            "X": pert_expr,  # the perturbed cell’s data
            "basal": ctrl_expr,   # will be filled in by the mapping strategy
            "pert": pert_onehot,
            "pert_name": pert_name,
            "cell_type": cell_type,
            "gem_group": batch,
        }
        # Optionally, if raw gene expression is needed:
        if self.store_raw_expression:
            sample["X_gene"] = self.fetch_gene_expression(underlying_idx)
        return sample

    def get_batch(self, idx: int) -> torch.Tensor:
        """
        Get the batch information for a given cell index. Returns a scalar tensor.
        """
        assert self.batch_onehot_map is not None, "No batch onehot map, run setup."
        batch_name = self.metadata_cache.batch_names[idx]
        batch = torch.argmax(self.batch_onehot_map[batch_name])
        return batch.item()

    def get_dim_for_obsm(self, key: str) -> int:
        """
        Get the feature dimensionality of obsm data with the specified key (e.g., 'X_uce').
        """
        return self.h5_file[f"obsm/{key}"].shape[1]

    def get_cell_type(self, idx):
        code = self.metadata_cache.cell_type_codes[idx]
        return self.metadata_cache.cell_type_categories[code]
    
    def get_all_cell_types(self, indices):
        codes = self.metadata_cache.cell_type_codes[indices]
        return self.metadata_cache.cell_type_categories[codes]
    
    def get_perturbation_name(self, idx):
        pert_code = self.metadata_cache.pert_codes[idx]
        return self.metadata_cache.pert_categories[pert_code]

    # TODO-Abhi: can we move perturbed idx logic and control idx logic internally so these don't have to be passed in?
    def to_subset_dataset(
        self,
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
        subsample_fraction: float = 1.0,
    ) -> Subset:
        """
        Creates a Subset of this dataset that includes only the specified perturbed_indices.
        If `self.should_yield_control_cells` flag is True, the Subset will also yield control cells.

        Args:
            split: Name of the split to create, one of 'train', 'val', 'test', or 'train_eval'
            perturbed_indices: Indices of perturbed cells to include
            control_indices: Indices of control cells to include
            subsample_fraction: Fraction of perturbed cells to include, the rest are ignored (default 1.0)
        """
        if subsample_fraction < 1.0:
            # randomly subsample the perturbed indices
            n_keep = int(len(perturbed_indices) * subsample_fraction)
            perturbed_indices = self.rng.choice(perturbed_indices, size=n_keep, replace=False)

        # sort them for stable ordering
        perturbed_indices = np.sort(perturbed_indices)
        control_indices = np.sort(control_indices)

        # Register them in the dataset
        self._register_split_indices(split, perturbed_indices, control_indices)

        # Return a Subset containing perturbed cells and optionally control cells
        if self.should_yield_control_cells:
            all_indices = np.concatenate([perturbed_indices, control_indices])
            return Subset(self, all_indices)
        else:
            return Subset(self, perturbed_indices)

    def fetch_gene_expression(self, idx: int) -> torch.Tensor:
        if hasattr(self, "preloaded_data") and "X" in self.preloaded_data:
            return torch.tensor(self.preloaded_data["X"][idx], dtype=torch.float32)

        attrs = dict(self.h5_file["X"].attrs)
        if attrs["encoding-type"] == "csr_matrix":
            indptr = self.h5_file["/X/indptr"]
            start_ptr = indptr[idx]
            end_ptr = indptr[idx + 1]
            sub_data = torch.tensor(self.h5_file["/X/data"][start_ptr:end_ptr], dtype=torch.float32)
            sub_indices = torch.tensor(self.h5_file["/X/indices"][start_ptr:end_ptr], dtype=torch.long)
            counts = torch.sparse_csr_tensor(
                torch.tensor([0], dtype=torch.long),
                sub_indices,
                sub_data,
                (1, self.n_genes),
            )
            data = counts.to_dense().squeeze()
        else:
            row_data = self.h5_file["/X"][idx]
            data = torch.tensor(row_data)
        return data

    def fetch_obsm_expression(self, idx: int, key: str) -> torch.Tensor:
        if hasattr(self, "preloaded_data") and key in self.preloaded_data:
            return torch.tensor(self.preloaded_data[key][idx], dtype=torch.float32)
        row_data = self.h5_file[f"/obsm/{key}"][idx]
        return torch.tensor(row_data, dtype=torch.float32)

    def get_gene_names(self) -> List[str]:
        """
        Get the gene names, which are under adata.var.index, using h5.
        """
        try:
            genes = self.h5_file["var/gene_name"][:].astype(str).tolist()
        except:
            try:
                categories = self.h5_file["var/gene_name/categories"][:].astype(str)
                codes = self.h5_file["var/gene_name/codes"][:]
                genes = categories[codes].tolist()
            except:
                genes = self.h5_file["var/_index"][:].astype(str).tolist()

        return genes

    ##############################
    # Static methods
    ##############################
    @staticmethod
    def collate_fn(batch, transform=None, cell_sentence_len=32):
        """
        Custom collate that reshapes data into sequences.
        """
        # First do normal collation
        batch_dict = {
            "X": torch.stack([item["X"] for item in batch]),
            "basal": torch.stack([item["basal"] for item in batch]),
            "pert": torch.stack([item["pert"] for item in batch]),
            "pert_name": [item["pert_name"] for item in batch],
            "cell_type": [item["cell_type"] for item in batch],
            "gem_group": torch.tensor([item["gem_group"] for item in batch]),
        }

        # If the first sample has "X_gene", assume the entire batch does
        if "X_gene" in batch[0]:
            batch_dict["X_gene"] = torch.stack([item["X_gene"] for item in batch])

        # Apply transform if provided
        if transform is not None:
            batch_dict["X"] = torch.log1p(batch_dict["X"])
            batch_dict["basal"] = torch.log1p(batch_dict["basal"])

        # # Reshape into sequences
        # B = len(batch) // cell_sentence_len
        # for k in ["X", "basal", "pert"]:
        #     if torch.is_tensor(batch_dict[k]):
        #         batch_dict[k] = batch_dict[k].view(B, cell_sentence_len, -1)
                
        return batch_dict

    ##############################
    # Utility methods
    ##############################
    def _register_split_indices(self, split: str, perturbed_indices: np.ndarray, control_indices: np.ndarray):
        """
        Register which cell indices belong to the perturbed vs. control set for
        a given split.

        These are passed to the mapping strategy to let it build its internal structures as needed.
        """
        if split not in self.split_perturbed_indices:
            raise ValueError(f"Invalid split {split}")

        # update them in the dataset
        self.split_perturbed_indices[split] |= set(perturbed_indices)
        self.split_control_indices[split] |= set(control_indices)

        # forward these to the mapping strategy
        self.mapping_strategy.register_split_indices(self, split, perturbed_indices, control_indices)

    def _find_split_for_idx(self, idx: int) -> Optional[str]:
        """Utility to find which split (train/val/test) this idx belongs to."""
        for s in self.split_perturbed_indices.keys():
            if idx in self.split_perturbed_indices[s] or idx in self.split_control_indices[s]:
                return s
        return None

    def _get_num_genes(self) -> int:
        """Return the number of genes in the X matrix."""
        try:
            n_cols = self.h5_file["X"].shape[1]
        except Exception:
            # If stored as sparse, infer from indices
            try:
                indices = self.h5_file["X/indices"][:]
                n_cols = indices.max() + 1
            except:
                n_cols = self.h5_file["obsm/X_hvg"].shape[1]
        return n_cols

    def _get_num_cells(self) -> int:
        """Return the total number of cells in the file."""
        try:
            n_rows = self.h5_file["X"].shape[0]
        except Exception:
            try:
                # If stored as sparse
                indptr = self.h5_file["X/indptr"][:]
                n_rows = len(indptr) - 1
            except Exception:
                # if this also fails, fall back to obsm
                n_rows = self.h5_file["obsm/X_hvg"].shape[0]
        return n_rows

    def get_pert_name(self, idx: int) -> str:
        """Get perturbation name for a given index."""
        return self.metadata_cache.pert_names[idx]

    def __len__(self) -> int:
        return self.n_cells

    def __getstate__(self):
        """
        Return a dictionary of this dataset's state without the open h5 file object.
        """
        # Copy the object's dict
        state = self.__dict__.copy()
        # Remove the open file object if it exists
        if "h5_file" in state:
            # We'll also store whether it's currently open, so that we can re-open later if needed
            del state["h5_file"]
        return state

    def __setstate__(self, state):
        """
        Reconstruct the dataset after unpickling. Re-open the HDF5 file by path.
        """
        # TODO-Abhi: remove this before release
        self.__dict__.update(state)
        # This ensures that after we unpickle, we have a valid h5_file handle again
        self.h5_file = h5py.File(self.h5_path, "r")
        self.metadata_cache = GlobalH5MetadataCache().get_cache(
            str(self.h5_path),
            self.pert_col,
            self.cell_type_key,
            self.control_pert,
            self.batch_col,
        )

    def _preload_all_data(self):
        """Preload all data into memory."""
        logger.info(f"[{self.name}] Preloading all data into memory...")

        # Load gene expression
        self.preloaded_data["X"] = torch.stack([self.fetch_gene_expression(i) for i in range(self.n_cells)])

        # Load embeddings if used
        if self.embed_key:
            self.preloaded_data[self.embed_key] = torch.stack(
                [self.fetch_obsm_expression(i, self.embed_key) for i in range(self.n_cells)]
            )

        logger.info(f"[{self.name}] Preload complete.")
