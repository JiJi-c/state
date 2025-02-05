"""
PerturbationDataset is used to load perturbation data from h5 files.
Originally, each file was assumed to contain a single cell type.
Now, we remove that assumption so that each file (a plate) may contain
multiple cell types. When constructing a dataset, an optional parameter
`filter_cell_type` can be provided to only keep cells matching that type.
This enables experiments such as cell state transfer (e.g. train on all but a few cell types and test on held‐out ones).
"""

from typing import Dict, List, Optional, Union, Literal
import functools
import torch
from torch.utils.data import Dataset, Subset
import h5py
import numpy as np
from pathlib import Path
import logging

# We import our mapping strategy base class for type hints
from data.mapping_strategies import BaseMappingStrategy

logger = logging.getLogger(__name__)

def safe_decode_array(arr):
    """
    Helper that accepts a numpy array and if its elements are bytes,
    decodes them to utf-8 strings. Otherwise returns the array as a list.
    """
    try:
        # Try to decode (this works if elements are bytes)
        return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in arr]
    except Exception:
        # Fallback: simply convert each element to string
        return [str(x) for x in arr]

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
        # pert_col: str = "gene",
        pert_col: str = "drug",
        # cell_type_key: str = "cell_type",
        cell_type_key: str = "cell_name",
        # batch_col: str = "gem_group",
        batch_col: str = "drug",
        control_pert: str = "non-targeting",
        embed_key: Literal["X_uce", "X_pca"] = "X_uce",
        filter_cell_type: Optional[str] = None,
        store_raw_expression: bool = False,
        random_state: int = 42,
        pert_tracker = None,
        should_yield_control_cells: bool = False,
        split_train_val_controls: bool = False,
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
        self.filter_cell_type = filter_cell_type
        self.store_raw_expression = store_raw_expression
        self.rng = np.random.RandomState(random_state)
        self.pert_tracker = pert_tracker
        self.should_yield_control_cells = should_yield_control_cells
        self.should_yield_controls = should_yield_control_cells
        self.split_train_val_controls = split_train_val_controls

        # Load file
        self.h5_file = h5py.File(self.h5_path, "r")

        # Read perturbation categories and decode safely
        raw_pert_categories = self.h5_file[f"obs/{self.pert_col}/categories"][:]
        self.pert_categories = np.array(safe_decode_array(raw_pert_categories))

        # Read cell type categories and decode safely
        raw_cell_types = self.h5_file[f"obs/{self.cell_type_key}/categories"][:]
        self.all_cell_types = np.array(safe_decode_array(raw_cell_types))

        # Read batch information
        try:
            # If batch is stored directly as numbers
            raw_batches = self.h5_file[f"obs/{self.batch_col}"][:]
            self.batch_is_categorical = False
            self.batch_categories = raw_batches.astype(str)
        except Exception:
            # Otherwise, if stored as a categorical group
            raw_batches = self.h5_file[f"obs/{self.batch_col}/categories"][:]
            self.batch_is_categorical = True
            self.batch_categories = np.array(safe_decode_array(raw_batches))

        # Determine the full set of indices in the file
        self.all_indices = np.arange(self._get_num_cells())
        # If filter_cell_type is specified, filter the indices accordingly.
        if self.filter_cell_type is not None:
            # Find indices where the cell type matches
            self.filtered_indices = np.where(self.all_cell_types == self.filter_cell_type)[0]
            if len(self.filtered_indices) == 0:
                logger.warning(f"No cells with type {self.filter_cell_type} found in {self.h5_path}")
        else:
            self.filtered_indices = self.all_indices

        # Store number of genes from the expression matrix (will use in _get_num_genes)
        self.n_genes = self._get_num_genes()

        # Also track the number of cells (after filtering)
        self.n_cells = len(self.filtered_indices)

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
        for split_name in self.split_perturbed_indices:
            # gather perturbed + control as arrays
            pert_array = np.array(list(self.split_perturbed_indices[split_name]))
            ctrl_array = np.array(list(self.split_control_indices[split_name]))
            # call the new strategy’s register
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
        underlying_idx = int(self.filtered_indices[idx])
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
        
        # Get the perturbation information
        pert_code = self.h5_file[f"obs/{self.pert_col}/codes"][underlying_idx]
        pert_name = self.pert_categories[int(pert_code)]
        if self.pert_onehot_map is not None:
            pert_onehot = self.pert_onehot_map[pert_name]
        else:
            pert_onehot = None

        # Get the cell type for this cell
        code = self.h5_file[f"obs/{self.cell_type_key}/codes"][underlying_idx]
        cell_type = self.all_cell_types[int(code)]

        # Get batch information
        if self.batch_is_categorical:
            batch_code = self.h5_file[f"obs/{self.batch_col}/codes"][underlying_idx]
            batch = self.batch_categories[int(batch_code)]
        else:
            batch = str(self.h5_file[f"obs/{self.batch_col}"][underlying_idx])

        sample = {
            "X": pert_expr,  # the perturbed cell’s data
            "basal": ctrl_expr,   # will be filled in by the mapping strategy
            "pert": pert_onehot,
            "pert_name": pert_name,
            "cell_type": cell_type,
            "gem_group": batch,
        }
        # Optionally, if raw gene expression is needed:
        if "store_raw_expression" in self.__dict__ and self.__dict__.get("store_raw_expression", False):
            sample["X_gene"] = self.fetch_gene_expression(underlying_idx)
        return sample

    def get_batch(self, idx: int) -> torch.Tensor:
        """
        Get the batch information for a given cell index. Returns a scalar tensor.
        """
        assert self.batch_onehot_map is not None, "No batch onehot map, run setup."

        if self.batch_is_categorical:
            gem_code = self.h5_file[f"obs/{self.batch_col}/codes"][idx]
            batch_name = self.batch_categories[gem_code]
        else:
            batch_name = str(self.h5_file[f"obs/{self.batch_col}"][idx])

        batch = torch.argmax(self.batch_onehot_map[batch_name])
        return batch.item()

    def get_dim_for_obsm(self, key: str) -> int:
        """
        Get the feature dimensionality of obsm data with the specified key (e.g., 'X_uce').
        """
        return self.h5_file[f"obsm/{key}"].shape[1]

    def get_cell_type(self, idx):
        code = self.h5_file[f"obs/{self.cell_type_key}/codes"][idx]
        return self.all_cell_types[int(code)]

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

    def prepare_training_splits(self, val_split: float = 0.10, rng: np.random.Generator = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split the filtered indices into train and val splits based on perturbation categories.
        This version uses only the filtered indices.
        Returns:
            A dictionary with keys "train" and "val", each containing:
                - "perturbed": np.ndarray of indices for perturbed cells (non-control)
                - "control": np.ndarray of indices for control cells
        """
        if rng is None:
            rng = np.random.default_rng(42)
        # Use the filtered indices only
        indices = self.filtered_indices
        pert_codes = self.h5_file[f"obs/{self.pert_col}/codes"][indices]
        pert_names = np.array(self.pert_categories)[pert_codes]
        
        # Group indices by perturbation (excluding control)
        pert_groups = {}
        for pert in np.unique(pert_names):
            if pert == self.control_pert:
                continue
            group_indices = indices[pert_names == pert]
            pert_groups[pert] = group_indices
        
        total_cells = sum(len(arr) for arr in pert_groups.values())
        target_val_cells = val_split * total_cells

        # Greedy selection to form validation set (by perturbation groups)
        train_perts = []
        val_perts = []
        current_val = 0
        pert_list = list(pert_groups.keys())
        rng.shuffle(pert_list)
        for pert in pert_list:
            group_size = len(pert_groups[pert])
            if abs((current_val + group_size) - target_val_cells) < abs(current_val - target_val_cells):
                val_perts.append(pert)
                current_val += group_size
            else:
                train_perts.append(pert)
        # Get indices
        train_indices = np.concatenate([pert_groups[p] for p in train_perts]) if train_perts else np.array([], dtype=int)
        val_indices = np.concatenate([pert_groups[p] for p in val_perts]) if val_perts else np.array([], dtype=int)

        # Also get control indices (from filtered indices where pert==control_pert)
        ctrl_indices = indices[pert_names == self.control_pert]

        return {
            "train": {"perturbed": train_indices, "control": ctrl_indices},
            "val": {"perturbed": val_indices, "control": ctrl_indices},
        }

    def prepare_fewshot_splits(self, few_shot_percent: float = 0.3, val_split: float = 0.15, rng: np.random.Generator = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Similar to prepare_training_splits but splits the perturbed cells into train/val/test for few-shot learning.
        Operates on the filtered indices.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        indices = self.filtered_indices
        pert_codes = self.h5_file[f"obs/{self.pert_col}/codes"][indices]
        pert_names = np.array(self.pert_categories)[pert_codes]
        pert_groups = {}
        for pert in np.unique(pert_names):
            if pert == self.control_pert:
                continue
            pert_groups[pert] = indices[pert_names == pert]
        # Split into train_val and test based on few_shot_percent
        total_cells = sum(len(arr) for arr in pert_groups.values())
        target_test_cells = (1 - few_shot_percent) * total_cells

        train_val_perts = []
        test_perts = []
        current_test = 0
        pert_list = list(pert_groups.keys())
        rng.shuffle(pert_list)
        for pert in pert_list:
            group_size = len(pert_groups[pert])
            if abs((current_test + group_size) - target_test_cells) < abs(current_test - target_test_cells):
                test_perts.append(pert)
                current_test += group_size
            else:
                train_val_perts.append(pert)
        # Then further split train_val_perts into train and val
        train_groups = {}
        val_groups = {}
        for pert in train_val_perts:
            arr = pert_groups[pert]
            n = len(arr)
            n_val = int(n * val_split)
            rng.shuffle(arr)
            val_groups[pert] = arr[:n_val]
            train_groups[pert] = arr[n_val:]
        train_indices = np.concatenate(list(train_groups.values())) if train_groups else np.array([], dtype=int)
        val_indices = np.concatenate(list(val_groups.values())) if val_groups else np.array([], dtype=int)
        test_indices = np.concatenate([pert_groups[p] for p in test_perts]) if test_perts else np.array([], dtype=int)
        ctrl_indices = indices[pert_names == self.control_pert]

        return {
            "train": {"perturbed": train_indices, "control": ctrl_indices},
            "val": {"perturbed": val_indices, "control": ctrl_indices},
            "test": {"perturbed": test_indices, "control": ctrl_indices},
        }

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
        return torch.tensor(row_data)

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
    def collate_fn(batch, transform=None):
        """
        Custom collate function that can apply transforms on batched data.

        The transform is bound to the dataset instance via a partial, hence why this
        is a static method.
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
            batch_dict["X"] = transform.encode(batch_dict["X"])
            batch_dict["basal"] = transform.encode(batch_dict["basal"])

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

        # set them in the dataset
        self.split_perturbed_indices[split] = set(perturbed_indices)
        self.split_control_indices[split] = set(control_indices)

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
            indices = self.h5_file["X/indices"][:]
            n_cols = indices.max() + 1
        return n_cols

    def _get_num_cells(self) -> int:
        """Return the total number of cells in the file."""
        try:
            n_rows = self.h5_file["X"].shape[0]
        except Exception:
            # If stored as sparse
            indptr = self.h5_file["X/indptr"][:]
            n_rows = len(indptr) - 1
        return n_rows

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
