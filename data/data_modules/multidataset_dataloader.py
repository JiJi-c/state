from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Literal, Set
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from data.utils.data_utils import generate_onehot_map  # assuming this helper exists
from lightning.pytorch import LightningDataModule
from data.dataset.perturbation_dataset import PerturbationDataset
from data.data_modules.perturbation_tracker import PerturbationTracker
from data.data_modules.samplers import PerturbationBatchSampler
from data.data_modules.tasks import TaskSpec, TaskType
from data.mapping_strategies import (
    CentroidMappingStrategy,
    ClusteringMappingStrategy,
    BatchMappingStrategy,
    RandomMappingStrategy,
    NearestNeighborMappingStrategy,
    PseudoNearestMappingStrategy,
    PseudoBulkMappingStrategy,
)
from data.transforms.pca import PCATransform  # if needed

import h5py
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class MetadataConcatDataset(ConcatDataset):
    """
    A ConcatDataset that maintains metadata from constituent datasets.
    Validates that all datasets share the same metadata values.
    """

    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)

        # Get metadata from first dataset
        first_dataset = datasets[0].dataset
        self.embed_key = first_dataset.embed_key
        self.control_pert = first_dataset.control_pert
        self.pert_col = first_dataset.pert_col
        self.batch_col = first_dataset.batch_col
        self.soft_mapped_nn_ctrl_indices = None
        self.mapped_nn_ctrl_indices = None

        self.pert_tracker = getattr(first_dataset, "pert_tracker", None)

        # Validate all datasets have same metadata
        for dataset in datasets:
            base_dataset = dataset.dataset
            if base_dataset.embed_key != self.embed_key:
                raise ValueError("All datasets must have same embed_key")
            if base_dataset.control_pert != self.control_pert:
                raise ValueError("All datasets must have same control_pert")
            if base_dataset.pert_col != self.pert_col:
                raise ValueError("All datasets must have same pert_col")
            if base_dataset.batch_col != self.batch_col:
                raise ValueError("All datasets must have same batch_col")

        # TODO-Abhi: Aggregate individual dataset nearest neighbor maps, based on
        # per-dataset indices, to create a unified mapping across all datasets
        # for evals.


class MultiDatasetPerturbationDataModule(LightningDataModule):
    """
    A unified data module that sets up train/val/test splits for multiple dataset/celltype
    combos. Allows zero-shot, few-shot tasks, and uses a pluggable mapping strategy
    (batch, random, nearest) to match perturbed cells with control cells.
    """

    def __init__(
        self,
        train_specs: List[TaskSpec],
        test_specs: List[TaskSpec],
        data_dir: str,
        dataset_subsampling: Optional[Dict[str, float]] = {},
        batch_size: int = 128,
        num_workers: int = 8,
        few_shot_percent: float = 0.3,
        random_seed: int = 42,
        val_split: float = 0.10,
        embed_key: Optional[Literal["X_uce", "X_pca", "X_scGPT"]] = None,
        output_space: Literal["gene", "latent"] = "gene",
        basal_mapping_strategy: Literal["batch", "random", "nearest"] = "batch",
        n_basal_samples: int = 1,
        k_neighbors: int = 10,
        eval_pert: Optional[str] = None,
        should_yield_control_cells: bool = True,
        split_train_val_controls: bool = False,
        preload_data: bool = False,
        **kwargs,
    ):
        """
        This class is responsible for serving multiple PerturbationDataset's each of which is specific
        to a dataset/cell type combo. It sets up training, validation, and test splits for each dataset
        and cell type, and uses a pluggable mapping strategy to match perturbed cells with control cells.

        Args:
            train_specs: A list of TaskSpec for training tasks
            test_specs: A list of TaskSpec for testing tasks
            data_dir: Path to the root directory containing the H5 files
            dataset_subsampling: Optional dict specifying the fraction to subsample each dataset
            batch_size: Batch size for PyTorch DataLoader
            num_workers: Num workers for PyTorch DataLoader
            few_shot_percent: Fraction of data to use for few-shot tasks
            random_seed: For reproducible splits & sampling
            val_split: Fraction of data to use as validation portion
            embed_key: Embedding key or matrix in the H5 file to use for feauturizing cells
            output_space: The output space for model predictions (gene or latent, which uses embed_key)
            basal_mapping_strategy: One of {"batch","random","nearest","ot"}
            n_basal_samples: Number of control cells to sample per perturbed cell
            k_neighbors: For nearest or OT approaches, how many neighbors to store or sample
            eval_pert: Name of dataset to evaluate shared perturbations against, if any
        """
        super().__init__()
        self.train_specs = train_specs
        self.test_specs = test_specs
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.few_shot_percent = few_shot_percent
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.val_split = val_split
        self.embed_key = embed_key
        self.output_space = output_space
        self.basal_mapping_strategy = basal_mapping_strategy
        self.n_basal_samples = n_basal_samples
        self.k_neighbors = k_neighbors
        self.should_yield_control_cells = should_yield_control_cells
        self.split_train_val_controls = split_train_val_controls
        self.preload_data = preload_data

        self.pert_col = 'drug'
        self.control_pert = 'DMSO_TF'
        self.batch_col = 'drug'
        self.cell_type_key = 'cell_name'

        self.train_datasets: List[Dataset] = []
        self.train_eval_datasets: List[Dataset] = []
        self.val_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []

        # Build the chosen mapping strategy
        self.mapping_strategy_cls = {
            "centroid": CentroidMappingStrategy,
            "clustering": ClusteringMappingStrategy,
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
            "nearest": NearestNeighborMappingStrategy,
            "pseudo_nearest": PseudoNearestMappingStrategy,
            "pseudobulk": PseudoBulkMappingStrategy,
        }[basal_mapping_strategy]

        self.neighborhood_fraction = kwargs.get(
            "neighborhood_fraction", 0.0
        )  # move this to a mapping strategy specific config

        # Use the chosen data transform if applicable
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = PCATransform(n_components=200, device=device) if embed_key == "X_pca" else None
        
        # TODO-Abhi: is there a way to detect if the transform is needed?
        self.transform = True if embed_key == "X_hvg" and self.pert_col == 'drug' else False

        if not self.split_train_val_controls:
            logger.info("NOTE: Control cells will be shared between train and val splits.")

        # Few-shot store: (dataset_name, cell_type) -> dict of splits
        self.fewshot_splits: Dict[(str, str), Dict[str, np.ndarray]] = {}

        # Global perturbation map
        self.all_perts: Set[str] = set()
        self.pert_onehot_map: Optional[Dict[str, torch.Tensor]] = None
        self.batch_onehot_map: Optional[Dict[str, torch.Tensor]] = None
        self.celltype_onehot_map: Optional[Dict[str, torch.Tensor]] = None

        # Dataset subsampling.
        # TODO: move this to a separate module called filtering to also include filters
        self.dataset_subsampling = dataset_subsampling
        for dataset, fraction in self.dataset_subsampling.items():
            logger.info(f"Dataset {dataset} will be subsampled to {fraction:.1%}")

        # track shared perts for evaluation
        self.eval_pert = eval_pert
        self.pert_tracker = PerturbationTracker() if eval_pert else None

        self._pseudo_nearest_global_basal = None
        self._pseudo_nearest_pert_offsets = None

        self._setup_global_maps()

    def _find_dataset_files(self, dataset_name: str) -> List[Path]:
        """
        Instead of returning a mapping from cell type to file,
        return a list of all plate (h5) files in the directory for the dataset.
        """
        pattern = "*.h5"
        working_folder = self.data_dir / dataset_name
        if not working_folder.exists():
            raise FileNotFoundError(f"No directory named {working_folder}")
        files = sorted(working_folder.glob(pattern))
        return files

    def _setup_global_maps(self):
        """
        Set up global one-hot maps for perturbations and batches.
        For perturbations, we scan through all files in all train_specs and test_specs.
        """
        all_perts = set()
        all_batches = set()
        dataset_names = {spec.dataset for spec in (self.train_specs + self.test_specs)}
        for ds_name in dataset_names:
            files = self._find_dataset_files(ds_name)
            for fpath in files:
                with h5py.File(files[fpath], "r") as f:
                    pert_arr = f["obs/{}/categories".format(self.pert_col)][:]
                    perts = set(safe_decode_array(pert_arr))
                    all_perts.update(perts)

                    try:
                        batch_arr = f["obs/{}/categories".format(self.batch_col)][:]
                    except KeyError:
                        batch_arr = f["obs/{}".format(self.batch_col)][:]
                    batches = set(safe_decode_array(batch_arr))
                    all_batches.update(batches)

        # Create one-hot maps
        self.pert_onehot_map = generate_onehot_map(all_perts)
        self.batch_onehot_map = generate_onehot_map(all_batches)

    def _group_specs_by_dataset(self, specs: List[TaskSpec]) -> Dict[str, List[TaskSpec]]:
        """
        Group TaskSpec objects by dataset name.
        """
        dmap = {}
        for spec in specs:
            dmap.setdefault(spec.dataset, []).append(spec)
        return dmap
    
    def _setup_global_fewshot_splits(self):
        """
        Do a global scan across files to decide train/test perturbation splits for each fewshot cell type.
        """
        self.fewshot_splits = {}  # {cell_type: {"train_perts": set(), "test_perts": set()}}
        
        # Find all fewshot specs
        fewshot_specs = [s for s in self.test_specs if s.task_type == TaskType.FEWSHOT]
        
        for spec in fewshot_specs:
            ct = spec.cell_type
            # Aggregate counts across all files
            pert_counts = defaultdict(int)
            
            files = self._find_dataset_files(spec.dataset)
            for fpath in files:
                with h5py.File(fpath, "r") as f:
                    # Get cell type codes
                    ct_code = np.where(safe_decode_array(f[f"obs/{self.cell_type_key}/categories"][:]) == ct)[0][0]
                    ct_mask = f[f"obs/{self.cell_type_key}/codes"][:] == ct_code
                    
                    # Get perturbation names for this cell type's cells
                    pert_codes = f[f"obs/{self.pert_col}/codes"][:][ct_mask]
                    pert_categories = safe_decode_array(f[f"obs/{self.pert_col}/categories"][:])
                    pert_names = pert_categories[pert_codes]
                    
                    # Count non-control perturbations
                    for pert in np.unique(pert_names):
                        if pert != self.control_pert:
                            pert_counts[pert] += np.sum(pert_names == pert)

            # Split perturbations greedily based on counts
            sorted_perts = sorted([(p, c) for p, c in pert_counts.items()], 
                                key=lambda x: x[1], reverse=True)
            
            total_cells = sum(c for _, c in sorted_perts)
            target_train = total_cells * self.few_shot_percent
            
            train_perts = set()
            test_perts = set()
            current_train = 0
            
            for pert, count in sorted_perts:
                if abs((current_train + count) - target_train) < abs(current_train - target_train):
                    train_perts.add(pert)
                    current_train += count
                else:
                    test_perts.add(pert)

            self.fewshot_splits[ct] = {
                "train_perts": train_perts,
                "test_perts": test_perts
            }
            logger.info(f"Cell type {ct}: {len(train_perts)} train perts, {len(test_perts)} test perts")

    def _setup_training_datasets(self):
        """
        Set up training datasets with proper handling of zeroshot/fewshot splits.
        """
        # First compute global fewshot splits
        self._setup_global_fewshot_splits()
        
        # Get zeroshot cell types
        zeroshot_cts = {spec.cell_type for spec in self.test_specs 
                        if spec.task_type == TaskType.ZEROSHOT}
        
        # Process each training spec
        train_map = self._group_specs_by_dataset(self.train_specs)
        for ds_name, specs in train_map.items():
            files = self._find_dataset_files(ds_name)
            for fpath in files:
                fpath = files[fpath]
                # Create base dataset
                ds = PerturbationDataset(
                    name=ds_name,
                    h5_path=fpath,
                    mapping_strategy=self.mapping_strategy_cls(
                        random_state=self.random_seed, 
                        n_basal_samples=self.n_basal_samples
                    ),
                    embed_key=self.embed_key,
                    pert_onehot_map=self.pert_onehot_map,
                    batch_onehot_map=self.batch_onehot_map,
                    pert_col=self.pert_col,
                    cell_type_key="cell_name",
                    batch_col="drug",
                    control_pert="DMSO_TF",
                    random_state=self.random_seed,
                    should_yield_control_cells=self.should_yield_control_cells,
                )
                
                with h5py.File(fpath, "r") as f:
                    # Get all cell types in this file
                    cell_types = safe_decode_array(f[f"obs/{self.cell_type_key}/categories"][:])
                    cell_codes = f[f"obs/{self.cell_type_key}/codes"][:]
                    pert_categories = safe_decode_array(f[f"obs/{self.pert_col}/categories"][:])
                    pert_codes = f[f"obs/{self.pert_col}/codes"][:]
                    
                    for ct_idx, ct in enumerate(cell_types):
                        # Skip if no cells of this type
                        ct_mask = cell_codes == ct_idx
                        if not np.any(ct_mask):
                            continue
                            
                        if ct in zeroshot_cts:
                            # For zeroshot, all cells (including controls) go to test
                            ct_indices = np.where(ct_mask)[0]
                            test_subset = ds.to_subset_dataset("test", ct_indices, np.array([]))
                            self.test_datasets.append(test_subset)
                            
                        elif ct in self.fewshot_splits:
                            # For fewshot, split based on pre-computed perturbation allocation
                            train_perts = self.fewshot_splits[ct]["train_perts"]
                            test_perts = self.fewshot_splits[ct]["test_perts"]
                            
                            # Get indices for this cell type
                            ct_indices = np.where(ct_mask)[0]
                            
                            # Split controls according to fewshot_split
                            ctrl_mask = pert_categories[pert_codes[ct_indices]] == self.control_pert
                            ctrl_indices = ct_indices[ctrl_mask]
                            rng = np.random.default_rng(self.random_seed)
                            rng.shuffle(ctrl_indices)
                            n_train_controls = int(len(ctrl_indices) * self.few_shot_percent)
                            train_controls = ctrl_indices[:n_train_controls]
                            test_controls = ctrl_indices[n_train_controls:]
                            
                            # Split perturbed cells based on perturbation allocation
                            pert_indices = ct_indices[~ctrl_mask]
                            pert_names = pert_categories[pert_codes[pert_indices]]
                            
                            # Split perturbed cells based on perturbation allocation
                            train_pert_mask = np.isin(pert_names, list(train_perts))
                            train_val_pert_indices = pert_indices[train_pert_mask]
                            test_pert_indices = pert_indices[~train_pert_mask]

                            # Further split train_val into train/val
                            if len(train_val_pert_indices) > 0:
                                rng.shuffle(train_val_pert_indices)  # In-place shuffle
                                n_val = int(len(train_val_pert_indices) * self.val_split)
                                val_pert_indices = train_val_pert_indices[:n_val]
                                train_pert_indices = train_val_pert_indices[n_val:]
                                
                                # Create train subset
                                if len(train_pert_indices) > 0:
                                    train_subset = ds.to_subset_dataset(
                                        "train", train_pert_indices, train_controls)
                                    self.train_datasets.append(train_subset)
                                
                                # Create val subset
                                if len(val_pert_indices) > 0:
                                    val_subset = ds.to_subset_dataset(
                                        "val", val_pert_indices, train_controls)
                                    self.val_datasets.append(val_subset)

                            # Create test subset
                            if len(test_pert_indices) > 0:
                                test_subset = ds.to_subset_dataset(
                                    "test", test_pert_indices, test_controls)
                                self.test_datasets.append(test_subset)
                                
                        else:
                            # Regular training cell type - split into train/val
                            splits = ds.prepare_training_splits(self.val_split, self.rng)
                            
                            if len(splits["train"]["perturbed"]) > 0:
                                train_subset = ds.to_subset_dataset(
                                    "train", 
                                    splits["train"]["perturbed"],
                                    splits["train"]["control"]
                                )
                                self.train_datasets.append(train_subset)
                                
                            if len(splits["val"]["perturbed"]) > 0:
                                val_subset = ds.to_subset_dataset(
                                    "val",
                                    splits["val"]["perturbed"],
                                    splits["val"]["control"]
                                )
                                self.val_datasets.append(val_subset)

    def _setup_global_fewshot_splits(self):
        """
        Do a global scan across files to decide train/test perturbation splits for each fewshot cell type.
        """
        self.fewshot_splits = {}  # {cell_type: {"train_perts": set(), "test_perts": set()}}
        
        # Find all fewshot specs
        fewshot_specs = [s for s in self.test_specs if s.task_type == TaskType.FEWSHOT]
        
        for spec in fewshot_specs:
            ct = spec.cell_type
            # Aggregate counts across all files
            pert_counts = defaultdict(int)
            
            files = self._find_dataset_files(spec.dataset)
            for fpath in files:
                fpath = files[fpath]
                with h5py.File(fpath, "r") as f:
                    # Get cell type codes
                    ct_code = np.where(safe_decode_array(f[f"obs/{self.cell_type_key}/categories"][:]) == ct)[0][0]
                    ct_mask = f[f"obs/{self.cell_type_key}/codes"][:] == ct_code
                    
                    # Get perturbation names for this cell type's cells
                    pert_codes = f[f"obs/{self.pert_col}/codes"][:][ct_mask]
                    pert_categories = safe_decode_array(f[f"obs/{self.pert_col}/categories"][:])
                    pert_names = pert_categories[pert_codes]
                    
                    # Count non-control perturbations
                    for pert in np.unique(pert_names):
                        if pert != self.control_pert:
                            pert_counts[pert] += np.sum(pert_names == pert)

            # Split perturbations greedily based on counts
            sorted_perts = sorted([(p, c) for p, c in pert_counts.items()], 
                                key=lambda x: x[1], reverse=True)
            
            total_cells = sum(c for _, c in sorted_perts)
            target_train = total_cells * self.few_shot_percent
            
            train_perts = set()
            test_perts = set()
            current_train = 0
            
            for pert, count in sorted_perts:
                if abs((current_train + count) - target_train) < abs(current_train - target_train):
                    train_perts.add(pert)
                    current_train += count
                else:
                    test_perts.add(pert)

            self.fewshot_splits[ct] = {
                "train_perts": train_perts,
                "test_perts": test_perts
            }
            logger.info(f"Cell type {ct}: {len(train_perts)} train perts, {len(test_perts)} test perts")

    def setup(self, stage: Optional[str] = None):
        """
        Set up training and test datasets.
        """
        if len(self.train_datasets) == 0:
            logger.info("Setting up training and test datasets...")
            self._setup_training_datasets()
            logger.info(
                "Done! Train / Val / Test splits: %d / %d / %d",
                len(self.train_datasets),
                len(self.val_datasets),
                len(self.test_datasets)
            )

    def train_dataloader(self):
        if len(self.train_datasets) == 0:
            return None
        collate_fn = lambda batch: PerturbationDataset.collate_fn(batch, transform=self.transform)
        ds = MetadataConcatDataset(self.train_datasets)
        sampler = PerturbationBatchSampler(dataset=ds, batch_size=self.batch_size, drop_last=False)
        return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=collate_fn)

    def train_eval_dataloader(self):
        if len(self.train_eval_datasets) == 0:
            return None
        collate_fn = lambda batch: PerturbationDataset.collate_fn(batch, transform=self.transform)
        ds = MetadataConcatDataset(self.train_eval_datasets)
        sampler = PerturbationBatchSampler(dataset=ds, batch_size=self.batch_size, drop_last=False)
        return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        if len(self.val_datasets) == 0:
            return None
        collate_fn = lambda batch: PerturbationDataset.collate_fn(batch, transform=self.transform)
        ds = MetadataConcatDataset(self.val_datasets)
        sampler = PerturbationBatchSampler(dataset=ds, batch_size=self.batch_size, drop_last=False)
        return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        if len(self.test_datasets) == 0:
            return None
        collate_fn = lambda batch: PerturbationDataset.collate_fn(batch, transform=self.transform)
        ds = MetadataConcatDataset(self.test_datasets)
        sampler = PerturbationBatchSampler(dataset=ds, batch_size=self.batch_size, drop_last=False)
        return DataLoader(ds, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=collate_fn)

    def predict_dataloader(self):
        return self.test_dataloader()

    def set_inference_mapping_strategy(self, strategy_cls, **strategy_kwargs):
        """
        If user picks 'PseudoNearestMappingStrategy', we ensure we have the
        global offsets. If not, we compute them. Then we create an instance
        of that strategy and call each test dataset's reset_mapping_strategy.
        """

        if strategy_cls.__name__ == "PseudoNearestMappingStrategy":
            # compute offsets
            use_gene_space = self.output_space == "gene"

            logger.info("Offsets not found, computing them for pseudo_nearest strategy...")
            self._compute_pert_mean_offsets_for_inference(use_gene_space=use_gene_space, embed_key=self.embed_key)
            # re-insert the 'use_gene_space' key if needed
            strategy_kwargs["use_gene_space"] = use_gene_space

            # Now we instantiate the new strategy
            logger.info("Instantiating PseudoNearestMappingStrategy with precomputed offsets...")
            strategy_kwargs["pert_mean_offsets"] = self._pseudo_nearest_pert_offsets
            strategy_kwargs["global_basal"] = self._pseudo_nearest_global_basal

        # normal usage for e.g. NearestNeighborMappingStrategy, etc.
        self.mapping_strategy_cls = strategy_cls
        self.mapping_strategy_kwargs = strategy_kwargs

        for ds_subset in self.test_datasets:
            ds: PerturbationDataset = ds_subset.dataset
            ds.reset_mapping_strategy(strategy_cls, stage="inference", **strategy_kwargs)

        logger.info("Mapping strategy set to %s for test datasets.", strategy_cls.__name__)

    def get_var_dims(self):
        underlying_ds: PerturbationDataset = self.test_datasets[0].dataset
        if self.embed_key:
            # if self.transform:  # PCA transform, todo change this.
            #     input_dim = self.transform.n_components  # data is processed on the fly here
            # else:
            #     # TODO- if we peek into the files we can get dimensions before having to call setup on the datamodule
            input_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
        else:
            input_dim = underlying_ds.n_genes

        if self.output_space == "gene":
            output_dim = underlying_ds.n_genes
        else:
            output_dim = underlying_ds.get_dim_for_obsm(self.embed_key)

        gene_names = underlying_ds.get_gene_names()

        return {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "pert_dim": len(self.pert_onehot_map),
            "gene_names": gene_names,
        }

    def get_shared_perturbations(self) -> Set[str]:
        """
        Compute shared perturbations between train and test sets by inspecting
        only the actual subset indices in self.train_datasets and self.test_datasets.

        This ensures we don't accidentally include all perturbations from the entire h5 file.
        """

        def _extract_perts_from_subset(subset) -> Set[str]:
            """
            Helper that returns the set of perturbation names for the
            exact subset indices in 'subset'.
            """
            ds = subset.dataset  # The underlying PerturbationDataset
            idxs = subset.indices  # The subset of row indices relevant to this Subset

            # ds.pert_col typically is 'gene' or similar
            pert_codes = ds.h5_file[f"obs/{ds.pert_col}/codes"][sorted(idxs)]
            # Convert each code to its corresponding string label
            pert_names = ds.pert_categories[pert_codes]

            return set(pert_names)

        # 1) Gather all perturbations found across the *actual training subsets*
        train_perts = set()
        for subset in self.train_datasets:
            train_perts.update(_extract_perts_from_subset(subset))

        # 2) Gather all perturbations found across the *actual testing subsets*
        test_perts = set()
        for subset in self.test_datasets:
            test_perts.update(_extract_perts_from_subset(subset))

        # 3) Intersection = shared across both train and test
        shared_perts = train_perts & test_perts

        logger.info(f"Found {len(train_perts)} distinct perts in the train subsets.")
        logger.info(f"Found {len(test_perts)} distinct perts in the test subsets.")
        logger.info(f"Found {len(shared_perts)} shared perturbations (train ∩ test).")

        return shared_perts

    
    def get_control_pert(self):
        # Return the control perturbation name
        return self.train_datasets[0].dataset.control_pert

    def _compute_pert_mean_offsets_for_inference(self, use_gene_space: bool = False, embed_key: str = "X_uce"):
        """
        This replicates the logic from GlobalSimpleSum to compute a single
        global_basal and per-pert offset across all training data subsets.
        We'll do a single pass over the training subsets, accumulate sums
        and counts, then store them in self._pseudo_nearest_global_basal
        and self._pseudo_nearest_pert_offsets.

        If use_gene_space=True, we fetch gene expression; else we fetch embed_key (UCE, scGPT, etc).
        """
        logger.info("Computing global offset for 'pseudo_nearest' inference...")

        # Sums for all control cells
        global_ctrl_sum = None
        global_ctrl_count = 0

        # Sums for each perturbation
        pert_sum = defaultdict(lambda: None)
        pert_count = defaultdict(int)

        # We iterate over each dataset subset in our training set
        train_loader = self.train_dataloader()
        if train_loader is None:
            logger.warning("No train dataloader found. Cannot compute offsets.")
            return

        for subset_ds in train_loader.dataset.datasets:
            ds = subset_ds.dataset
            indices = subset_ds.indices

            # We loop over the actual indices in that subset
            for idx in indices:
                # 1) fetch embedding or gene expression
                if use_gene_space:
                    arr = ds.fetch_gene_expression(idx).cpu().numpy()
                else:
                    arr = ds.fetch_obsm_expression(idx, embed_key).cpu().numpy()

                # 2) see if it is control or not
                is_ctrl = ds.control_mask[idx]
                # get perturbation name
                code = ds.h5_file[f"obs/{ds.pert_col}/codes"][idx]
                p_name = ds.pert_categories[code]

                if is_ctrl:
                    # accumulate in global_ctrl_sum
                    if global_ctrl_sum is None:
                        global_ctrl_sum = np.zeros_like(arr)
                    global_ctrl_sum += arr
                    global_ctrl_count += 1
                else:
                    # accumulate in pert_sum
                    if pert_sum[p_name] is None:
                        pert_sum[p_name] = np.zeros_like(arr)
                    pert_sum[p_name] += arr
                    pert_count[p_name] += 1

        # Now compute global mean
        if global_ctrl_count < 1:
            logger.warning("No control cells in training?? Using zero vector as basal.")
            self._pseudo_nearest_global_basal = None
            return

        global_basal = global_ctrl_sum / float(global_ctrl_count)

        # compute offsets for each pert
        offsets = {}
        for p_name, sum_arr in pert_sum.items():
            c = pert_count[p_name]
            if c < 1:
                offsets[p_name] = np.zeros_like(global_basal)
            else:
                p_mean = sum_arr / float(c)
                offsets[p_name] = p_mean - global_basal

        # store in self
        self._pseudo_nearest_global_basal = global_basal
        self._pseudo_nearest_pert_offsets = offsets
        logger.info(
            "Done. Stored global basal shape=%s, #offsets=%d",
            global_basal.shape if global_basal is not None else None,
            len(offsets),
        )

    def _setup_global_celltype_map(self):
        """
        Create a global cell type map across all H5 files in train+test specs,
        so that each dataset can use a consistent one-hot scheme.
        """
        dataset_files = {spec.dataset for spec in (self.train_specs + self.test_specs)}
        # For each dataset, gather all .h5 files and read out the pert categories
        all_celltypes = set()
        for ds_name in dataset_files:
            files_dict = self._find_dataset_files(ds_name)
            for file_path in files_dict.values():
                with h5py.File(file_path, "r") as f:
                    # cats = f["obs/cell_type/categories"][:].astype(str)
                    cats = [x.decode('utf-8') for x in f["obs/cell_name/categories"][:]]
                    all_celltypes.update(cats)

        if len(all_celltypes) == 0:
            raise ValueError("No cell types found across datasets?")

        self.celltype_onehot_map = generate_onehot_map(all_celltypes)
        self.num_celltypes = len(self.celltype_onehot_map)

    def _setup_global_pert_map(self):
        """
        Create a global perturbation map across all H5 files in train+test specs,
        so that each dataset can use a consistent one-hot scheme.
        """
        dataset_files = {spec.dataset for spec in (self.train_specs + self.test_specs)}
        # For each dataset, gather all .h5 files and read out the pert categories
        all_perts = set()
        for ds_name in dataset_files:
            files_dict = self._find_dataset_files(ds_name)
            for file_path in files_dict.values():
                with h5py.File(file_path, "r") as f:
                    # cats = f["obs/drug/categories"][:].astype(str)
                    cats = [x.decode('utf-8') for x in f["obs/drug/categories"][:]]
                    all_perts.update(cats)

        if len(all_perts) == 0:
            raise ValueError("No perturbations found across datasets?")

        self.pert_onehot_map = generate_onehot_map(all_perts)
        self.num_perts = len(self.pert_onehot_map)

    def _setup_global_batch_map(self):
        """
        Create a global gem_group / batch map across all H5 files in train+test specs,
        so that each dataset can use a consistent one-hot scheme.
        """
        dataset_files = {spec.dataset for spec in (self.train_specs + self.test_specs)}
        # For each dataset, gather all .h5 files and read out the pert categories
        all_batches = set()
        for ds_name in dataset_files:
            files_dict = self._find_dataset_files(ds_name)
            for file_path in files_dict.values():
                with h5py.File(file_path, "r") as f:
                    try:
                        # cats = f["obs/gem_group/categories"][:].astype(str)
                        cats = [x.decode('utf-8') for x in f["obs/drug/categories"][:]]
                    except KeyError:
                        cats = f["obs/gem_group"][:].astype(str)
                    all_batches.update(cats)

        if len(all_batches) == 0:
            raise ValueError("No perturbations found across datasets?")

        self.batch_onehot_map = generate_onehot_map(all_batches)
        self.num_batches = len(self.batch_onehot_map)

    ############################
    # ADDITIONAL HELPERS
    ############################
    ############################
    # FILE DISCOVERY + HELPERS
    ############################
    def _find_dataset_files(self, dataset_name: str) -> Dict[str, Path]:
        """
        Locate *.h5 files for a given dataset_name inside self.data_dir.
        Return a dict: {cell_type -> file_path}.

        """
        pattern = "*.h5"

        # TODO-Abhi: currently using a debug prefix to use unmerged replogle
        # datasets for testing, change this.
        working_folder = self.data_dir / f"{dataset_name}"
        if not working_folder.exists():
            raise FileNotFoundError(f"No directory named {working_folder}")

        files = sorted(working_folder.glob(pattern))
        cell_types = {}
        for fpath in files:
            # Typically naming: <cell_type>.h5
            cell_type = fpath.stem.split(".")[0]
            cell_types[cell_type] = fpath
        return cell_types

    def _group_specs_by_dataset(self, specs: List[TaskSpec]) -> Dict[str, List[TaskSpec]]:
        """
        Group a list of TaskSpecs by dataset name.
        Return dict: {dataset_name -> [TaskSpec, TaskSpec, ...]}
        """
        dmap = {}
        for spec in specs:
            dmap.setdefault(spec.dataset, []).append(spec)
        return dmap

    def _get_test_cell_types(self, dataset: str, test_map: Dict[str, List[TaskSpec]]) -> Set[str]:
        """
        Identify cell types used for testing in the given dataset.
        So we can skip them for training, in the zero-shot approach.
        """
        test_cell_types = set()
        if dataset in test_map:
            for spec in test_map[dataset]:
                if spec.task_type in (TaskType.ZEROSHOT, TaskType.FEWSHOT):
                    test_cell_types.add(spec.cell_type)
        return test_cell_types

    def _get_training_cell_types(
        self, specs: List[TaskSpec], files_dict: Dict[str, Path], test_cts: Set[str]
    ) -> Set[str]:
        """
        Determine which cell types are used for training. If a spec has a cell_type
        of None, we use all available (except those in 'test_cts').
        """
        all_cts = set(files_dict.keys())
        train_cts = set()
        for s in specs:
            if s.cell_type is None:
                # means "train on all cts except the test ones"
                train_cts.update(all_cts - test_cts)
            else:
                if s.cell_type not in files_dict:
                    raise ValueError(f"Cell type {s.cell_type} not found in dataset files.")
                # skip if it's in test cts
                if s.cell_type in test_cts:
                    logger.warning(
                        f"Spec says train on cell_type {s.cell_type}, but it is also a test cell type. Skipping."
                    )
                else:
                    train_cts.add(s.cell_type)
        return train_cts

# A small helper to decode arrays (so we can reuse it in this module if needed)
def safe_decode_array(arr):
    try:
        return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr])
    except Exception:
        return np.array([str(x) for x in arr])