from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Literal, Set
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from lightning import LightningDataModule
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
from data.transforms.pca import PCATransform  # TODO-Abhi: change to BaseTransform
from data.utils.data_utils import generate_onehot_map
from enum import Enum
from functools import partial

import h5py
import anndata as ad
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    REPLOGLE = "replogle"
    MCFALINE = "mcfaline"
    JIANG = "jiang"
    SCIPLEX = "sciplex"
    FENG = "feng"


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
        embed_key: Literal["X_uce", "X_pca", "X_scGPT"] = "X_uce",
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

        if not self.split_train_val_controls:
            logger.info("NOTE: Control cells will be shared between train and val splits.")

        # Few-shot store: (dataset_name, cell_type) -> dict of splits
        self.fewshot_splits: Dict[(str, str), Dict[str, np.ndarray]] = {}

        # Global perturbation map
        self.all_perts: Set[str] = set()
        self.pert_onehot_map: Optional[Dict[str, torch.Tensor]] = None

        self.batch_onehot_map: Optional[Dict[str, torch.Tensor]] = None

        self.celltype_onehot_map: Optional[Dict[str, torch.Tensor]] = None
        self.num_celltypes: Optional[int] = None

        self.num_genes: Optional[int] = None
        self.num_perts: Optional[int] = None

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

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Builds train / val / test subsets for each dataset and cell type, and stores
        them in the appropriate lists.
        """
        if stage not in ("fit", "test", None):
            return

        logger.info("Starting dataset setup...")

        # Create global perturbation map if needed
        if not self.pert_onehot_map:
            # this should be across test data too
            self._setup_global_pert_map()
            logger.info(f"Created global perturbation map with {self.num_perts} perturbations")

        if not self.batch_onehot_map:
            # this should be across test data too
            self._setup_global_batch_map()
            logger.info(f"Created global batch map with {self.num_batches} batches")

        if not self.celltype_onehot_map:
            # this should be across test data too
            self._setup_global_celltype_map()
            logger.info(f"Created global cell type map with {self.num_celltypes} cell types")

        # Avoid re-creating training datasets if already done
        if (stage == "fit" or stage is None) and len(self.train_datasets) == 0:
            logger.info("Setting up training datasets...")
            self._setup_training_datasets()

            if len(self.train_datasets) > 0:
                if self.transform:
                    all_train_data = []

                    for n, train_subset in enumerate(self.train_datasets):
                        indices = train_subset.indices
                        underlying_ds: PerturbationDataset = train_subset.dataset
                        logger.info(f"Adding data for train ds {n}")
                        for idx in indices:
                            all_train_data.append(underlying_ds.fetch_expression(idx))

                    all_train_data = torch.vstack(all_train_data)
                    logger.info(f"Fitting {self.transform.name()} on training data...")

                    self.num_genes = underlying_ds.n_genes
                    logger.info(f"Set num_genes={self.num_genes}")

                    # Create temporary anndata for fitting
                    self.transform.fit(all_train_data)
                    logger.info("Fitting complete")
                    del all_train_data
            else:
                logger.warning("No training datasets created")

        # Avoid re-creating testing datasets if already done
        if (stage == "test" or stage is None) and len(self.test_datasets) == 0:
            logger.info("Setting up test datasets...")
            self._setup_test_datasets()

            if len(self.test_datasets) == 0:
                logger.warning("No test datasets created")

        logger.info("Dataset setup complete")

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
        if self.embed_key:
            if self.transform:  # PCA transform, todo change this.
                input_dim = self.transform.n_components  # data is processed on the fly here
            else:
                # TODO- if we peek into the files we can get dimensions before having to call setup on the datamodule
                underlying_ds: PerturbationDataset = self.test_datasets[0].dataset
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

    def train_dataloader(self):
        if len(self.train_datasets) == 0:
            return None

        collate_with_transform = partial(PerturbationDataset.collate_fn, transform=self.transform)

        ds = MetadataConcatDataset(self.train_datasets)
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=self.batch_size,
            drop_last=False,  # Drop incomplete batches during training
        )

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_with_transform,
        )

    def train_eval_dataloader(self):
        if len(self.train_eval_datasets) == 0:
            return None

        collate_with_transform = partial(PerturbationDataset.collate_fn, transform=self.transform)

        ds = MetadataConcatDataset(self.train_eval_datasets)
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=self.batch_size,
            drop_last=False,  # Drop incomplete batches during training
        )

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_with_transform,
        )

    def val_dataloader(self):
        if len(self.val_datasets) == 0:
            return None

        collate_with_transform = partial(PerturbationDataset.collate_fn, transform=self.transform)

        ds = MetadataConcatDataset(self.val_datasets)
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=self.batch_size,
            drop_last=False,  # Drop incomplete batches during training
        )

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_with_transform,
        )

    def test_dataloader(self):
        if len(self.test_datasets) == 0:
            return None

        collate_with_transform = partial(PerturbationDataset.collate_fn, transform=self.transform)

        ds = MetadataConcatDataset(self.test_datasets)
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=self.batch_size,
            drop_last=False,  # Drop incomplete batches during training
        )

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_with_transform,
        )

    def get_control_pert(self):
        # Return the control perturbation name
        return self.train_datasets[0].dataset.control_pert

    def predict_dataloader(self):
        # Use test data for predictions
        return self.test_dataloader()

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
                    cats = f["obs/cell_type/categories"][:].astype(str)
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
                    cats = f["obs/gene/categories"][:].astype(str)
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
                        cats = f["obs/gem_group/categories"][:].astype(str)
                    except KeyError:
                        cats = f["obs/gem_group"][:].astype(str)
                    all_batches.update(cats)

        if len(all_batches) == 0:
            raise ValueError("No perturbations found across datasets?")

        self.batch_onehot_map = generate_onehot_map(all_batches)
        self.num_batches = len(self.batch_onehot_map)

    def _setup_training_datasets(self):
        """
        Creates the training & validation splits for each dataset/cell_type
        in self.train_specs, also handling few-shot if needed.
        """
        # Group train/test specs by dataset
        train_map = self._group_specs_by_dataset(self.train_specs)
        test_map = self._group_specs_by_dataset(self.test_specs)

        for dataset_name, specs in train_map.items():
            # 1) load all h5 files for that dataset
            files_dict = self._find_dataset_files(dataset_name)
            # 2) figure out which cts are for testing, so we skip them in training
            test_cell_types = self._get_test_cell_types(dataset_name, test_map)
            # 3) figure out which cts to train on
            training_cell_types = self._get_training_cell_types(specs, files_dict, test_cell_types)
            print(f"Assigning {training_cell_types} for dataset {dataset_name} to train data")
            print(f"Assigning {test_cell_types} for dataset {dataset_name} to test data")

            # Subsampling fraction
            subsample_fraction = self.dataset_subsampling.get(dataset_name, 1.0)
            logger.info(f"Setting up {dataset_name} with {subsample_fraction:.1%} subsampling")

            # Build for each cell type
            for ct in training_cell_types:
                logger.info(f"Adding training cell type: {ct} for dataset {dataset_name}")

                # 1) Create mapping strategy instance
                base_params = {
                    "random_state": self.random_seed,
                    "n_basal_samples": self.n_basal_samples,
                    "k_neighbors": self.k_neighbors,
                    "pca_transform": self.transform,
                    "n_clusters": 50,
                    "neighborhood_fraction": self.neighborhood_fraction,
                }

                strategy_obj = self.mapping_strategy_cls(**base_params)

                # 2) Build PerturbationDataset
                ds_obj = PerturbationDataset(
                    name=f"{dataset_name}",  # or use DatasetType(dataset_name).value
                    h5_path=files_dict[ct],
                    mapping_strategy=strategy_obj,
                    embed_key=self.embed_key,
                    store_raw_expression=self.output_space == "gene",
                    random_state=self.random_seed,
                    pert_onehot_map=self.pert_onehot_map,
                    batch_onehot_map=self.batch_onehot_map,
                    pert_tracker=self.pert_tracker,
                    should_yield_control_cells=self.should_yield_control_cells,
                    split_train_val_controls=self.split_train_val_controls,
                    preload_data=self.preload_data,
                )

                logger.info("\t Cell type {} has {} cells".format(ct, len(ds_obj)))

                # 3) Split train/val (for perturbed)
                splits = ds_obj.prepare_training_splits(self.val_split, self.rng)
                train_pert = splits["train"]["perturbed"]
                train_ctrl = splits["train"]["control"]
                val_pert = splits["val"]["perturbed"]
                val_ctrl = splits["val"]["control"]

                # 4) Create subset for train
                if len(train_pert) > 0:
                    train_subset = ds_obj.to_subset_dataset("train", train_pert, train_ctrl, subsample_fraction)
                    self.train_datasets.append(train_subset)

                    # Also create a "train_eval" subset from 10% of the train pert
                    # TODO-Abhi: reaching into PerturbationDataset internals here, not ideal, fix
                    retained_train_pert = np.array(list(train_subset.dataset.split_perturbed_indices["train"]))
                    if len(retained_train_pert) > 0:
                        rng = np.random.default_rng(self.random_seed)
                        train_eval_size = int(self.few_shot_percent * len(retained_train_pert))
                        if train_eval_size > 0:
                            train_eval_pert = rng.choice(retained_train_pert, size=train_eval_size, replace=False)
                            train_eval_subset = ds_obj.to_subset_dataset(
                                "train_eval", train_eval_pert, train_ctrl, subsample_fraction
                            )
                            self.train_eval_datasets.append(train_eval_subset)

                # 5) Create subset for val
                if len(val_pert) > 0:
                    val_subset = ds_obj.to_subset_dataset("val", val_pert, val_ctrl, subsample_fraction)
                    self.val_datasets.append(val_subset)

            # Also handle few-shot tasks to add to training if specified
            self._setup_fewshot_datasets(dataset_name, files_dict, specs, test_map)

    def _setup_test_datasets(self):
        """
        Creates zero-shot or few-shot test splits. The code checks if the
        dataset/cell_type is in a zero-shot or few-shot task spec, and builds
        a corresponding subset from all perturbed cells.
        """
        test_map = self._group_specs_by_dataset(self.test_specs)
        for dataset_name, specs in test_map.items():
            files_dict = self._find_dataset_files(dataset_name)
            for spec in specs:
                # 1) Build a mapping strategy
                strategy_obj = self.mapping_strategy_cls(
                    random_state=self.random_seed,
                    n_basal_samples=self.n_basal_samples,
                    k_neighbors=self.k_neighbors,
                    neighborhood_fraction=self.neighborhood_fraction,
                )
                # 2) Build dataset
                ds_obj = PerturbationDataset(
                    name=f"{dataset_name}",
                    h5_path=files_dict[spec.cell_type],
                    mapping_strategy=strategy_obj,
                    embed_key=self.embed_key,
                    store_raw_expression=True,
                    random_state=self.random_seed,
                    pert_onehot_map=self.pert_onehot_map,
                    batch_onehot_map=self.batch_onehot_map,
                    pert_tracker=self.pert_tracker,
                    should_yield_control_cells=self.should_yield_control_cells,
                    split_train_val_controls=self.split_train_val_controls,
                    preload_data=self.preload_data,
                )

                if spec.task_type == TaskType.ZEROSHOT:
                    # All controls are accessible to both train and val
                    test_pert = np.where(~ds_obj.control_mask)[0]
                    test_ctrl = np.where(ds_obj.control_mask)[0]

                    if len(test_pert) > 0:
                        test_subset = ds_obj.to_subset_dataset("test", test_pert, test_ctrl)
                        self.test_datasets.append(test_subset)
                elif spec.task_type == TaskType.FEWSHOT:
                    # If a few-shot partition for this dataset/cell_type does not exist, compute it on the fly.
                    if (dataset_name, spec.cell_type) not in self.fewshot_splits:
                        logger.info(f"No few-shot partition found for {dataset_name}/{spec.cell_type} in training. Computing on the fly.")
                        splits = ds_obj.prepare_fewshot_splits(self.few_shot_percent, self.val_split, self.rng)
                        self.fewshot_splits[(dataset_name, spec.cell_type)] = splits

                        train_pert = splits["train"]["perturbed"]
                        train_ctrl = splits["train"]["control"]
                        val_pert = splits["val"]["perturbed"]
                        val_ctrl = splits["val"]["control"]

                        if len(train_pert) > 0:
                            train_subset = ds_obj.to_subset_dataset("train", train_pert, train_ctrl)
                            self.train_datasets.append(train_subset)
                        if len(val_pert) > 0:
                            val_subset = ds_obj.to_subset_dataset("val", val_pert, val_ctrl)
                            self.val_datasets.append(val_subset)
                    else:
                        splits = self.fewshot_splits[(dataset_name, spec.cell_type)]
                    test_pert = splits["test"]["perturbed"]
                    test_ctrl = splits["test"]["control"]
                    if len(test_pert) > 0:
                        test_subset = ds_obj.to_subset_dataset("test", test_pert, test_ctrl)
                        self.test_datasets.append(test_subset)

    ############################
    # ADDITIONAL HELPERS
    ############################
    def _setup_fewshot_datasets(
        self,
        dataset_name: str,
        files_dict: Dict[str, Path],
        train_specs: List[TaskSpec],
        test_map: Dict[str, List[TaskSpec]],
    ):
        """
        If the dataset is used in a few-shot scenario, we create a partial
        training/validation split. Then we add that partial data to our
        overall train_datasets, etc. Also store the test partition for
        eventual few-shot testing.
        """
        # Identify any FEWSHOT specs for this dataset (found in test_map).
        if dataset_name not in test_map:
            return

        # For each spec that is FEWSHOT, we create a mini-dataset for it
        specs_for_fewshot = [s for s in test_map[dataset_name] if s.task_type == TaskType.FEWSHOT]
        if not specs_for_fewshot:
            return

        for spec in specs_for_fewshot:
            ct = spec.cell_type
            logger.info(f"Setting up FEWSHOT for {dataset_name}, cell type={ct}")

            strategy_obj = self.mapping_strategy_cls(
                random_state=self.random_seed,
                n_basal_samples=self.n_basal_samples,
                k_neighbors=self.k_neighbors,
                neighborhood_fraction=self.neighborhood_fraction,
            )
            ds_obj = PerturbationDataset(
                name=f"{dataset_name}",
                h5_path=files_dict[ct],
                mapping_strategy=strategy_obj,
                embed_key=self.embed_key,
                store_raw_expression=self.output_space == "gene",
                random_state=self.random_seed,
                pert_onehot_map=self.pert_onehot_map,
                batch_onehot_map=self.batch_onehot_map,
                pert_tracker=self.pert_tracker,
                should_yield_control_cells=self.should_yield_control_cells,
                split_train_val_controls=self.split_train_val_controls,
                preload_data=self.preload_data,
            )

            logger.info("\t Fewshot cell type {} has {} cells".format(ct, len(ds_obj)))

            # Partition a small portion for train, val, rest is test
            splits = ds_obj.prepare_fewshot_splits(self.few_shot_percent, self.val_split, self.rng)
            self.fewshot_splits[(dataset_name, ct)] = splits
            logger.info("Few-shot partitions computed for: " + str(list(self.fewshot_splits.keys())))

            # Add the few-shot train/val to our train_datasets & val_datasets
            # we do not create a test subset here; that is built in _setup_test_datasets
            train_pert = splits["train"]["perturbed"]
            train_ctrl = splits["train"]["control"]
            val_pert = splits["val"]["perturbed"]
            val_ctrl = splits["val"]["control"]

            if len(train_pert) > 0:
                train_subset = ds_obj.to_subset_dataset("train", train_pert, train_ctrl)
                self.train_datasets.append(train_subset)

                # Also a small portion as "train_eval"
                retained_train_pert = np.array(list(train_subset.dataset.split_perturbed_indices["train"]))
                if len(retained_train_pert) > 0:
                    rng = np.random.default_rng(self.random_seed)
                    train_eval_size = int(self.few_shot_percent * len(retained_train_pert))
                    if train_eval_size > 0:
                        train_eval_pert = rng.choice(retained_train_pert, size=train_eval_size, replace=False)
                        train_eval_subset = ds_obj.to_subset_dataset("train_eval", train_eval_pert, train_ctrl)
                        self.train_eval_datasets.append(train_eval_subset)

            if len(val_pert) > 0:
                val_subset = ds_obj.to_subset_dataset("val", val_pert, val_ctrl)
                self.val_datasets.append(val_subset)

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
