import numpy as np
import torch
from collections import defaultdict
from data.utils.data_utils import H5MetadataCache
from typing import Dict, List, Tuple, Iterator
from torch.utils.data import Sampler

import logging
import time
from tqdm import tqdm  # progress bar

logger = logging.getLogger(__name__)

class PerturbationBatchSampler(Sampler):
    """Samples batches ensuring cells in each batch share (cell_type, perturbation)."""
    
    def __init__(self, dataset: "MetadataConcatDataset", batch_size: int, drop_last: bool = False):
        logger.info("Creating perturbation batch sampler with metadata caching...")
        start_time = time.time()
        
        self.dataset = dataset.data_source if hasattr(dataset, "data_source") else dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Create caches for all unique H5 files
        self.metadata_caches = {}
        for subset in self.dataset.datasets:
            base_dataset = subset.dataset
            if base_dataset.h5_path not in self.metadata_caches:
                self.metadata_caches[base_dataset.h5_path] = H5MetadataCache(
                    str(base_dataset.h5_path),
                    base_dataset.pert_col,
                    base_dataset.cell_type_key,
                    base_dataset.control_pert
                )
        
        # Create batches
        self.batches = self._create_batches()
        
        end_time = time.time()
        logger.info(
            f"Sampler created with {len(self.batches)} batches in {end_time - start_time:.2f} seconds."
        )

    def _process_subset(self, global_offset: int, subset: "Subset") -> List[List[int]]:
        """Process a single subset to create batches."""
        base_dataset = subset.dataset
        indices = np.array(subset.indices)
        cache = self.metadata_caches[base_dataset.h5_path]
        
        # Get cell types and perturbations
        if base_dataset.filter_cell_type is not None:
            cell_types = np.full(len(indices), base_dataset.filter_cell_type)
            _, pert_names = cache.get_cell_info(indices)
        else:
            cell_types, pert_names = cache.get_cell_info(indices)
        
        # Create global indices
        global_indices = np.arange(global_offset, global_offset + len(indices))
        
        # Create a structured array for efficient grouping
        dt = np.dtype([('cell', cell_types.dtype), ('pert', pert_names.dtype)])
        groups = np.empty(len(indices), dtype=dt)
        groups['cell'] = cell_types
        groups['pert'] = pert_names
        
        # Group by (cell_type, perturbation) using np.unique
        subset_batches = []
        for group_key in np.unique(groups):
            mask = (groups == group_key)
            group_indices = global_indices[mask]
            np.random.shuffle(group_indices)
            
            # Create batches for this group
            for i in range(0, len(group_indices), self.batch_size):
                batch = group_indices[i:i + self.batch_size].tolist()
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                subset_batches.append(batch)
                
        return subset_batches

    def _create_batches(self) -> List[List[int]]:
        """Create batches sequentially across subsets."""
        global_offset = 0
        all_batches = []
        
        for subset in self.dataset.datasets:
            subset_batches = self._process_subset(global_offset, subset)
            all_batches.extend(subset_batches)
            global_offset += len(subset)
            
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        np.random.shuffle(self.batches)
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)