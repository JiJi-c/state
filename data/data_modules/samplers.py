import numpy as np
import torch
from data.dataset.perturbation_dataset import PerturbationDataset
from data.utils.data_utils import H5MetadataCache
from typing import List, Iterator
from torch.utils.data import Sampler
import logging
import time

logger = logging.getLogger(__name__)

class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that cells in each batch share the same 
    (cell_type, perturbation) combination, using only H5 codes.
    
    Instead of grouping by cell type and perturbation names, this sampler
    groups based on integer codes stored in the H5 file (e.g. `cell_type_codes`
    and `pert_codes` in the H5MetadataCache). This avoids repeated string operations.
    """
    
    def __init__(self, dataset: "MetadataConcatDataset", batch_size: int, drop_last: bool = False):
        logger.info("Creating perturbation batch sampler with metadata caching (using codes)...")
        start_time = time.time()

        # If the provided dataset has a `.data_source` attribute, use that.
        self.dataset = dataset.data_source if hasattr(dataset, "data_source") else dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Create caches for all unique H5 files.
        self.metadata_caches = {}
        for subset in self.dataset.datasets:
            base_dataset: PerturbationDataset = subset.dataset
            self.metadata_caches[base_dataset.h5_path] = base_dataset.metadata_cache
        
        # Create batches using the code-based grouping.
        self.batches = self._create_batches()
        
        end_time = time.time()
        logger.info(
            f"Sampler created with {len(self.batches)} batches in {end_time - start_time:.2f} seconds."
        )

    def _process_subset(self, global_offset: int, subset: "Subset") -> List[List[int]]:
        """
        Process a single subset to create batches based on H5 codes.
        
        For each subset, the method:
          - Retrieves the subset indices.
          - Extracts the corresponding cell type and perturbation codes from the cache.
          - Constructs a structured array with two fields (cell, pert) so that unique
            (cell_type, perturbation) pairs can be identified using np.unique.
          - For each unique pair, shuffles the indices and splits them into batches.
        """
        base_dataset = subset.dataset
        indices = np.array(subset.indices)
        cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]
        
        # Use codes directly rather than names.
        cell_codes = cache.cell_type_codes[indices]
        pert_codes = cache.pert_codes[indices]
        
        # Create global indices (assuming that indices in each subset refer to a global concatenation).
        global_indices = np.arange(global_offset, global_offset + len(indices))
        
        # Build a structured array for grouping.
        dt = np.dtype([('cell', cell_codes.dtype), ('pert', pert_codes.dtype)])
        groups = np.empty(len(indices), dtype=dt)
        groups['cell'] = cell_codes
        groups['pert'] = pert_codes
        
        subset_batches = []
        # Group by unique (cell, pert) pairs.
        for group_key in np.unique(groups):
            mask = (groups == group_key)
            group_indices = global_indices[mask]
            np.random.shuffle(group_indices)
            
            # Split the group indices into batches.
            for i in range(0, len(group_indices), self.batch_size):
                batch = group_indices[i:i + self.batch_size].tolist()
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                subset_batches.append(batch)
                
        return subset_batches

    def _create_batches(self) -> List[List[int]]:
        """
        Process each subset sequentially (across all datasets) and combine the batches.
        """
        global_offset = 0
        all_batches = []
        for subset in self.dataset.datasets:
            subset_batches = self._process_subset(global_offset, subset)
            all_batches.extend(subset_batches)
            global_offset += len(subset)
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle the order of batches each time we iterate.
        np.random.shuffle(self.batches)
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)