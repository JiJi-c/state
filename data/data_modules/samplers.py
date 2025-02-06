import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Iterator
from torch.utils.data import Sampler

import logging
import time
from tqdm import tqdm  # progress bar

logger = logging.getLogger(__name__)

class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that each batch contains cells from the same (cell_type, perturbation) pair.
    
    This revised version uses vectorized file I/O per dataset subset and includes progress bars (via tqdm)
    for both the per-subset loop and the inner loop over unique groups. Note that batches are computed
    on a per-file (subset) basis.
    """
    def __init__(self, dataset: "MetadataConcatDataset", batch_size: int, drop_last: bool = False):
        logger.info("Creating a fast perturbation batch sampler (per-file grouping)...")
        start_time = time.time()
        # Use the underlying dataset (if wrapped in a data_source, use that)
        if hasattr(dataset, "data_source"):
            self.dataset = dataset.data_source
        else:
            self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # List to store batches (each batch is a list of global indices)
        self.batches = []
        global_offset = 0  # will hold the offset of indices across subsets

        # Process each subset (each typically corresponds to one file)
        for subset in self.dataset.datasets:
            base_dataset = subset.dataset  # the underlying PerturbationDataset
            # Get the indices for this subset as a NumPy array (indices relative to the file)
            indices = np.array(subset.indices)
            n_samples = len(indices)
            logger.info("Processing subset %s with %d samples...", base_dataset.h5_file.filename, n_samples)
            # Sort the indices (required for safe h5py indexing)
            sort_start = time.time()
            sort_order = np.argsort(indices)
            sorted_indices = indices[sort_order]
            sort_end = time.time()
            logger.info("Sorted indices for subset in %.2f seconds.", sort_end - sort_start)

            # Compute global indices for these samples (and reorder accordingly)
            global_indices = np.arange(global_offset, global_offset + n_samples)[sort_order]
            global_offset += n_samples

            # Obtain cell type for these indices
            ct_start = time.time()
            if base_dataset.filter_cell_type is not None:
                # If a fixed cell type is given, use that for all samples
                cell_types = np.full(n_samples, base_dataset.filter_cell_type)
            else:
                # Vectorized reading from h5py: get all codes at once
                ct_codes = base_dataset.h5_file[f"obs/{base_dataset.cell_type_key}/codes"][sorted_indices]
                cell_types = base_dataset.all_cell_types[ct_codes.astype(int)]
            ct_end = time.time()
            logger.info("Obtained cell types for subset in %.2f seconds.", ct_end - ct_start)
            
            # Obtain perturbation (drug) codes likewise
            pert_start = time.time()
            pert_codes = base_dataset.h5_file[f"obs/{base_dataset.pert_col}/codes"][sorted_indices]
            pert_names = base_dataset.pert_categories[pert_codes.astype(int)]
            pert_end = time.time()
            logger.info("Obtained perturbations for subset in %.2f seconds.", pert_end - pert_start)
            
            # Build a structured array so we can group by (cell_type, pert)
            dt = np.dtype([("cell", cell_types.dtype), ("pert", pert_names.dtype)])
            keys = np.empty(n_samples, dtype=dt)
            keys["cell"] = cell_types
            keys["pert"] = pert_names
            
            # Use np.unique with return_inverse to group indices per (cell_type, pert)
            group_start = time.time()
            uniq_keys, inverse = np.unique(keys, return_inverse=True)
            # Loop over unique groups
            for i, uniq_key in enumerate(uniq_keys):
                # Get global indices for samples in this group
                group_global_indices = global_indices[inverse == i]
                # Shuffle the indices within the group
                group_global_indices = group_global_indices.copy()
                np.random.shuffle(group_global_indices)
                # Now form batches from this group
                n_in_group = len(group_global_indices)
                for j in range(0, n_in_group, self.batch_size):
                    batch = group_global_indices[j : j + self.batch_size].tolist()
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    self.batches.append(batch)
            group_end = time.time()
            logger.info("Grouped indices for subset in %.2f seconds.", group_end - group_start)
        
        end_time = time.time()
        logger.info("Fast sampler created with %d batches in %.2f seconds.", len(self.batches), end_time - start_time)

    def __iter__(self) -> Iterator[List[int]]:
        # Optionally shuffle the list of batches at the beginning of each epoch.
        np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)