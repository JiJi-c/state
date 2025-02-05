import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Iterator
from torch.utils.data import Sampler

class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that each batch contains cells from the same (cell_type, perturbation) pair.
    This minimizes within-batch variance and helps cells attend to one another.
    
    This sampler works with a MetadataConcatDataset (a concatenation of multiple PerturbationDataset
    subsets). In the new design, each PerturbationDataset may contain multiple cell types.
    If the dataset was created with a filter (via filter_cell_type), then that value is used; otherwise,
    each sample’s cell type is read from the file.
    """
    def __init__(self, dataset: "MetadataConcatDataset", batch_size: int, drop_last: bool = False):
        # If the dataset has a data_source attribute, use it.
        if hasattr(dataset, "data_source"):
            self.dataset = dataset.data_source
        else:
            self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by (cell_type, perturbation)
        self.grouped_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        offset = 0
        # Iterate over each subset in the concatenated dataset.
        for subset in self.dataset.datasets:
            base_dataset = subset.dataset  # This is a PerturbationDataset
            # Loop over the indices in this subset
            for i in range(len(subset)):
                global_idx = offset + i                # Index in the concatenated dataset
                base_idx = subset.indices[i]             # Actual index in the base HDF5 file

                # Determine the cell type for this sample:
                if base_dataset.filter_cell_type is not None:
                    cell_type = base_dataset.filter_cell_type
                else:
                    code = base_dataset.h5_file[f"obs/{base_dataset.cell_type_key}/codes"][base_idx]
                    cell_type = base_dataset.all_cell_types[int(code)]

                # Get the perturbation name using the base index.
                pert_code = base_dataset.h5_file[f"obs/{base_dataset.pert_col}/codes"][base_idx]
                pert_name = base_dataset.pert_categories[int(pert_code)]

                # Group the global index by (cell_type, pert_name)
                self.grouped_indices[(cell_type, pert_name)].append(global_idx)
            offset += len(subset)

        # Calculate total number of batches.
        self.num_batches = sum(
            len(indices) // self.batch_size + (0 if self.drop_last else 1)
            for indices in self.grouped_indices.values()
        )

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each group.
        grouped_indices_lists = {key: indices.copy() for key, indices in self.grouped_indices.items()}
        for indices in grouped_indices_lists.values():
            np.random.shuffle(indices)
        # Yield batches from each group.
        for indices in grouped_indices_lists.values():
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                yield batch_indices

    def __len__(self) -> int:
        return self.num_batches