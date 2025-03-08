from typing import Dict, List, Optional, Set
import numpy as np
import logging

from .mapping_strategies import BaseMappingStrategy


class BatchMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control(s) drawn from the same batch (or else
    the same cell type if no controls are available in the batch).

    This mapping strategy is with respect to a specific cell type in a specific dataset.
    """

    def __init__(self, name="batch", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples, **kwargs)
        # A dict for each split: { split_name: { batch_name: [ctrl_indices] } }
        self.split_control_maps = {"train": {}, "train_eval": {}, "val": {}, "test": {}}

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        _perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        Build a map from batch -> control_indices for each split.
        """
        # We'll store which controls are valid for each batch in that split
        for idx in control_indices:
            batch = dataset.get_batch(idx)
            if batch not in self.split_control_maps[split]:
                self.split_control_maps[split][batch] = []
            self.split_control_maps[split][batch].append(idx)

    def get_control_indices(self, dataset: "PerturbationDataset", split: str, perturbed_idx: int) -> np.ndarray:
        """
        If the batch has no controls, fallback to all control cells in the split.

        Returns `n_basal_samples` control indices for perturbed cell at `perturbed_idx`.
        """
        # TODO-Abhi find a way to cyclic import PerturbationDataset type hint
        batch = dataset.get_batch(perturbed_idx)

        # Attempt to use same-batch control
        batch_pool = self.split_control_maps[split].get(batch, [])
        # Filter them down to ones that are actually in the split's control set
        if not batch_pool:
            # fallback: gather all controls from this split across all batches
            batch_pool = [idx for subdict in self.split_control_maps[split].values() for idx in subdict]

        if not batch_pool:
            raise ValueError("No control cells found in BatchMappingStrategy.")

        # sample n_basal_samples from the pool
        return self.rng.choice(batch_pool, size=self.n_basal_samples, replace=True)
