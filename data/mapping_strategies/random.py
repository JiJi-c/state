import numpy as np
from .mapping_strategies import BaseMappingStrategy

class RandomMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control cell(s) drawn from the same plate.
    We ensure that only control cells with the same cell type
    as the perturbed cell are considered.
    """

    def __init__(self, name="random", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples)
        # Instead of a flat list, we use a dict: for each split, map cell type -> list of control indices.
        self.split_control_pool = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }

    def register_split_indices(self, dataset: "PerturbationDataset", split: str,
                                 perturbed_indices: np.ndarray, control_indices: np.ndarray):
        """
        For the given split, group all control indices by their cell type.
        We assume that if a filter is provided in the dataset then all indices belong to the same cell type;
        but if no filter was applied, then this grouping is necessary.
        """
        pool_by_cell = {}
        for idx in control_indices:
            ct = dataset.get_cell_type(idx)
            if ct not in pool_by_cell:
                pool_by_cell[ct] = []
            pool_by_cell[ct].append(idx)
        self.split_control_pool[split] = pool_by_cell

    def get_control_indices(self, dataset: "PerturbationDataset", split: str, perturbed_idx: int) -> np.ndarray:
        """
        Returns n_basal_samples control indices that are from the same cell type as the perturbed cell.
        """
        # Get the cell type of the perturbed cell.
        pert_cell_type = dataset.get_cell_type(perturbed_idx)
        pool = self.split_control_pool[split].get(pert_cell_type, None)
        if pool is None or len(pool) == 0:
            raise ValueError(f"No control cells available in split '{split}' for cell type '{pert_cell_type}'.")
        return self.rng.choice(pool, size=self.n_basal_samples, replace=True)