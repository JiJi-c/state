import numpy as np
from .mapping_strategies import BaseMappingStrategy

class RandomMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control cell(s) drawn from the same plate.
    It assumes that the underlying PerturbationDataset has been filtered (via filter_cell_type)
    so that all cells are of the same cell type.
    """

    def __init__(self, name="random", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples)
        # For each split, store the control indices.
        self.split_control_pool = {"train": [], "train_eval": [], "val": [], "test": []}

    def register_split_indices(self, dataset: "PerturbationDataset", split: str,
                                 perturbed_indices: np.ndarray, control_indices: np.ndarray):
        """
        Simply store all control indices for each split.
        We add an assertion that the control cells all have the same cell type as the filtered cell type.
        """
        self.split_control_pool[split] = control_indices

    def get_control_indices(self, dataset: "PerturbationDataset", split: str, perturbed_idx: int) -> np.ndarray:
        """
        Sample n_basal_samples control indices from the pool for this split.
        """
        pool = self.split_control_pool[split]
        if len(pool) == 0:
            raise ValueError("No control cells available in RandomMappingStrategy for this split.")
        # For robustness, one could check that the cell type of the perturbed cell matches that of the pool.
        pert_cell_type = dataset.get_cell_type(perturbed_idx)
        # (Optional) check first cell in pool
        ctrl_cell_type = dataset.get_cell_type(pool[0])
        assert pert_cell_type == ctrl_cell_type, "Perturbed cell and control pool have different cell types."
        return self.rng.choice(pool, size=self.n_basal_samples, replace=True)