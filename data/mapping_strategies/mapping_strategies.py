from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class BaseMappingStrategy(ABC):
    """
    Abstract base class for mapping a perturbed cell to one or more control cells.
    Each strategy can store internal data structures that assist in retrieving
    control indices for a perturbed cell (e.g. nearest neighbor graphs, OT plans, etc.).
    The main parameters for this class are:
    - random_state: seed for random number generation
    - n_basal_samples: number of control cells to return for each perturbed cell
    """

    def __init__(
        self,
        name=None,
        random_state: int = 42,
        n_basal_samples: int = 1,
        stage: str = "train",
        **kwargs,
    ):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.n_basal_samples = n_basal_samples
        self.name = name
        self.stage = stage
        print(f"Using {self.name} mapping strategy.")

    @abstractmethod
    def register_split_indices(
        self,
        dataset: "PerturbationDataset",  # a reference to the PerturbationDataset
        split: str,
        perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        Called once per split (train/val/test) to initialize or compute
        the mapping information for that split.
        """
        pass

    @abstractmethod
    def get_control_indices(self, dataset: "PerturbationDataset", split: str, perturbed_idx: int) -> np.ndarray:
        """
        Returns the control indices for a given perturbed index in a particular split.
        """
        pass

    def get_mapped_expressions(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Base implementation that handles both control and perturbed cells.

        For control cells:
            - Returns (control_expr, control_expr) where control_expr is that cell's expression
        For perturbed cells:
            - Returns (perturbed_expr, control_expr) using get_control_indices()
        """
        pert_name = dataset.get_perturbation_name(perturbed_idx)
        is_control = (pert_name == dataset.control_pert)

        # Get expression(s) based on embed_key
        if dataset.embed_key:
            if is_control:
                expr = torch.tensor(dataset.fetch_obsm_expression(perturbed_idx, dataset.embed_key))
                return expr, expr  # both X and basal are same control
            else:
                control_indices = self.get_control_indices(dataset, split, perturbed_idx)
                pert_expr = torch.tensor(dataset.fetch_obsm_expression(perturbed_idx, dataset.embed_key))
                if control_indices is None:
                    ctrl_expr = torch.zeros_like(pert_expr) # default to zero vector 
                else:
                    ctrl_expr = dataset.fetch_obsm_expression(control_indices[0], dataset.embed_key) 
                return pert_expr, ctrl_expr
        else:
            if is_control:
                expr = dataset.fetch_gene_expression(perturbed_idx)
                return expr, expr  # both X and basal are same control
            else:
                control_indices = self.get_control_indices(dataset, split, perturbed_idx)
                ctrl_expr = torch.stack([dataset.fetch_gene_expression(idx) for idx in control_indices]).mean(0)
                pert_expr = dataset.fetch_gene_expression(perturbed_idx)
                return pert_expr, ctrl_expr
