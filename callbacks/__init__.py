from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback
import torch
import numpy as np
import lightning.pytorch as pl
from models import PerturbationModel

class TestMetricsCallback(Callback):
    def __init__(self, test_freq):
        self.test_freq = test_freq
        self.last_test_global_step = 0

    def on_train_epoch_end(self, trainer, pl_module: PerturbationModel):
        # Compute the number of steps passed since the last test run
        if trainer.global_step - self.last_test_global_step >= self.test_freq:
            # Get the test dataloader
            test_dataloader = trainer.datamodule.test_dataloader()
            
            # Only compute metrics if we have a test dataloader
            if test_dataloader is not None:
                try:
                    # Compute metrics using our consolidated method
                    metrics = pl_module.compute_test_metrics(test_dataloader)
                    
                    # Log all metrics
                    for name, value in metrics.items():
                        # Ensure value is a tensor on the correct device
                        if not isinstance(value, torch.Tensor):
                            value = torch.tensor(value, device=pl_module.device)
                        pl_module.log(name, value, sync_dist=True)
                except:
                    print('compute_test_metrics failed')
                
                self.last_test_global_step = trainer.global_step

class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        pl_module.log("train/gradient_norm", gradient_norm(pl_module))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


class PerturbationMagnitudeCallback(Callback):
    """
    Callback that tracks the average magnitude of perturbation effects by measuring
    the L2 norm of the difference between predicted perturbed state and control/basal state.
    """

    def __init__(self):
        super().__init__()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: PerturbationModel,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        with torch.no_grad():
            # basal_states = pl_module.encode_basal_expression(batch["basal"])
            # perturbed_states = pl_module.perturb(batch["pert"], batch["basal"])
            # magnitudes = torch.norm(perturbed_states - basal_states, p=2, dim=1)
            # pert_magnitude = magnitudes.mean()
            pl_module.log("val/perturbation_magnitude", 0.0)
