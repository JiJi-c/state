import time
import logging

import torch
import lightning as L

from typing import Any


class LogLR(L.Callback):
    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *args,
    ) -> None:
        if trainer.global_rank == 0:
            if trainer.global_step % self.interval == 0 and trainer.logger is not None:
                trainer.logger.log_metrics(
                    { "trainer/learning_rate": pl_module.lr_schedulers().get_last_lr()[0] },
                    step=trainer.global_step,
                )


class ProfilerCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.iterations_count = 0
        self.last_ipm_time = None
        self.ipm_history = []
        self.cfg = cfg

        self.profile_steps = cfg.experiment.profile.profile_steps

    def on_train_batch_start(self, trainer: L.Trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        if batch_idx == self.profile_steps[0]:
            logging.info(f"Starting NSys profiling at step {batch_idx}")
            torch.cuda.nvtx.range_push("VCIProfiledSection")

    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()

        # Calculate batch time
        if self.batch_start_time:
            batch_time = current_time - self.batch_start_time
            self.batch_times.append(batch_time)

        # Track iterations per minute
        self.iterations_count += 1
        if self.last_ipm_time is None:
            self.last_ipm_time = current_time

        time_diff = current_time - self.last_ipm_time
        if time_diff >= 60:
            ipm = (self.iterations_count / time_diff) * 60
            self.ipm_history.append(ipm)
            trainer.logger.log_metrics({"perf/ipm": ipm}, step=trainer.global_step)
            # Reset counters
            self.iterations_count = 0
            self.last_ipm_time = current_time

        if batch_idx == self.profile_steps[1]:
            logging.info(f"Stopping NSys profiling at step {batch_idx}")
            torch.cuda.nvtx.range_pop()
