# Author: marcelroed
import time
import logging

import numpy as np
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
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_rank == 0:
            if trainer.global_step % self.interval == 0 and trainer.logger is not None:
                trainer.logger.log_metrics(
                    {"trainer/learning_rate": pl_module.lr_schedulers().get_last_lr()[0]},
                    step=trainer.global_step,
                )


class PerformanceMonitorCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.iterations_count = 0
        self.last_ipm_time = None
        self.ipm_history = []

    def on_train_batch_start(self, trainer: L.Trainer, *args, **kwargs):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer: L.Trainer, *args, **kwargs):
        current_time = time.time()

        # Calculate batch time
        if self.batch_start_time:
            batch_time = current_time - self.batch_start_time
            self.batch_times.append(batch_time)

        # Track iterations per minute
        self.iterations_count += 1
        if self.last_ipm_time is None:
            self.last_ipm_time = current_time

        # Calculate IPM every 60 seconds
        time_diff = current_time - self.last_ipm_time
        if time_diff >= 60:
            ipm = (self.iterations_count / time_diff) * 60
            self.ipm_history.append(ipm)
            trainer.logger.log_metrics({"perf/ipm": ipm}, step=trainer.global_step)
            # Reset counters
            self.iterations_count = 0
            self.last_ipm_time = current_time

