import time
import logging
from contextlib import contextmanager
from numba import cuda

from .uce_utils import UCEGenePredictor


@contextmanager
def time_it(timer_name: str):
    logging.debug(f"Starting timer {timer_name}")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.debug(f"Elapsed time {timer_name}: {elapsed_time:.4f} seconds")


def is_gpu_available():
    """
    Check if RAPIDS is available, if available return number of GPUs otherwise return 0
    """
    try:
        num_gpus = cuda.get_num_gpus()
        return num_gpus
    except:
        return 0
