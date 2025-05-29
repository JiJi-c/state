import time
import logging
from contextlib import contextmanager


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
