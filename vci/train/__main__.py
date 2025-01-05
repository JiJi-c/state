import os
import sys
import logging
import argparse

from pathlib import Path
from hydra import compose, initialize

sys.path.append('../../')
import vci.train.trainer as train

log = logging.getLogger(__name__)


def main(config_file):
    config_file = Path(config_file)
    config_path = os.path.relpath(Path(config_file).parent, Path(__file__).parent)
    with initialize(version_base=None, config_path=config_path):
        log.info(f'Loading config {config_file}...')
        cfg = compose(config_name=config_file.name)

    os.environ['MASTER_ADDR'] = cfg.experiment.master
    os.environ['MASTER_PORT'] = str(cfg.experiment.port)

    # WAR: Workaround for sbatch failing when --ntasks-per-node is set.
    # lightning expects this to be set.
    os.environ['SLURM_NTASKS_PER_NODE'] = str(cfg.experiment.num_gpus_per_node)

    # TODO: Not sure why this is set. Delete if not necessary.
    os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "10"

    log.info(f'*************** Training {cfg.experiment.name} ***************')
    log.info(cfg)

    train.main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create dataset list CSV file"
    )
    parser.add_argument(
        '-c', "--config",
        type=str,
        help="Training configuration file.",
    )
    args = parser.parse_args()

    main(args.config)
