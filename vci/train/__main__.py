import os
import sys
import hydra
import logging

from omegaconf import DictConfig

sys.path.append('../../')
import vci.train.trainer as train

log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="defaults")
def main(cfg: DictConfig):
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
    main()
