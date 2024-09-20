import os
import glob

from pathlib import Path


def get_latest_checkpoint(cfg):
    run_name = "exp_{0}_layers_{1}_dmodel_{2}_samples_{3}_max_lr_{4}_op_dim_{5}".format(
        cfg.experiment.name,
        cfg.model.nlayers,
        cfg.model.emsize,
        cfg.dataset.pad_length,
        cfg.optimizer.max_lr,
        cfg.model.output_dim)

    chk_dir = os.path.join(cfg.experiment.checkpoint.path,
                           cfg.experiment.name)
    chks = glob.glob(os.path.join(chk_dir, f'{run_name}*'))

    chk = None
    if chks:
        chk = sorted(chks)[-1]

    return run_name, chk

def parse_chk_info(chk):
    chk_filename = Path(chk)
    epoch = chk_filename.stem.split('_')[-1].split('-')[1].split('=')[1]
    steps = chk_filename.stem.split('_')[-1].split('-')[2].split('=')[1]

    return int(epoch), int(steps)