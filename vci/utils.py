import os
import glob


def get_latest_checkpoint(cfg):
    run_name = "exp_{0}_layers_{1}_dmodel_{2}_samples_{3}_max_lr_{4}_op_dim_{5}".format(
        cfg.experiment.name,
        cfg.model.nlayers,
        cfg.model.emsize,
        cfg.dataset.pad_length,
        cfg.optimizer.max_lr,
        cfg.model.output_dim)

    chks = glob.glob(os.path.join(cfg.experiment.checkpoint.path, f'{run_name}*'))

    chk = None
    if chks:
        chk = sorted(chks)[-1]

    return run_name, chk

