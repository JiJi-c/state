import argparse
import subprocess
from jinja2 import Template

sbatch_script_template = """#!/bin/bash

#SBATCH --job-name={{ exp_name }}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --gres=gpu:{{ num_gpus_per_node }}
#SBATCH --ntasks-per-node={{ num_gpus_per_node }}
#SBATCH --partition=gpu_batch
##SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/{{ exp_name }}.log
#SBATCH --open-mode=append
#SBATCH --account=vci

GPUS_PER_NODE={{ num_gpus_per_node }}

# Get the names of all nodes involved in training.
scontrol show hostname ${SLURM_JOB_NODELIST} > hostfile
sed -i "s/$/ slots=${GPUS_PER_NODE}/" hostfile

MASTER_ADDR=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n 1)
MASTER_PORT='12355'

NCCL_DEBUG=INFO
PYTHONFAULTHANDLER=1

# WAR for an issue with SLURM version and Pytorch lightning
SLURM_NTASKS_PER_NODE={{ num_gpus_per_node }}

srun python -m vci.train \\
    experiment.master=${MASTER_ADDR} \\
    experiment.port=${MASTER_PORT} {{ exp_overrides }}
"""

def parse_vars(extra_vars):
    """
     Parses comma seperated key value pair strings into dict.
    """
    vars_list = []
    if extra_vars:
        for i in extra_vars:
            items = i.split('=')
            key = items[0].strip()
            if len(items) > 1:
                value = '='.join(items[1:])
                vars_list.append((key, value))
    return dict(vars_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create dataset list CSV file"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        default='vci_training',
        help="Experiment name. This will be used to name generated artifacts.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use for this training job.",
    )
    parser.add_argument(
        "--gpus_per_nodes",
        type=int,
        default=4,
        help="Number of GPUs per node",
    )
    parser.add_argument(
        "-d", "--dryrun",
        dest='dryrun',
        action='store_true',
        default=False,
        help="Only generate slurm sbatch script",
    )
    parser.add_argument(
        "-t", "--tail",
        dest='tail',
        action='store_true',
        default=False,
        help="Tails the log file. Killing will not kill the training.",
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        nargs='+',
        default=None,
        help="Values to be overriden for the training."
             "Please refer ./conf/defaults.yaml")

    args = parser.parse_args()

    bind_param = {
        "exp_name": args.exp_name,
        "num_nodes": args.num_nodes,
        "num_gpus_per_node": args.gpus_per_nodes,
    }

    overrides =  {
        "experiment.name": args.exp_name,
        "experiment.num_nodes": args.num_nodes,
        "experiment.num_gpus_per_node": args.gpus_per_nodes,
    }
    overrides.update(parse_vars(args.set))
    exp_overrides = ''
    for key, value in overrides.items():
        exp_overrides = exp_overrides + f'\\\n\t{key}={value} '

    bind_param['exp_overrides'] = exp_overrides

    template = Template(sbatch_script_template)
    rendered_script = template.render(bind_param)

    slurm_script = f"vci_job_{args.exp_name}.slurm"
    with open(slurm_script, "w") as f:
        f.write(rendered_script)

    if not args.dryrun:
        subprocess.call(['sbatch', slurm_script])
        if args.tail:
            subprocess.call(['tail', '-f', f'outputs/{args.exp_name}.log'])
