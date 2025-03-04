#!/bin/bash

python3 -m vci.tools.slurm \
    --exp_name vci_mse_20250225 \
    --set dataset.name=vci \
    --set experiment.checkpoint.path /scratch/ctc/ML/vci/checkpoint/pretrain/20250225 \
    --set experiment.val_check_interval=1000 \
    --set loss.name=mse




python3 -m vci.tools.slurm \
    --exp_name vci_for_new_dataset \
    --set dataset.name=vci \
    --set experiment.num_gpus_per_node=2 \
    --set val_check_interval=50 \
    --set validations.diff_exp.eval_interval_multiple=1 \
    --set dataset.train=/scratch/ctc/ML/uce/rpe1_pert.csv \
    --set dataset.val=/scratch/ctc/ML/uce/rpe1_pert.csv