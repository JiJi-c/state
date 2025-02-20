#!/bin/bash

python3 -m vci.tools.slurm \
    --exp_name vci_repro_flash_attn_issue \
    --set dataset.name=vci \
    --set model.backbone=vgg16 \
    --set experiment.num_gpus_per_node=2 \
    --set wandb.enable=false \
    --set val_check_interval=50 \
    --set validations.diff_exp.eval_interval_multiple=1




python3 -m vci.tools.slurm \
    --exp_name vci_for_new_dataset \
    --set dataset.name=vci \
    --set experiment.num_gpus_per_node=2 \
    --set val_check_interval=50 \
    --set validations.diff_exp.eval_interval_multiple=1 \
    --set wandb.enable=false \
    --set dataset.train=/scratch/ctc/ML/uce/rpe1_pert.csv \
    --set dataset.val=/scratch/ctc/ML/uce/rpe1_pert.csv