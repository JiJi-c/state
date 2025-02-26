#!/bin/bash

# python3 -m vci.tools.slurm \
#     --exp_name vci_repro_flash_attn_issue \
#     --set dataset.name=vci \
#     --set model.backbone=vgg16 \
#     --set experiment.num_gpus_per_node=2 \
#     --set wandb.enable=false \
#     --set val_check_interval=50 \
#     --set validations.diff_exp.eval_interval_multiple=1


python3 -m vci.tools.slurm \
    --exp_name vci_multinode_2_1 \
    -n 2 -g 1 -p preemptible \
    --set dataset.name=vci \
    --set optimizer.gradient_accumulation_steps=100 \
    --set val_check_interval=50 \
    --set validations.diff_exp.eval_interval_multiple=1 \
    --set wandb.enable=false \
    --set dataset.train=/scratch/ctc/ML/uce/rpe1_pert.csv \
    --set dataset.val=/scratch/ctc/ML/uce/rpe1_pert.csv
