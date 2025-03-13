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
    --exp_name vci_mse_20250225 \
    --set dataset.name=vci \
    --set experiment.checkpoint.path /scratch/ctc/ML/vci/checkpoint/pretrain/20250225 \
    --set experiment.val_check_interval=1000 \
    --set loss.name=mse



python3 -m vci.tools.slurm \
    --exp_name vci_scbasecamp \
    -n 1 -g 1 \
    --set dataset.name=vci \
    optimizer.gradient_accumulation_steps=100 \
    experiment.val_check_interval=1000 \
    validations.diff_exp.eval_interval_multiple=1 \
    embeddings.current=evo2-scbasecamp
