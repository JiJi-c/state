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
    --exp_name vci_scbase_esm2_human_all \
    -n 1 -g 1 \
    --set embeddings.current=esm2-scbasecamp \
          dataset.current=scbasecamp \
          dataset.scbasecamp.filter_by_species=Homo_sapiens


python3 -m vci.tools.slurm \
    --exp_name vci_scbase_esm2_human_all_bce \
    -n 1 -g 1 \
    --set embeddings.current=esm2-scbasecamp \
          dataset.current=scbasecamp \
          dataset.scbasecamp.filter_by_species=Homo_sapiens \
          loss.name=cross_entropy




python3 -m vci.tools.slurm \
    --exp_name vci_scbasecamp_human \
    -n 1 -g 1 \
    --set validations.diff_exp.eval_interval_multiple=1 \
          embeddings.current=evo2-scbasecamp \
          dataset.current=scbasecamp \
          dataset.scbasecamp.filter_by_species=Homo_sapiens


python3 -m vci.tools.slurm \
    --exp_name vci_2048-1024_1024_16_4 \
    -n 1 -g 1 \
    --set embeddings.current=esm2-cellxgene \
          dataset.current=cellxgene \
          model.d_hid=2048 \
          model.emsize=1024 \
          model.output_dim=1024 \
          model.nhead=16 \
          model.nlayers=4 \
          model.rda=true\
          loss.name=mse