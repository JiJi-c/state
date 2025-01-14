#!/bin/bash

EMB_SIZE=2048

python3 -m vci.tools.slurm \
    --exp_name vci_${EMB_SIZE} \
    --set dataset.name=vci \
          dataset.data_dir=/large_experiments/goodarzilab/mohsen/cellxgene/processed/ \
          dataset.val=/scratch/ctc/ML/uce/h5ad_val_dataset.csv \
          dataset.train=/scratch/ctc/ML/uce/h5ad_dataset.csv \
          experiment.checkpoint.path=/scratch/ctc/ML/vci/checkpoint/pretrain \
          experiment.num_epochs=5 \
          model.name=vci_${EMB_SIZE} \
          model.emsize=${EMB_SIZE} \
          model.output_dim=${EMB_SIZE}
