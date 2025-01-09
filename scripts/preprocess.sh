#!/bin/bash

#SBATCH --job-name=vci-preprocessing
#SBATCH --nodes=1
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/preprocess.log
#SBATCH --open-mode=append
#SBATCH --account=ctc

python3 ./scripts/processing.py

# Prep for unit test
# python3 ./scripts/processing.py \
#     --data_path    /tmp/data_for_test \
#     --destination  /home/rajesh.ilango/Projects/vci/pert-bench/tests/data/inference \
#     --summary_file /home/rajesh.ilango/Projects/vci/pert-bench/tests/data/inference/summary.csv \
#     --emb_idx_file /home/rajesh.ilango/Projects/vci/pert-bench/tests/data/inference/test_embidx_mapping.torch \
#     --species human
