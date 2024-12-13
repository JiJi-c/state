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

##cd /home/rajesh.ilango/Projects/github/vci_starter

python3 ./scripts/processing.py
