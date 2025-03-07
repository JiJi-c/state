#!/bin/bash

#SBATCH --job-name=evo2_emb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
##SBATCH --mem=200G
#SBATCH --time=0-05:00:00
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/emb/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --partition=gpu_batch,gpu_high_mem,gpu_batch_high_mem,preemptible
#SBATCH --exclude=GPU115A


## ls /large_storage/ctc/projects/vci/ref_genome -1 | xargs -I{} sbatch scripts/emb/evo2/slurm.sh {}

srun  python3 ./scripts/preprocess_scbasecamp.py inferEvo2 $1
