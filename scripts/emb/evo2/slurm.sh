#!/bin/bash

#SBATCH --job-name=evo2_emb
#SBATCH --nodes=1
##SBATCH --gres=cpu:10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=0-05:00:00
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/emb/evo2/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --partition=gpu_batch,gpu_high_mem,gpu_batch_high_mem,preemptible
#SBATCH --exclude=GPU115A


## ls /large_storage/ctc/projects/vci/ref_genome -1 | xargs -I{} sbatch scripts/emb/evo2/slurm.sh {}

# df.shape
# (30138, 5)
# srun  python3 ./scripts/preprocess_scbasecamp.py inferEvo2 $1
# srun python3 ./scripts/preprocess_scbasecamp.py dataset_embedding_mapping \
#     --emb_model Evo2 --start $1 --end $2

srun python3 ./scripts/preprocess_scbasecamp.py dataset_embedding_mapping_by_species --species_dirs $1