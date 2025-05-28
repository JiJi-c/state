#!/bin/bash

#SBATCH --job-name=esm2_emb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=10
##SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/emb/esm2/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --partition=gpu_batch,gpu_high_mem,gpu_batch_high_mem
#SBATCH --exclude=GPU115A

unset SLURM_CPUS_PER_TASK
## ls /scratch/ctc/ML/uce/scBasecamp -1 | xargs -I{} sbatch scripts/emb/esm2/slurm.sh {}
# srun python3 ./scripts/preprocess_scbasecamp.py resolve_gene_symbols --species_dir $1


## ls /large_storage/ctc/projects/vci/ref_genome -1 | xargs -I{} sbatch scripts/emb/esm2/slurm.sh {}
srun  python3 ./scripts/preprocess_scbasecamp.py inferESM2 $1
# srun python3 ./scripts/preprocess_scbasecamp.py merge_all_species --model ESM2_3B $1
# python3 ./scripts/preprocess_scbasecamp.py dataset_embedding_mapping_by_species --emb_model ESM2_3B
# python3 ./scripts/preprocess_scbasecamp.py consolidate --emb_model ESM2_3B


