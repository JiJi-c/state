#!/bin/bash

#SBATCH --job-name=esm2_3b_emb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/emb/esm2_3B/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --partition=gpu_batch,gpu_high_mem,gpu_batch_high_mem,preemptible,vci_gpu_priority
#SBATCH --exclude=GPU115A

unset SLURM_GPUS
unset SLURM_CPUS_PER_TASK
#echo "$(set)"

export ESM_API_TOKEN=$(cat ~/.esm_token)

## ls /large_storage/ctc/projects/vci/ref_genome -1 | xargs -I{} sbatch scripts/emb/esm2/slurm.sh {}

# srun python3 ./scripts/preprocess_scbasecamp.py \
#     dataset_embedding_mapping_by_species --emb_model ESM2 --species_dirs $1

## ls /scratch/ctc/ML/uce/scBasecamp -1 | xargs -I{} sbatch scripts/emb/esm/slurm_esm2.sh {}
echo "sbatch scripts/emb/esm/slurm_esm2.sh $1"
srun  python3 ./scripts/preprocess_scbasecamp.py inferESM2 --species $1


## ls /scratch/ctc/ML/uce/scBasecamp -1 | xargs -I{} sbatch scripts/emb/esm2/slurm.sh {}
# srun python3 ./scripts/preprocess_scbasecamp.py resolve_gene_symbols --species_dir $1
