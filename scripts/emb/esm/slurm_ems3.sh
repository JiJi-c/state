#!/bin/bash

#SBATCH --job-name=esm3_emb
##SBATCH --nodes=1
##SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/emb/esm3/%j.log
#SBATCH --open-mode=append
#SBATCH --partition=cpu_batch,cpu_high_mem,cpu_batch_high_mem,preemptible
#SBATCH --exclude=GPU115A

unset SLURM_GPUS
unset SLURM_CPUS_PER_TASK

num_tokens=2
export ESM_API_TOKEN=$(cat ~/.esm_token_$(($1 % num_tokens)))

## ls /large_storage/ctc/projects/vci/ref_genome -1 | xargs -I{} sbatch scripts/emb/esm2/slurm.sh {}

# srun python3 ./scripts/preprocess_scbasecamp.py \
#     dataset_embedding_mapping_by_species --emb_model ESM2 --species_dirs $1

## ls /scratch/ctc/ML/uce/scBasecamp -1 | xargs -I{} sbatch scripts/emb/esm/slurm_ems3.sh {}
## ls /scratch/ctc/ML/uce/scBasecamp -1 | awk '{print NR, $0}' | xargs -n 2 sbatch scripts/emb/esm/slurm_ems3.sh
echo "sbatch scripts/emb/esm/slurm_ems3.sh $@"
srun  python3 ./scripts/preprocess_scbasecamp.py inferESM3 --species $2

## ls /scratch/ctc/ML/uce/scBasecamp -1 | xargs -I{} sbatch scripts/emb/esm2/slurm.sh {}
# srun python3 ./scripts/preprocess_scbasecamp.py resolve_gene_symbols --species_dir $1