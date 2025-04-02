#!/bin/bash

#SBATCH --job-name=evo2_emb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
##SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/emb/evo2/%j.log
#SBATCH --open-mode=append
#SBATCH --partition=gpu_batch,gpu_high_mem,gpu_batch_high_mem,preemptible,vci_gpu_priority
#SBATCH --exclude=GPU115A

## ls /large_storage/ctc/projects/vci/ref_genome -1 | xargs -I{} sbatch scripts/emb/evo2/slurm.sh {}

# sbatch ./scripts/emb/evo2/slurm.sh Bos_taurus
# sbatch ./scripts/emb/evo2/slurm.sh Caenorhabditis_elegans
# sbatch ./scripts/emb/evo2/slurm.sh Callithrix_jacchus
# sbatch ./scripts/emb/evo2/slurm.sh Danio_rerio
# sbatch ./scripts/emb/evo2/slurm.sh Drosophila_melanogaster
# sbatch ./scripts/emb/evo2/slurm.sh Equus_caballus
# sbatch ./scripts/emb/evo2/slurm.sh Gallus_gallus
# sbatch ./scripts/emb/evo2/slurm.sh Gorilla_gorilla
# sbatch ./scripts/emb/evo2/slurm.sh Heterocephalus_glaber
# sbatch ./scripts/emb/evo2/slurm.sh Homo_sapiens
# sbatch ./scripts/emb/evo2/slurm.sh Macaca_mulatta
# sbatch ./scripts/emb/evo2/slurm.sh Mus_musculus
# sbatch ./scripts/emb/evo2/slurm.sh Oryctolagus_cuniculus
# sbatch ./scripts/emb/evo2/slurm.sh Ovis_aries
# sbatch ./scripts/emb/evo2/slurm.sh Pan_troglodytes
# sbatch ./scripts/emb/evo2/slurm.sh Schistosoma_mansoni
# sbatch ./scripts/emb/evo2/slurm.sh Sus_scrofa
#
# srun python3 ./scripts/preprocess_scbasecamp.py create_genelist --ref_genome $1
# srun python3 ./scripts/preprocess_scbasecamp.py create_gene_seq_mapping --species $1
# srun python3 ./scripts/preprocess_scbasecamp.py inferEvo2 $1
# srun python3 ./scripts/preprocess_scbasecamp.py dataset_embedding_mapping --emb_model Evo2 --start $1 --end $2

# srun python3 ./scripts/preprocess_scbasecamp.py dataset_embedding_mapping_by_species --species_dirs $1

# srun python3 ./scripts/preprocess_scbasecamp.py resolve_gene_symbols --species_dir  $1
