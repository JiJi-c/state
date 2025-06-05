#!/bin/bash                                                                                                                                                                                                                             
#SBATCH --partition=gpu_batch_high_mem,gpu_high_mem,vci_gpu_priority,gpu_batch,gpu_20gb,gpu_40gb                                                                                                                                        
#SBATCH --nodes=1                                                                                                                                                                                                                       
#SBATCH --ntasks-per-node=1                                                                                                                                                                                                             
#SBATCH --cpus-per-task=8                                                                                                                                                                                                               
#SBATCH --mem=128GB                                                                                                                                                                                                                     
#SBATCH --gres=gpu:1                                                                                                                                                                                                                    
#SBATCH --time=1-00:00:00                                                                                                                                                                                                               
#SBATCH --array=1-4                         
#SBATCH --output=logs/replogle_llama_11645640_%a.out                                                                                                                                                                                    
#SBATCH --error=logs/replogle_llama_11645640_%a.err                                                                                                                                                                                     
#SBATCH --job-name=replogle_llama_11645640                                                                                                                                                                                           
                                                                                                                                                                                                                                        
# Define TOML config path for each fold - each fold holds out a different cell type                                                                                                                                                     
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then                                                                                                                                                                                                 
    CELLTYPE_PATH="/large_storage/ctc/ML/state_sets/replogle/hepg2"                                                                                                                                                                     
    CELLTYPE="hepg2"                                                                                                                                                                                                                    
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then                                                                                                                                                                                               
    CELLTYPE_PATH="/large_storage/ctc/ML/state_sets/replogle/jurkat"                                                                                                                                                                    
    CELLTYPE="jurkat"                                                                                                                                                                                                                   
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then                                                                                                                                                                                               
    CELLTYPE_PATH="/large_storage/ctc/ML/state_sets/replogle/k562"                                                                                                                                                                      
    CELLTYPE="k562"                                                                                                                                                                                                                     
else                                                                                                                                                                                                                                    
    CELLTYPE_PATH="/large_storage/ctc/ML/state_sets/replogle/rpe1"                                                                                                                                                                      
    CELLTYPE="rpe1"                                                                                                                                                                                                                     
fi                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                        
echo "Using TOML config: ${CELLTYPE_PATH}.toml"                                                                                                                                                                                         
                                                                                                                                                                                                                                        
state-sets sets train \
    "data.kwargs.toml_config_path=${CELLTYPE_PATH}.toml" \
    "data.kwargs.embed_key=X_vci_1.5.2_4" \
    "data.kwargs.basal_mapping_strategy=random" \
    "data.kwargs.output_space=gene" \
    "data.kwargs.num_workers=12" \
    "data.kwargs.batch_col=gem_group" \
    "data.kwargs.pert_col=gene" \
    "data.kwargs.cell_type_key=cell_type" \
    "data.kwargs.control_pert=non-targeting" \
    "model.kwargs.batch_encoder=true" \
    "training.max_steps=200000" \
    "training.val_freq=2000" \
    "training.ckpt_every_n_steps=4000" \
    "training.batch_size=64" \
    "training.lr=1e-4" \
    "model.kwargs.cell_set_len=64" \
    "wandb.tags=[architecture_search]" \
    "model=replogle_llama_11645640" \
    "output_dir=/large_storage/ctc/userspace/rohankshah/preprint/replogle_llama_11645640" \
    "name=${CELLTYPE}"