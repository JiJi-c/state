# Perturbation Sets

To install the environment, use the provided `.yml` file:

```
conda env create -f pertsets_environment.yml
conda activate pertsets
```

# Disclaimer

Some parts of the code can be messy. If something is confusing, just ask! Chances are we can improve the code.

# The machine learning task

We are primarily interested in cell type generalization of perturbation prediction. That is, can we predict the effects of perturbations that we have seen in our training data on unseen cell types, either in the zeroshot or fewshot setting.

<p align="center">
    <img src="assets/generalization_task.png" alt="Generalization Task" width="50%">
</p>

# Frequently Asked Questions

## Data Parameters

The current datasets are a bit scattered, but here are the key ideas:

1. `data.kwargs.embed_key`: This is the key in the `.obsm` of the h5 file that contains the embeddings. For example, `X_hvg` is the key for the highly variable genes embeddings. Other options include `X_uce`, `X_scGPT`, `X_scfound`, etc.

2. As a byproduct of data pre-processing, the embeddings for cells are separated by folder. So for example, in the slurm script below, to have access to key `X_scfound`, you would need to change all occurrences of `tahoe_45_ct_scgpt` to `tahoe_45_ct_scfound`.

3. To train on the full transcriptome (19k gene featurization of the cell), use `tahoe_45_ct_processed` and set `data.kwargs.embed_key=null`.

4. `data.kwargs.basal_mapping_strategy`: This is the strategy used to map perturbed cells to control cells. For our experiments right now, this should always be `random`.

5. `data.kwargs.output_space`: This is the space that the model is predicting. For our experiments right now, this should always be `gene`. Then, if `data.kwargs.embed_key` is not `X_hvg`, the model automatically trains a decoder to HVG space for you. Otherwise, if `data.kwargs.embed_key` is `X_hvg`, no decoder is trained.

6. To change the size of the model, change `model.kwargs.hidden_dim`.

7. To set tags automatically for organizing your wandb, use `wandb.tags`.

## Parameters you should always set constant

You should always keep the following defaults: `data.kwargs.should_yield_control_cells=True`, `data.kwargs.map_controls=True`, `model.kwargs.softplus=True`

# Example Commands

## Example Tahoe Training Command (5-fold cross validation on cell type)
```sh
#!/bin/bash
#SBATCH --partition=gpu_batch,vci_gpu_priority,gpu_high_mem,gpu_batch_high_mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --output=<YOUR OUTPUT LOG>
#SBATCH --error=<YOUR ERROR LOG>
#SBATCH --job-name=<YOUR JOB NAME>
#SBATCH --exclude=GPU115A ## this node is currently broken on chimera

# Define test tasks for each fold
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    TEST_TASKS="[tahoe_45_ct_scvi:SNU-423:fewshot,tahoe_45_ct_scvi:NCI-H661:fewshot,tahoe_45_ct_scvi:AN3 CA:fewshot,tahoe_45_ct_scvi:BT-474:fewshot,tahoe_45_ct_scvi:RKO:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    TEST_TASKS="[tahoe_45_ct_scvi:HCT15:fewshot,tahoe_45_ct_scvi:CHP-212:fewshot,tahoe_45_ct_scvi:HEC-1-A:fewshot,tahoe_45_ct_scvi:HS-578T:fewshot,tahoe_45_ct_scvi:J82:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    TEST_TASKS="[tahoe_45_ct_scvi:NCI-H1792:fewshot,tahoe_45_ct_scvi:SNU-1:fewshot,tahoe_45_ct_scvi:A-172:fewshot,tahoe_45_ct_scvi:H4:fewshot,tahoe_45_ct_scvi:HT-29:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    TEST_TASKS="[tahoe_45_ct_scvi:SK-MEL-2:fewshot,tahoe_45_ct_scvi:SW480:fewshot,tahoe_45_ct_scvi:CFPAC-1:fewshot,tahoe_45_ct_scvi:NCI-H23:fewshot,tahoe_45_ct_scvi:SHP-77:fewshot]"
else
    TEST_TASKS="[tahoe_45_ct_scvi:LS 180:fewshot,tahoe_45_ct_scvi:NCI-H1573:fewshot,tahoe_45_ct_scvi:C-33 A:fewshot,tahoe_45_ct_scvi:COLO 205:fewshot,tahoe_45_ct_scvi:RPMI-7951:fewshot]"
fi

python -m train \
    data.kwargs.data_dir="/large_storage/ctc/userspace/aadduri/datasets" \
    data.kwargs.train_task=tahoe_45_ct_scgpt \
    data.kwargs.test_task="$TEST_TASKS" \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.output_space=gene \
    data.kwargs.should_yield_control_cells=True \
    data.kwargs.num_workers=16 \
    data.kwargs.batch_col=sample \
    data.kwargs.pert_col=drugname_drugconc \
    data.kwargs.cell_type_key=cell_name \
    data.kwargs.control_pert=DMSO_TF \
    data.kwargs.map_controls=True \
    training.max_steps=82000\
    training.val_freq=10000 \
    training.ckpt_every_n_steps=4000 \
    training.batch_size=32 \
    model.kwargs.cell_set_len=512 \
    wandb.tags="[feb28_conc,cv5,tahoe,fold${SLURM_ARRAY_TASK_ID},batched]" \
    model=pertsets\
    output_dir=<YOUR OUTPUT DIRECTORY> \
    name="fold${SLURM_ARRAY_TASK_ID}"
```

This will create five folders, one for each cell-type split of Tahoe, in the output directory you specify. The names will be `fold1`, etc.

## Example Replogle Training Command (1-fold cross validation on cell type)
```sh
#!/bin/bash
#SBATCH --partition=gpu_batch,vci_gpu_priority,gpu_high_mem,gpu_batch_high_mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-4
#SBATCH --output=<YOUR OUTPUT LOG>
#SBATCH --error=<YOUR ERROR LOG>
#SBATCH --job-name=<YOUR JOB NAME>
#SBATCH --exclude=GPU115A

# Define test tasks for each fold - each fold holds out a different cell type
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    TEST_TASKS="[replogle:hepg2:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    TEST_TASKS="[replogle:jurkat:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    TEST_TASKS="[replogle:k562:fewshot]"
else
    TEST_TASKS="[replogle:rpe1:fewshot]"
fi

python -m train \
    data.kwargs.data_dir="/large_storage/ctc/userspace/aadduri/datasets/filter" \
    data.kwargs.train_task=replogle \
    data.kwargs.test_task="$TEST_TASKS" \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.output_space=gene \
    data.kwargs.should_yield_control_cells=True \
    data.kwargs.num_workers=16 \
    data.kwargs.batch_col=gem_group \
    data.kwargs.pert_col=gene \
    data.kwargs.cell_type_key=cell_type \
    data.kwargs.control_pert=non-targeting \
    data.kwargs.map_controls=True \
    training.max_steps=20000 \
    training.val_freq=5000 \
    training.ckpt_every_n_steps=4000 \
    training.batch_size=256 \
    model.kwargs.cell_set_len=32 \
    model.kwargs.softplus=True \
    wandb.tags="[replogle,replogle4,hvg,fold${SLURM_ARRAY_TASK_ID}]" \
    model=pertsets \
    output_dir=<YOUR OUTPUT FOLDER> \
    name="fold${SLURM_ARRAY_TASK_ID}" \
    model.kwargs.hidden_dim=256 # model size 20M recommended for replogle
```

# Example Evaluation Script

```
#!/bin/bash
#SBATCH --partition=gpu_batch_high_mem,preemptible,gpu_high_mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --output=<YOUR_OUTPUT_LOG>_%a.out
#SBATCH --error=<YOUR_ERROR_LOG>_%a.err
#SBATCH --job-name=eval_hvg_samp_ctrl_50m
#SBATCH --exclude=GPU115A

EVAL_MAP_TYPE="random"

# This is the output folder from your job + the name
EXPERIMENT_DIR="/large_storage/ctc/userspace/aadduri/feb28_conc/cv_hvg_samp_ctrl/fold$SLURM_ARRAY_TASK_ID"

# make sure that ${EXPERIMENT_DIR}/last.ckpt is actually the most up to date step
CKPT="last.ckpt"

# Print job info for logging
echo "Running job $SLURM_ARRAY_TASK_ID"
echo "Fold $SLURM_ARRAY_TASK_ID"
echo "Checking for evaluation directory: $EXPERIMENT_DIR"

python -m train.evaluate_model \
        --output_dir ${EXPERIMENT_DIR} \
        --checkpoint ${CKPT} \
        --map_type ${EVAL_MAP_TYPE}
```

The resulting metrics will connect to wandb if you have that set up. They will also be printed out to your error file.

## Model Options

PertBench provides several model architectures:

1. **PertSets**: A set-based transformer model for cellular perturbation predictions.
   ```python
   model=pertsets
   ```
2. **GlobalSimpleSum**: Basic baseline that computes global control means and perturbation offsets.
   ```python
   model=globalsimplesum
   ```

3. **SimpleSum**: Maps between control and perturbed cells using specified strategy.
   ```python
   model=simplesum
   ```

4. **EmbedSum**: Projects perturbations and cell states into a shared latent space.
   ```python
   model=embedsum
   ```

5. **Old NeuralOT**: Model using GPT2 backbone and a distributional loss function.
   ```python
   model=old_neuralot
   ```

## Training Configuration

### Input/Output Spaces

PertBench supports two types of input/output configurations:

1. **Gene Expression Space**:
```yaml
data.kwargs.embed_key=X_hvg # Set the inputs and regression targets to be gene expression space
data.kwargs.output_space=gene # Set the output of the model to be gene expression space
```

2. **Latent Space** (recommended):
```yaml
data.kwargs.embed_key=X_uce  # or X_scGPT, X_uce, X_scfound, etc
data.kwargs.output_space=gene # Set the inputs and regression targets to the specified latent space
```

3. **Dataset Specific Keys** (example with Tahoe):
```yaml
data.kwargs.batch_col=sample \
data.kwargs.pert_col=drugname_drugconc \
data.kwargs.cell_type_key=cell_name \
data.kwargs.control_pert=DMSO_TF \
```

### Mapping Strategies

Available strategies for mapping perturbed cells to control cells:

1. **batch**: Maps within same experimental batch
```yaml
data.kwargs.basal_mapping_strategy=batch
```

2. **random**: Random control cell mapping
```yaml
data.kwargs.basal_mapping_strategy=random
```

3. **pseudobulk**: Average over local neighborhood
```yaml
data.kwargs.basal_mapping_strategy=pseudobulk
data.kwargs.neighborhood_fraction=0.5  # Fraction of population for averaging higher means more aggressive pseudobulking
```

It is recommended to use this mapping strategy with control cells as well, e.g., `data.kwargs.map_controls=True`.

### Task Specification

Specify training and testing tasks using dataset and cell type combinations:

```yaml
# Train on all cell types in replogle except jurkat
data.kwargs.train_task=replogle
data.kwargs.test_task=replogle:jurkat:zeroshot

# Train on multiple datasets
data.kwargs.train_task=replogle,jiang_K562
data.kwargs.test_task=replogle:rpe1:zeroshot,jiang_HepG2:fewshot
```

For few-shot learning, specify the percentage of test data to use in training:
```yaml
data.kwargs.few_shot_percent=0.3  # Use 30% for training
```

## Example Training Commands

1. Zero-shot learning on jurkat:
```bash
python -m benchmark.train \
    data.kwargs.train_task=replogle,jiang \
    data.kwargs.test_task=jiang:K562:zeroshot \
    data.kwargs.embed_key=X_uce \
    data.kwargs.output_space=gene \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.neighborhood_fraction=0.5 \
    data.kwargs.split_train_val_controls=True \
    model=pertsets \
    output_dir=/path/to/output \
    name=jiang_K562_zeroshot
```

2. Few-shot learning on k562:
```bash
python -m benchmark.train \
    data.kwargs.train_task=replogle \
    data.kwargs.test_task=replogle:k562:fewshot \
    data.kwargs.embed_key=X_uce \
    data.kwargs.output_space=gene \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.neighborhood_fraction=0.5 \
    model=pertsets \
    output_dir=/path/to/output \
    name=k562_fewshot
```

## Model Evaluation

The evaluation script allows testing different mapping strategies:

```bash
python -m benchmark.train.evaluate_model \
    --output_dir /path/to/experiment \
    --checkpoint last.ckpt \
    --map_type random  # Can differ from training strategy
```

Here, `/path/to/experiment` should be `/path/to/output/name` where `/path/to/output` and `name` are from the training command. Evaluation automatically computes:
- Pearson correlation
- Cosine similarity
- DE gene overlap

## Best Practices

1. **Model Selection**:
   - Use latent space models with output_space="gene" for best performance

2. **Mapping Strategy**:
   - For training: random is recommended for large datasets
   - For evaluation: use same strategy as training for fair comparison
   
3. **Data Organization**:
   - Ensure consistent gene names across datasets

## Common Issues

1. Missing control cells in test set:
   - Ensure control cells are present

2. Make sure that your `data.kwargs.embed_key` appears in the `.obsm` of your h5 file.
