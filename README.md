# Perturbation Sets

To install the environment, use the provided `.yml` file:

```
conda env create -f pertsets_environment.yml
conda activate pertsets
```

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
    TEST_TASKS="[tahoe_45_ct_scgpt:RKO:fewshot,tahoe_45_ct_scgpt:SNU-423:fewshot,tahoe_45_ct_scgpt:BT-474:fewshot,tahoe_45_ct_scgpt:AN3 CA:fewshot,tahoe_45_ct_scgpt:NCI-H661:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    TEST_TASKS="[tahoe_45_ct_scgpt:HEC-1-A:fewshot,tahoe_45_ct_scgpt:HCT15:fewshot,tahoe_45_ct_scgpt:HS-578T:fewshot,tahoe_45_ct_scgpt:J82:fewshot,tahoe_45_ct_scgpt:CHP-212:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    TEST_TASKS="[tahoe_45_ct_scgpt:A-172:fewshot,tahoe_45_ct_scgpt:NCI-H1792:fewshot,tahoe_45_ct_scgpt:H4:fewshot,tahoe_45_ct_scgpt:HT-29:fewshot,tahoe_45_ct_scgpt:SNU-1:fewshot]"
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    TEST_TASKS="[tahoe_45_ct_scgpt:CFPAC-1:fewshot,tahoe_45_ct_scgpt:SK-MEL-2:fewshot,tahoe_45_ct_scgpt:NCI-H23:fewshot,tahoe_45_ct_scgpt:SHP-77:fewshot,tahoe_45_ct_scgpt:SW480:fewshot]"
else
    TEST_TASKS="[tahoe_45_ct_scgpt:C-33 A:fewshot,tahoe_45_ct_scgpt:LS 180:fewshot,tahoe_45_ct_scgpt:RPMI-7951:fewshot,tahoe_45_ct_scgpt:COLO 205:fewshot,tahoe_45_ct_scgpt:NCI-H1573:fewshot]"
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
    model.kwargs.transformer_backbone_kwargs.n_embd=256 \
    model.kwargs.hidden_dim=256 # model size 20M recommended for replogle
```

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
