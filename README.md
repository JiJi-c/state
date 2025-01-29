# Perturbation Sets

We use pert-bench for dataloaders and evaluation. See pert-bench for installation and more details.

## Model Options

PertBench provides several model architectures:

1. **GlobalSimpleSum**: Basic baseline that computes global control means and perturbation offsets.
   ```python
   model=globalsimplesum
   ```

2. **SimpleSum**: Maps between control and perturbed cells using specified strategy.
   ```python
   model=simplesum
   ```

3. **EmbedSum**: Projects perturbations and cell states into a shared latent space.
   ```python
   model=embedsum
   ```

4. **NeuralOT**: Model using GPT2 backbone and a distributional loss function.
   ```python
   model=neuralot
   ```

## Training Configuration

### Input/Output Spaces

PertBench supports two types of input/output configurations:

1. **Gene Expression Space**:
```yaml
data.kwargs.embed_key=None # Set the inputs and regression targets to be gene expression space
data.kwargs.output_space=gene # Set the output of the model to be gene expression space
```

2. **Latent Space** (recommended):
```yaml
data.kwargs.embed_key=X_uce  # or X_scGPT, 
data.kwargs.output_space=latent # Set the inputs and regression targets to the specified latent space
```

### Mapping Strategies

Available strategies for mapping perturbed cells to control cells:

1. **nearest**: Maps to closest control cell (training only)
```yaml
data.kwargs.basal_mapping_strategy=nearest
data.kwargs.k_neighbors=10
```

2. **batch**: Maps within same experimental batch
```yaml
data.kwargs.basal_mapping_strategy=batch
```

3. **random**: Random control cell mapping
```yaml
data.kwargs.basal_mapping_strategy=random
```

4. **pseudobulk**: Average over local neighborhood
```yaml
data.kwargs.basal_mapping_strategy=pseudobulk
data.kwargs.neighborhood_fraction=0.5  # Fraction of population for averaging higher means more aggressive pseudobulking
```

5. **pseudo_nearest**: For inference using learned offsets
```yaml
data.kwargs.basal_mapping_strategy=pseudo_nearest
```

### Task Specification

Specify training and testing tasks using dataset and cell type combinations:

```yaml
# Train on all cell types in replogle except jurkat
data.kwargs.train_task=replogle
data.kwargs.test_task=replogle_jurkat:zeroshot

# Train on multiple datasets
data.kwargs.train_task=replogle,jiang_K562
data.kwargs.test_task=replogle_rpe1:zeroshot,jiang_HepG2:fewshot
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
    data.kwargs.test_task=jiang_K562:zeroshot \
    data.kwargs.embed_key=X_uce \
    data.kwargs.output_space=latent \
    data.kwargs.basal_mapping_strategy=pseudobulk \
    data.kwargs.neighborhood_fraction=0.5 \
    data.kwargs.split_train_val_controls=True \
    model=neuralot \
    output_dir=/path/to/output \
    name=jiang_K562_zeroshot
```

2. Few-shot learning on k562:
```bash
python -m benchmark.train \
    data.kwargs.train_task=replogle \
    data.kwargs.test_task=replogle_k562:fewshot \
    data.kwargs.embed_key=X_uce \
    data.kwargs.output_space=latent \
    data.kwargs.basal_mapping_strategy=pseudobulk \
    data.kwargs.neighborhood_fraction=0.5 \
    model=neuralot \
    output_dir=/path/to/output \
    name=k562_fewshot
```

## Model Evaluation

The evaluation script allows testing different mapping strategies:

```bash
python -m benchmark.train.evaluate_model \
    --output_dir /path/to/experiment \
    --checkpoint last.ckpt \
    --map_type pseudobulk  # Can differ from training strategy
```

Here, `/path/to/experiment` should be `/path/to/output/name` where `/path/to/output` and `name` are from the training command. Evaluation automatically computes:
- MSE
- Pearson correlation
- Wasserstein distance
- MMD
- Cosine similarity
- DE gene overlap

## Best Practices

1. **Model Selection**:
   - Use latent space models with output_space="latent" for best performance

2. **Mapping Strategy**:
   - For training: pseudobulk with neighborhood_fraction=0.5 is recommended
   - For evaluation: use same strategy as training for fair comparison
   
3. **Data Organization**:
   - Keep one cell type per H5 file
   - Ensure consistent gene names across datasets
   - Include batch information in gem_group

4. **Validation**:
   - Use control cells from test set for test-time fine-tuning
   - Compare multiple mapping strategies during evaluation

## Common Issues

1. Missing control cells in test set:
   - Ensure control cells (with label "non-targeting") are present
   - Check gem_group assignments
   - Make sure each h5 file has a cell_type variable in obs

2. Memory issues:
   - Use data.kwargs.preload_data=True for faster training
   - Adjust batch_size based on available GPU memory
