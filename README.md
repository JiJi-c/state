# VCI Starter Code

Using training code for the UCE model as starter code for building out new VCI model


# Getting started
## Creating Conda environment for developement and training
```
conda env create --name vci --file=environment.yml
```

To apply a change make to environment.yml please execute the following commands
```
conda activate vci
conda env update --file=environment.yml --prune
```

## Data preprocessing
Data preprocessing scripts are at `scripts/`. To run preprocessing:

```
sbatch scripts/preprocess.sh
```

## Start training
Please make sure wandb key is set.
```
export WANDB_API_KEY=<<YOUR WANDB KEY>>
```

Before starting please review `train.sh` and `conf/defaults.yml`. Ensure the number of
nodes is correct.

To start training
```
python3 -m vci.tools.slurm --exp_name <<EXP_NUM_1>>
```

Logs for the current run can be found at `outputs/<<exp_name>>.log.
Checkpoints and wandb names will also carry the `exp_name`.

To submit and tail the log file, please use `-t` option. For e.g.
```
python3 -m vci.tools.slurm -t --exp_name <<EXP_NUM_1>>
```

One can override default values at `conf/defaults.yml` in the command. To override the dataset path, please use the following command:
```
python3 -m vci.tools.slurm \
    --exp_name vci_medium_dataset \
    --set dataset.path=/checkpoint/ctc/ML/uce/h5ad_train_dataset_100.csv \
          datasest.name=<<dataset name>>
```

The script records the slurm script in the local dir before submitting the sbatch job.