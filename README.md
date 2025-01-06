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
Training data needs to be preprocessed for:
- Filtering cells and genes with less than minimum count.
- Supported datastructure for X. Current version does not support CSC matrix due to training time runtime performance.
- Create a summary/metadata files. This file is input to dataloader to improve training startup time.

To run preprocessing:
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

This creates the training config and the slurm script to start the training in ./output directory. This slurm script can be used for restarting training.

Logs for the current run can be found at `outputs/<<exp_name>>/training.log`.
Checkpoints and wandb names will also carry the `exp_name`.

One can override default values in `conf/defaults.yml` in the command. To override the dataset path, please use the following command:
```
python3 -m vci.tools.slurm \
    --exp_name vci_medium_dataset \
    --set dataset.path=/checkpoint/ctc/ML/uce/h5ad_train_dataset_100.csv \
          datasest.name=<<dataset name>>
```
As mentioned before the script records the slurm script at `./output/<<exp_name>>` direectory.

To create a record of all jobs in progress and to reproduce config file one can create bash script such as ./script/pretrain.sh. At some point default.conf must considered immutable, except when new properties are added.