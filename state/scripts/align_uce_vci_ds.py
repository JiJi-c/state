import pandas as pd


uce_ds = pd.read_csv('/scratch/ctc/ML/uce/full_train_datasets.csv')
vci_train_ds = pd.read_csv('/scratch/ctc/ML/uce/h5ad_train_dataset.csv')
vci_val_ds = pd.read_csv('/scratch/ctc/ML/uce/h5ad_val_dataset.csv')


valid_ds = set(vci_train_ds.names).intersection(set(uce_ds.names))

vci_train_ds = vci_train_ds[vci_train_ds.names.isin(valid_ds)]
vci_train_ds.to_csv('/scratch/ctc/ML/uce/h5ad_train_uce_filtered.csv')

vci_train_ds['census'] = uce_ds['census']


vci_val_ds = vci_val_ds[vci_val_ds.names.isin(valid_ds)]
vci_val_ds.to_csv('/scratch/ctc/ML/uce/h5ad_val_uce_filtered.csv')
