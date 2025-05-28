#!/bin/env python3

import numpy as np
import pandas as pd


full_dataset = '/large_storage/ctc/public/dataset/vci/h5ad_dataset.csv'
val_dataset = '/large_storage/ctc/public/dataset/vci/h5ad_val_dataset.csv'
train_dataset = '/large_storage/ctc/public/dataset/vci/h5ad_train_dataset.csv'

df = pd.read_csv(full_dataset)
msk = np.random.rand(df.shape[0]) < 0.95

df_train = df[msk]
df_val = df[~msk]

df_train.reset_index(drop=True, inplace=True)
df_train.to_csv(train_dataset)

df_val.reset_index(drop=True, inplace=True)
df_val.to_csv(val_dataset)
