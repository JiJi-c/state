import numpy as np
import pandas as pd


full_dataset = '/checkpoint/ctc/ML/uce/h5ad_dataset.csv'
train_dataset = '/checkpoint/ctc/ML/uce/h5ad_train_dataset.csv'
test_dataset = '/checkpoint/ctc/ML/uce/h5ad_test_dataset.csv'

df = pd.read_csv(full_dataset)
msk = np.random.rand(df.shape[0]) < 0.9

df_train = df[msk]
df_test = df[~msk]

df_train.reset_index(drop=True, inplace=True)
df_train.to_csv(train_dataset)

df_test.reset_index(drop=True, inplace=True)
df_test.to_csv(test_dataset)
